import csv
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import munch
import numpy as np
import pandas as pd
import pyproj as proj
from scipy.optimize import curve_fit
import skgstat as skg
from skgstat import models
import toml
import verde as vd

import llh2local # matlab script for conversion to local coords, pyproj preferred

def construct_variogram(config):
    
    # use toml config for easier management of parameters/args
    config = munch.munchify(toml.load(config))
    filepath = config.data.frame # insar frame defines region and resolution
    reliable_pixel_screen = config.filters.rel_pix # use reliable pixels?
    N_bins = config.variogram.bins # number of even bins for semivariogram
    lat_res = config.variogram.lat_res
    lng_res = config.variogram.lng_res
    plot_save_path = config.variogram.plot_save_path


    f = h5.File(filepath, 'r')

    vel = f['Velocity']
    vel_masked = np.ma.masked_array(vel, mask=np.isnan(vel))
    rows, cols = vel.shape[0], vel.shape[1]
    print(f'array rows and cols are {rows},{cols}')

    lat = f['Latitude']
    lng = f['Longitude']

    # np.ma.masked_array takes True for all invalid values, if we want reliable pixels we need the unreliable pixel = True (i.e. invert reliable mask before applying within masked array
    if reliable_pixel_screen:
        rel_pix_mask = f['Reliable_Pixels']
        lat_ma = np.ma.masked_array(lat, mask=~np.array(rel_pix_mask))
        lng_ma = np.ma.masked_array(lng, mask=~np.array(rel_pix_mask))

        lat_min = np.ma.min(lat_ma)
        print(f'Lat min is {lat_min}')
        lng_min = np.ma.min(lng_ma)
        print(f'Long min is {lng_min}')
        lat_max = np.ma.max(lat_ma)
        print(f'Lat max is {lat_max}')
        lng_max = np.ma.max(lng_ma)
        print(f'Long max is {lng_max}')

        vel_ma = np.ma.masked_array(vel, mask=~np.array(rel_pix_mask))
        print(vel_ma)

    else:
        lat_min = np.nanmin(lat)
        print(f'Lat min is {lat_min}')
        lng_min = np.nanmin(lng)
        print(f'Long min is {lng_min}')
        lat_max = np.nanmax(lat)
        print(f'Lat max is {lat_max}')
        lng_max = np.nanmax(lng)
        print(f'Long max is {lng_max}')

        vel = np.ma.masked_array(vel, mask=np.isnan(vel))

    lat_minmax_diff = lat_max - lat_min
    lng_minmax_diff = lng_max - lng_min
    lat_strides = np.int32(lat_minmax_diff // lat_res)
    print(f'Lat strides are {lat_strides}')
    lng_strides = np.int32(lng_minmax_diff // lng_res)
    print(f'Lng strides are {lng_strides}')

    with open('variogram_params_'+filepath.split('/')[-1]+'.csv', 'w') as csvfile:
        fieldnames = ['lat_start', 'lat_end', 'lng_start', 'lng_end', 'rmse', 'effective_range', 'partial_sill', 'sill', 'nugget', 'a', 'b']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    detrended_variogram_values = []
    detrended_covariance_values = []

    for lat_idx in range(lat_strides):
        for lng_idx in range(lng_strides):

            lat_start = lat_min + (lat_idx * lat_res)
            lat_end = lat_start + lat_res
            print(f'lat start/end is {lat_start}, {lat_end}')

            lng_start = lng_min + (lng_idx * lng_res)
            lng_end = lng_start + lng_res
            print(f'lng start/end is {lng_start}, {lng_end}')

            if reliable_pixel_screen:
                clip_mask = ((lat_ma > lat_start) & (lat_ma < lat_end)) & ((lng_ma > lng_start) & (lng_ma < lng_end))
                print(clip_mask.shape)
                rel_w_clip_mask = np.ma.mask_or(~np.array(rel_pix_mask), ~clip_mask)
                vel_subset = np.ma.masked_array(vel, mask=rel_w_clip_mask).compressed()
                lat_subset = np.ma.masked_array(lat, mask=rel_w_clip_mask).compressed()
                lng_subset = np.ma.masked_array(lng, mask=rel_w_clip_mask).compressed()
                print(f'Vel, Lat, Lng subset shapes for rel pixels are {vel_subset.shape}, {lat_subset.shape}, {lng_subset.shape}')
                if vel_subset.shape[0] == 0:
                    continue
            else:
                clip_mask = ((lat > lat_start) & (lat < lat_end)) & ((lng > lng_start) & (lng < lng_end))
                vel_subset = vel[clip_mask]
                lat_subset = lat[clip_mask]
                lng_subset = lng[clip_mask]
                print(f'Vel, Lat, Lng subset shapes for unscreened pixels are {vel_subset.shape}, {lat_subset.shape}, {lng_subset.shape}')

            if lat_subset.shape[0] < 2500:
                print('Reliable pixel density insufficient, skipping tile')
                continue

            crs_wgs = proj.Proj(init='epsg:4326')
            cust = proj.Proj("+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(lat_start, lng_start))
            x, y = proj.transform(crs_wgs, cust, lng_subset, lat_subset)
            print(x)
            print(y)

            coords_tuple = (np.float64(lng_subset), np.float64(lat_subset))
            trend = vd.Trend(degree=1).fit(coords_tuple, vel_subset)
            print(f'Trend coef Verde are: {trend.coef_}')
            trend_values = trend.predict(coords_tuple)
            vel_residuals = vel_subset - trend_values 

            xy_cint_stack = np.column_stack((x,y,np.ones_like(y)))
            print(xy_cint_stack)
            print(xy_cint_stack.shape)
            R, residuals, rank, sing = np.linalg.lstsq(xy_cint_stack, np.float32(vel_subset), rcond=None)
            np_trend = xy_cint_stack @ R
            np_vel_residuals = vel_subset - np_trend

            print(f'Trend coef Numpy linalg is: {R}')
            print(f'Trend residual Numpy linalg is: {residuals}')

            if reliable_pixel_screen:
               if lat_subset.shape[0] >= 5000:
                   if 5000 <= lat_subset.shape[0] < 10000:
                       rand_samp = np.random.randint(lat_subset.shape[0], size=round(0.5*lat_subset.shape[0]))
                   elif 10000 <= lat_subset.shape[0] < 20000:
                       rand_samp = np.random.randint(lat_subset.shape[0], size=round(0.25*lat_subset.shape[0]))
                   else:
                       rand_samp = np.random.randint(lat_subset.shape[0], size=round(0.15*lat_subset.shape[0]))

                   lat_subsample = lat_subset[rand_samp]
                   lng_subsample = lng_subset[rand_samp]
                   vel_subsample = vel_subset[rand_samp]
                   vel_resid_subsample = vel_residuals[rand_samp]
                   np_vel_resid_subsample = np_vel_residuals[rand_samp]
                   x_subsample = x[rand_samp]
                   y_subsample = y[rand_samp]

               else:
                   lat_subsample = lat_subset
                   lng_subsample = lng_subset
                   vel_subsample = vel_subset
                   vel_resid_subsample = vel_residuals
                   np_vel_resid_subsample = np_vel_residuals
                   x_subsample = x
                   y_subsample = y        
           
            print('calculating variogram')
            try:
                # create two variograms for comparison as numpy and verde 1deg polynomial detrend surfaces do not produce identical results
                coords = np.column_stack((x_subsample,y_subsample))
                V = skg.Variogram(coords, np_vel_resid_subsample, n_lags=N_bins, bin_func='even', use_nugget=True) # variogram generated from numpy detrended data
                V.model = 'exponential'
                V2 = skg.Variogram(coords, vel_subsample, n_lags=N_bins, bin_func='even', use_nugget=True) # variogram generated without detrended data
                V2.model = 'exponential'

                V_params = V.describe()
                V2_params = V2.describe()
                rmse = V.rmse
                sill = V_params['sill'] + V_params['nugget'] # sill = partial sill + nugget
                sill2 = V2_params['sill'] + V2_params['nugget'] # sill = partial_sill + nugget
                eff_r = V_params['effective_range'] # effective range
                phi = 3/eff_r #spatial decay constant, slide 10 of https://www.gla.ac.uk/media/Media_418095_smxx.pdf
                
                cov_func = (V_params['sill'] + V_params['nugget']) - V.experimental # values for covariance function from detrended data
                cov_func_no_dt = (V2_params['sill'] + V2_params['nugget']) - V2.experimental # values for covariance function from non-detrended data
                cov_func0 = np.array([i if i > 0 else 0 for i in cov_func]) # bound spatial covariance function for detrended data above zero as must be non-negative, for plotting
                #p0 = (sill-test['nugget'], phi)
                #cof, cov = curve_fit(exp_func, V.bins, cov_func0, p0=p0) # curve fit struggles
                xi = np.linspace(V.bins[0], V.bins[-1], 100)
                #yi = [models.exponential(h, *cof) for h in xi]
                yi2 = (sill-V_params['nugget'])*np.exp(-phi*(xi)) # plot spatial covariance function from parameters rather than curve fitting
                
                # plot results
                fig, ax = plt.subplots(3, 3, figsize=(20,15))
                
                # first row
                ff = ax[0,0].scatter(lng[::50, ::50], lat[::50, ::50], c=vel[::50, ::50], cmap='bwr_r', s=3, vmin=-10, vmax=10) # plot full insar frame every 50 pixels, velocity min/max = -10/10 mm/yr
                ax[0,0].scatter(lng_subsample, lat_subsample, c='k', s=3) # plot subsampled points for reference
                fs = ax[0,1].scatter(x_subsample, y_subsample, c=np_vel_resid_subsample, cmap='bwr_r', s=3, vmin=-10, vmax=10) # plot detrended subsampled points, velocity min/max = -10/10 mm/yr
                tr2 = ax[0,2].scatter(x, y, c=np_trend, s=3, cmap='plasma') # plot trend surface

                # second row
                V2.plot(axes=ax[1,0], show=False) # plot experimental variogram from non-detrended data
                ax[1,1].plot(V2.bins, cov_func_no_dt, '.b') # plot covariance function values for non-detrended data
                V2.distance_difference_plot(ax=ax[1,2], show=False) 
                
                # third row
                V.plot(axes=ax[2,0], show=False) # plot experimental variogram for detrended data
                ax[2,1].plot(V.bins, cov_func0, '.b') # plot covariance function values for detrended data (bounded above zero)
                #ax[2,1].plot(xi, yi, 'og')
                ax[2,1].plot(xi, yi2, 'r') # plot covariance function defined from parameters
                V.distance_difference_plot(ax=ax[2,2], show=False)
                
                # titles
                ax[0,0].set_title('Raw Full Frame (50 pixel sub-sampling)')
                ax[0,1].set_title('Detrended Sampled Region (local coordinates)')
                ax[0,2].set_title('Spatial Trend (deg 1 polynomial fit)')
                ax[1,0].set_title('Experimental Variogram')
                ax[1,1].set_title('Covariance (unbounded)')
                ax[2,0].set_title('Experimental Variogram (detrended data)')
                ax[2,1].set_title('Covariance (detrended data & bounded to 0)')

                # colorbars 
                cb = fig.colorbar(ff, ax=ax[0,:2])
                cb3 = fig.colorbar(tr2, ax=ax[0,2])
                cb.set_label(label='Velocity (mm/yr)')
                cb3.set_label(label='Velocity trend (mm/yr)')
                # save plot
                fig.savefig(plot_save_path+'inspect_variogram_'+filepath.split('/')[-1]+'_lat_'+str(lat_start)+'_'+str(lat_end)+'_long_'+str(lng_start)+'_'+str(lng_end)+'.png')

               
               # save parameters for further use
                param_list = [{'lat_start': lat_start, 'lat_end': lat_end, 'lng_start': lng_start, 'lng_end': lng_end, 'rmse': rmse, 'effective_range': eff_r, 'partial_sill': V_params['sill'], 'sill': sill, 'nugget': V_params['nugget'], 'a': sill - V_params['nugget'], 'b': phi}]
                param_df = pd.DataFrame(param_list)
                param_df.to_csv('variogram_params_'+filepath.split('/')[-1]+'.csv', mode='a', index=False, header=False)
                
                detrended_variogram_values.append(np.column_stack((V.bins,V.experimental)))
                dvv = np.array(detrended_variogram_values)
                np.save('detrended_variogram_values_'+filepath.split('/')[-1]+'.npy', dvv)
                detrended_covariance_values.append(np.column_stack((V.bins,cov_func0)))
                dcv = np.array(detrended_covariance_values)
                np.save('detrended_bounded_covariance_values_'+filepath.split('/')[-1]+'.npy', dcv)
                

            except RuntimeError:
                print('Runtime Error for curvefit')

if __name__ == '__main__':
    construct_variogram('/home/conradb/git/insar-syn-gen/configs/insar_synthetic_vel.toml')
