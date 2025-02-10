import csv
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pyproj as proj
from scipy.optimize import curve_fit
import skgstat as skg
from skgstat import models
import verde as vd

import llh2local

file = '096A_13547_NZ_GNS_rural.h5'
plot_save_path = '/home/conradb/git/insar-syn-gen/test_outputs/variogram_plots/'
subsample = False
samp_frac = 0.15
relpix = True
N_bins = 30
#row_start = 6000
#row_end = 6200
#col_start = 3000
#col_end = 3200

#row_pixels = 250
#col_pixels = 250

# assume 0.01 deg is equal to ~1km
lat_res = 0.045 #4.5km
lng_res = 0.045 #4.5km

def construct_variogram(filepath, lat_res, lng_res, reliable_pixel_screen=True, subsample=False, samp_frac=None):

    f = h5.File(filepath, 'r')
    datasetNames = [n for n in f.keys()]

    for n in datasetNames:
        print(n)
        print(type(f[n]))

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

    with open('variogram_params_'+file+'_'+'.csv', 'w') as csvfile:
        fieldnames = ['lat_start', 'lat_end', 'lng_start', 'lng_end', 'rmse', 'range', 'effective_range', 'partial_sill', 'sill', 'nugget', 'a', 'b']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    detrended_variogram_values = []
    detrended_covariance_values = []

        


    # TO DO - apply reliable pixel mask to velocities
    # testing filtered displacements, subtract two timesteps to mimic an interferogram
    #disp_raw = f['Cumulative_Displacement_TSmooth']
    #print(disp_raw.shape)
    #vel = np.diff(disp_raw[100:102, :, :], axis=0)
    #vel = vel[0, :, :]
    #print(vel.shape)

    #center_row = rows // 2
    #center_col = cols // 2
    #start_idx = 300
    #end_idx = 900
    #count_idx = 600

    # create a sliding window to move over frame
    #row_strides = rows // row_pixels
    #col_strides = cols // col_pixels
   
    #for r_idx in range(row_strides):
        #for c_idx in range(col_strides):

            #col_start = c_idx * col_pixels
            #col_end = (c_idx * col_pixels) + col_pixels
            #row_start = r_idx * row_pixels
            #row_end = (r_idx * row_pixels) + row_pixels
            #print(f'sampling rows {row_start} - {row_end}')
            #print(f'sampling columns {col_start} - {col_end}')

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
                #vel_subset = vel_subset[clip_mask.flatten()]
                #vel_subset = vel_subset.compressed()
                print(f'Vel, Lat, Lng subset shapes for rel pixels are {vel_subset.shape}, {lat_subset.shape}, {lng_subset.shape}')
                if vel_subset.shape[0] == 0:
                    continue
            else:
                clip_mask = ((lat > lat_start) & (lat < lat_end)) & ((lng > lng_start) & (lng < lng_end))
                vel_subset = vel[clip_mask]
                lat_subset = lat[clip_mask]
                lng_subset = lng[clip_mask]
                print(f'Vel, Lat, Lng subset shapes for unscreened pixels are {vel_subset.shape}, {lat_subset.shape}, {lng_subset.shape}')


    #if rows > cols:
		#rand_rows = np.random.randint(0, rows, round(0.8*cols,))
		#rand_cols = np.random.randint(0, cols, round(0.8*cols,))
	#else:
		#rand_rows = np.random.randint(0, rows, round(0.8*rows,))
		#rand_cols = np.random.randint(0, cols, round(0.8*rows))
	

	#rand_rows = np.random.randint(0, rows, (round(0.01*(rows*cols)),))
	#rand_cols = np.random.randint(0, cols, (round(0.01*(rows*cols)),))
		
	#coords = np.column_stack((rand_rows,rand_cols))

            #if subsample:
               #subsample_count = (round(samp_frac*((col_pixels * row_pixels))),)
               #print(f'subsampling with {subsample_count} data points')
               #rand_cols = np.random.randint(col_start, col_end, subsample_count)
               #rand_rows = np.random.randint(row_start, row_end, subsample_count)
               #coords = np.column_stack((rand_rows, rand_cols))

        #if rows > cols:
            #coords = np.random.randint(center_col+start_idx, center_col+end_idx, (round(0.6*(count_idx**2)),2))
        #else:
            #coords = np.random.randint(center_row+start_idx, center_row+end_idx, (round(0.6*(count_idx**2)),2))

               #print(f'Coords shape pre masking {coords.shape}')

            #else:
                #x = np.linspace(col_start, col_end, col_pixels, dtype=int)
                #y = np.linspace(row_start, row_end, row_pixels, dtype=int)
                #yy, xx = np.meshgrid(y, x)
                #coords = np.column_stack((yy.flatten(), xx.flatten()))

        #if rows > cols:
            #x = np.linspace(center_col+start_idx, center_col+end_idx, count_idx, dtype=int)
            #y = np.linspace(center_col+start_idx, center_col+end_idx, count_idx, dtype=int)
            #yy, xx = np.meshgrid(y, x)
            #coords = np.column_stack((yy.flatten(), xx.flatten()))
        #else:
            #x = np.linspace(center_row+start_idx, center_row+end_idx, count_idx, dtype=int)
            #y = np.linspace(center_row+start_idx, center_row+end_idx, count_idx, dtype=int)
            #yy, xx = np.meshgrid(y, x)
            #coords = np.column_stack((yy.flatten(), xx.flatten()))
                #print(f'Coords shape pre masking {coords.shape}')

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
               #rel_pix = f['Reliable_Pixels']
               #print(rel_pix.shape)
               #pix_mask = np.fromiter((rel_pix[c[0], c[1]] for c in coords), dtype=bool)
               #print(pix_mask.shape)
               #coords = coords[pix_mask]
               #print(f'Coords shape after reliable pixels mask {coords.shape}')
               #print(coords.dtype)

               
               #full_vel = np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float)
               #print(f'Full res velocity is shape {full_vel.shape}')
               #print(type(full_vel))

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
           
            #sll = np.vstack((lng_subsample, lat_subsample))
            #ref_point = np.array([lng_start, lat_start])
            #xy = llh2local.llh2local(sll, ref_point)



	
	    # only use values/coords where non-NaN
            
            #mask = ~np.isnan(np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float))
            #print(mask)
            #values = np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float)
            #masked_vals = values[mask]
            #if masked_vals.shape[0] < 800:
                #coords = coords[rand_samp, :]
                #coords = coords[rand_samp, :]
                #print('Pixel density insufficient, skipping tile')
                #continue
	#masked_rows = rand_rows[mask]
	#masked_cols = rand_cols[mask]
	#masked_coords = np.column_stack((masked_rows,masked_cols))
            #masked_coords = coords[mask]
            #print(values)
            #print(values.shape)
            #print(masked_vals)
            #print(masked_vals.shape)
            #print(masked_coords.shape)

            #lat = f['Latitude']
            #print('masking lat')
    #lat_masked = np.ma.masked_array(lat, mask=np.isnan(lat))
            #long = f['Longitude']
            #print('masking long')
    #long_masked = np.ma.masked_array(long, mask=np.isnan(long))
            #print('grouping lats')
            #lats_grouped = np.fromiter((lat[c[0], c[1]] for c in masked_coords), dtype=float)
            #print('grouping longs')
            #longs_grouped = np.fromiter((long[c[0], c[1]] for c in masked_coords), dtype=float)

    # assert value in original array equal masked array
    #assert vel[masked_rows[0], masked_cols[0]] == masked_vals[0]
            print('calculating variogram')
            try:
                coords = np.column_stack((x_subsample,y_subsample))
                V = skg.Variogram(coords, np_vel_resid_subsample, n_lags=N_bins, bin_func='even', use_nugget=True)
                V.model = 'exponential'
                V2 = skg.Variogram(coords, vel_subsample, n_lags=N_bins, bin_func='even', use_nugget=True)
                V2.model = 'exponential'

                params = V.parameters
                print(params)
                test = V.describe()
                test2 = V2.describe()
                print(test)
                rmse = V.rmse
                print(rmse)
                sill = test['sill']+test['nugget']
                sill2 = test2['sill']+test2['nugget']
                eff_r = test['effective_range']
                phi = 3/eff_r
                
                def exp_func(d, a, b):
                    return a*np.exp(-b*d)

                cov_func = (test['sill']+test['nugget']) - V.experimental
                cov_func0 = np.array([i if i > 0 else 0 for i in cov_func])
                print(cov_func)
                cov_func2 = (test2['sill']+test2['nugget']) - V2.experimental
                #p0 = [np.mean(V.bins), np.mean(cov_func), 0]
                p0 = (sill-test['nugget'], phi)
                cof, cov = curve_fit(exp_func, V.bins, cov_func0, p0=p0)
                xi = np.linspace(V.bins[0], V.bins[-1], 100)
                yi = [models.exponential(h, *cof) for h in xi]
                yi2 = (sill-test['nugget'])*np.exp(-phi*(xi))
                #V.fit_method ='lm
                #fig = V.plot(show=False)
                #fig.savefig('test_variogram_'+file+'.png'
                #fig2 = V.distance_difference_plot(
                #fig2.savefig('test_dist_diff_'+file+'.png'
                #min_vel = np.min(masked_vals)
                #print(min_vel)
                #max_vel = np.max(masked_vals)
                #print(max_vel)
                fig, ax = plt.subplots(3, 3, figsize=(20,15))

                print('plotting full frame')
                ff = ax[0,0].scatter(lng[::50, ::50], lat[::50, ::50], c=vel[::50, ::50], cmap='bwr_r', s=3, vmin=-10, vmax=10)
                ax[0,0].scatter(lng_subsample, lat_subsample, c='k', s=3)
                #ax[0,0] = plt.scatter(long_masked[::100, ::100], lat_masked[::100, ::100], c=vel_masked[::100, ::100], cmap='bwr_r')
                print('plotting detrended subset')
                fs = ax[0,1].scatter(x_subsample, y_subsample, c=np_vel_resid_subsample, cmap='bwr_r', s=3, vmin=-10, vmax=10)
                #ax[1,0].plot(V.bins, V.experimental, '.b')
                #tr = ax[0,2].scatter(coords_tuple[0], coords_tuple[1], c=trend_values, s=3, cmap='plasma')

                tr2 = ax[0,2].scatter(x, y, c=np_trend, s=3, cmap='plasma')

                V2.plot(axes=ax[1,0], show=False)
                ax[1,1].plot(V2.bins, cov_func2, '.b')
                V2.distance_difference_plot(ax=ax[1,2], show=False)

                V.plot(axes=ax[2,0], show=False)
                ax[2,1].plot(V.bins, cov_func0, '.b')
                #ax[2,1].plot(xi, yi, 'og')
                ax[2,1].plot(xi, yi2, 'r')
                V.distance_difference_plot(ax=ax[2,2], show=False)

                ax[0,0].set_title('Raw Full Frame (50 pixel sub-sampling)')
                ax[0,1].set_title('Detrended Sampled Region (local coordinates)')
                ax[0,2].set_title('Spatial Trend (deg 1 polynomial fit)')
                ax[1,0].set_title('Experimental Variogram')
                ax[1,1].set_title('Covariance (unbounded)')
                ax[2,0].set_title('Experimental Variogram (detrended)')
                ax[2,1].set_title('Covariance (detrended & bounded to 0)')

                
                cb = fig.colorbar(ff, ax=ax[0,:2])
                #cb2 = fig.colorbar(tr, ax=ax[0,2])
                cb3 = fig.colorbar(tr2, ax=ax[0,2])
                cb.set_label(label='Velocity (mm/yr)')
                #cb2.set_label(label='Verde - Velocity trend (mm/yr)')
                cb3.set_label(label='Velocity trend (mm/yr)')
                fig.savefig(plot_save_path+'inspect_variogram_'+file+'_lat_'+str(lat_start)+'_'+str(lat_end)+'_long_'+str(lng_start)+'_'+str(lng_end)+'.png')

               
               # Initialize data to lists.
                param_list = [{'lat_start': lat_start, 'lat_end': lat_end, 'lng_start': lng_start, 'lng_end': lng_end, 'rmse': rmse,'range': params[0], 'effective_range': eff_r, 'partial_sill': test['sill'], 'sill': sill, 'nugget': test['nugget'], 'a': sill - test['nugget'], 'b': phi}]
                param_df = pd.DataFrame(param_list)
                param_df.to_csv('variogram_params_'+file++'.csv', mode='a', index=False, header=False)
                
                detrended_variogram_values.append(np.column_stack((V.bins,V.experimental)))
                dvv = np.array(detrended_variogram_values)
                np.save('detrended_variogram_values_'+file+'.npy', dvv)
                detrended_covariance_values.append(np.column_stack((V.bins,cov_func0)))
                dcv = np.array(detrended_covariance_values)
                np.save('detrended_covariance_values_'+file+'.npy', dcv)
                

            except RuntimeError:
                print('Runtime Error for curvefit')
    # set estimator back
	#V.estimator = 'matheron'
	#V.model = 'spherical'
	#xdata = V.bins
	#ydata = V.experimental

	# initial guess, required for curve fit using the Levenberg-Marquardt algorithm
	#p0 = [np.mean(xdata), np.mean(ydata), 0]
	#cof, cov =curve_fit(skg.models.spherical, xdata, ydata, p0=p0)
	#print("range: %.2f   sill: %.f   nugget: %.2f" % (cof[0], cof[1], cof[2]))

	#plotting
	#xi =np.linspace(xdata[0], xdata[-1], 100)
	#yi = [models.spherical(h, *cof) for h in xi]
	#fig, ax = plt.subplots()
	#ax.plt(xdata, ydata, 'og')
	#ax.plt(xi, yi, '-b')
	#fig.savefig('curve_fit_'+file+'.png')

construct_variogram('/home/conradb/Downloads/'+file, lat_res, lng_res, reliable_pixel_screen=relpix, subsample=subsample, samp_frac=samp_frac)
