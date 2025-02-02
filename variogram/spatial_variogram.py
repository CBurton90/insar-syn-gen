import h5py as h5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import skgstat as skg
import verde as vd
from scipy.optimize import curve_fit
import csv

file = '146D_12547_NZ_GNS_hres.h5'
plot_save_path = '/home/conradb/git/insar-syn-gen/test_outputs/variogram_plots/'
subsample = False
samp_frac = 0.15
relpix = True
#row_start = 6000
#row_end = 6200
#col_start = 3000
#col_end = 3200

row_pixels = 250
col_pixels = 250

def construct_variogram(filepath, row_pixels, col_pixels, reliable_pixel_screen=True, subsample=False, samp_frac=None):
    f = h5.File(filepath, 'r')
    datasetNames = [n for n in f.keys()]
    for n in datasetNames:
        print(n)
        print(type(f[n]))
    vel = f['Velocity']
    vel_masked = np.ma.masked_array(vel, mask=np.isnan(vel))
    rows, cols = vel.shape[0], vel.shape[1]
    print(f'array rows and cols are {rows},{cols}')

    # TO DO - apply reliable pixel mask to velocities
    # testing filtered displacements, subtract two timesteps to mimic an interferogram
    #disp_raw = f['Cumulative_Displacement_TSmooth']
    #print(disp_raw.shape)
    #vel = np.diff(disp_raw[100:102, :, :], axis=0)
    #vel = vel[0, :, :]
    #print(vel.shape)

    center_row = rows // 2
    center_col = cols // 2
    start_idx = 300
    end_idx = 900
    count_idx = 600

    # create a sliding window to move over frame
    row_strides = rows // row_pixels
    col_strides = cols // col_pixels

    with open('variogram_params_'+file+'_'+'.csv', 'w') as csvfile:
        fieldnames = ['frame_coords','rmse','range', 'sill', 'nugget']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for r_idx in range(row_strides):
        for c_idx in range(col_strides):

            col_start = c_idx * col_pixels
            col_end = (c_idx * col_pixels) + col_pixels
            row_start = r_idx * row_pixels
            row_end = (r_idx * row_pixels) + row_pixels
            print(f'sampling rows {row_start} - {row_end}')
            print(f'sampling columns {col_start} - {col_end}')
    #if rows > cols:
		#rand_rows = np.random.randint(0, rows, round(0.8*cols,))
		#rand_cols = np.random.randint(0, cols, round(0.8*cols,))
	#else:
		#rand_rows = np.random.randint(0, rows, round(0.8*rows,))
		#rand_cols = np.random.randint(0, cols, round(0.8*rows))
	

	#rand_rows = np.random.randint(0, rows, (round(0.01*(rows*cols)),))
	#rand_cols = np.random.randint(0, cols, (round(0.01*(rows*cols)),))
		
	#coords = np.column_stack((rand_rows,rand_cols))
            if subsample:
               subsample_count = (round(samp_frac*((col_pixels * row_pixels))),)
               print(f'subsampling with {subsample_count} data points')
               rand_cols = np.random.randint(col_start, col_end, subsample_count)
               rand_rows = np.random.randint(row_start, row_end, subsample_count)
               coords = np.column_stack((rand_rows, rand_cols))

        #if rows > cols:
            #coords = np.random.randint(center_col+start_idx, center_col+end_idx, (round(0.6*(count_idx**2)),2))
        #else:
            #coords = np.random.randint(center_row+start_idx, center_row+end_idx, (round(0.6*(count_idx**2)),2))

               print(f'Coords shape pre masking {coords.shape}')

            else:
                x = np.linspace(col_start, col_end, col_pixels, dtype=int)
                y = np.linspace(row_start, row_end, row_pixels, dtype=int)
                yy, xx = np.meshgrid(y, x)
                coords = np.column_stack((yy.flatten(), xx.flatten()))

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
                print(f'Coords shape pre masking {coords.shape}')


            if reliable_pixel_screen:
               rel_pix = f['Reliable_Pixels']
               print(rel_pix.shape)
               pix_mask = np.fromiter((rel_pix[c[0], c[1]] for c in coords), dtype=bool)
               print(pix_mask.shape)
               coords = coords[pix_mask]
               print(f'Coords shape after reliable pixels mask {coords.shape}')
               print(coords.dtype)

               if coords.shape[0] < 800:
                  print('Reliable pixel density insufficient, skipping tile')
                  continue

               full_vel = np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float)
               print(f'Full res velocity is shape {full_vel.shape}')
               print(type(full_vel))
               coords_tuple = (np.float64(coords[:, 1]), np.float64(coords[:, 0]))
               trend = vd.Trend(degree=1).fit(coords_tuple, full_vel)
               trend_values = trend.predict(coords_tuple)

               if 5000 <= coords.shape[0] < 10000:
                  subsample_count = (round(samp_frac*((col_pixels * row_pixels))),)
                  print(f'subsampling with {subsample_count} data points')
                  rand_samp = np.random.randint(coords.shape[0], size=round(0.3*coords.shape[0]))
                  coords = coords[rand_samp, :]
               elif 10000 <= coords.shape[0] < 20000:
                  subsample_count = (round(samp_frac*((col_pixels * row_pixels))),)
                  print(f'subsampling with {subsample_count} data points')
                  rand_samp = np.random.randint(coords.shape[0], size=round(0.15*coords.shape[0]))
                  coords = coords[rand_samp, :]
               elif coords.shape[0] >= 20000:
                  subsample_count = (round(samp_frac*((col_pixels * row_pixels))),)
                  print(f'subsampling with {subsample_count} data points')
                  rand_samp = np.random.randint(coords.shape[0], size=round(0.1*coords.shape[0]))
                  coords = coords[rand_samp, :]
	
	    # only use values/coords where non-NaN
            mask = ~np.isnan(np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float))
            print(mask)
            values = np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float)
            masked_vals = values[mask]
            if masked_vals.shape[0] < 800:
               print('Pixel density insufficient, skipping tile')
               continue
	#masked_rows = rand_rows[mask]
	#masked_cols = rand_cols[mask]
	#masked_coords = np.column_stack((masked_rows,masked_cols))
            masked_coords = coords[mask]
            print(values)
            print(values.shape)
            print(masked_vals)
            print(masked_vals.shape)
            print(masked_coords.shape)

            lat = f['Latitude']
            print('masking lat')
    #lat_masked = np.ma.masked_array(lat, mask=np.isnan(lat))
            long = f['Longitude']
            print('masking long')
    #long_masked = np.ma.masked_array(long, mask=np.isnan(long))
            print('grouping lats')
            lats_grouped = np.fromiter((lat[c[0], c[1]] for c in masked_coords), dtype=float)
            print('grouping longs')
            longs_grouped = np.fromiter((long[c[0], c[1]] for c in masked_coords), dtype=float)

    # assert value in original array equal masked array
    #assert vel[masked_rows[0], masked_cols[0]] == masked_vals[0]
            print('calculating variogram')
            try:
               V = skg.Variogram(masked_coords, masked_vals, n_lags=30, bin_func='even')
               V.model = 'exponential'
    #V.fit_method ='lm'
    #fig = V.plot(show=False)
    #fig.savefig('test_variogram_'+file+'.png')
    #fig2 = V.distance_difference_plot()
    #fig2.savefig('test_dist_diff_'+file+'.png')

               min_vel = np.min(masked_vals)
               print(min_vel)
               max_vel = np.max(masked_vals)
               print(max_vel)

               fig, ax = plt.subplots(2, 3, figsize=(15,10))
               print('plotting full frame')
               ff = ax[0,0].scatter(long[::50, ::50], lat[::50, ::50], c=vel[::50, ::50], cmap='bwr_r', s=3, vmin=-10, vmax=10)
               ax[0,0].scatter(longs_grouped, lats_grouped, c='k', s=3)
               #ax[0,0] = plt.scatter(long_masked[::100, ::100], lat_masked[::100, ::100], c=vel_masked[::100, ::100], cmap='bwr_r')
               print('plotting frame subset')
               fs = ax[0,1].scatter(longs_grouped, lats_grouped, c=masked_vals, cmap='bwr_r', s=3, vmin=-10, vmax=10)
               #ax[1,0].plot(V.bins, V.experimental, '.b')
               tr = ax[0,2].scatter(coords_tuple[0], coords_tuple[1], c=trend_values, s=3, cmap='plasma')
               V.plot(axes=ax[1,0], show=False)

               V.distance_difference_plot(ax=ax[1,1], show=False)
               cb = fig.colorbar(ff, ax=ax[0, :2])
               cb2 = fig.colorbar(tr, ax=ax[0, 2])
               cb.set_label(label='Velocity (mm/yr)')
               cb2.set_label(label='Velocity (mm/yr)')
               fig.savefig(plot_save_path+'inspect_variogram_'+file+'_rows_'+str(row_start)+'-'+str(row_end)+'_cols_'+str(col_start)+'-'+str(col_end)+'.png')

               params = V.parameters
               print(params)
               rmse = V.rmse
               print(rmse)

               # Initialize data to lists.
               param_list = [{'frame_coords': 'rows_'+str(row_start)+'_'+str(row_end)+'_cols_'+str(col_start)+'_'+str(col_end),'rmse': rmse,'range': params[0], 'sill': params[1], 'nugget': params[2]}]
               param_df = pd.DataFrame(param_list)
               param_df.to_csv('variogram_params_'+file+'_'+'.csv', mode='a', index=False, header=False)

            except RuntimeError:                                                                                                                                                                                                                                           print('Runtime Error for curvefit')
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

construct_variogram('/home/conradb/Downloads/'+file, row_pixels, col_pixels, reliable_pixel_screen=relpix, subsample=subsample, samp_frac=samp_frac)
