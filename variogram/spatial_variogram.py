import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
import skgstat as skg
from scipy.optimize import curve_fit

file = '044D_12520_NZ_GNS_hres.h5'
subset = False

def construct_variogram(filepath):
    f = h5.File(filepath, 'r')
    datasetNames = [n for n in f.keys()]
    for n in datasetNames:
        print(n)
        print(type(f[n]))
    vel = f['Velocity']
    vel_masked = np.ma.masked_array(vel, mask=np.isnan(vel))
    rows, cols = vel.shape[0], vel.shape[1]
    print(f'array rows and cols are {rows},{cols}')

    center_row = rows // 2
    center_col = cols // 2
    start_idx = -500
    end_idx = 0
    count_idx = 500

    #if rows > cols:
		#rand_rows = np.random.randint(0, rows, round(0.8*cols,))
		#rand_cols = np.random.randint(0, cols, round(0.8*cols,))
	#else:
		#rand_rows = np.random.randint(0, rows, round(0.8*rows,))
		#rand_cols = np.random.randint(0, cols, round(0.8*rows))
	

	#rand_rows = np.random.randint(0, rows, (round(0.01*(rows*cols)),))
	#rand_cols = np.random.randint(0, cols, (round(0.01*(rows*cols)),))
		
	#coords = np.column_stack((rand_rows,rand_cols))
    if subset:
        if rows > cols:
            coords = np.random.randint(center_col+start_idx, center_col+end_idx, (round(0.6*(count_idx**2)),2))
        else:
            coords = np.random.randint(center_row+start_idx, center_row+end_idx, (round(0.6*(count_idx**2)),2))
        print(coords.shape)
    else:
        if rows > cols:
            x = np.linspace(center_col+start_idx, center_col+end_idx, count_idx, dtype=int)
            y = np.linspace(center_col+start_idx, center_col+end_idx, count_idx, dtype=int)
            yy, xx = np.meshgrid(y, x)
            coords = np.column_stack((yy.flatten(), xx.flatten()))
        else:
            x = np.linspace(center_row+start_idx, center_row+end_idx, count_idx, dtype=int)
            y = np.linspace(center_row+start_idx, center_row+end_idx, count_idx, dtype=int)
            yy, xx = np.meshgrid(y, x)
            coords = np.column_stack((yy.flatten(), xx.flatten()))
        print(coords.shape)
	
	# only use values/coords where non-NaN
    mask = ~np.isnan(np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float))
    values = np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float)
    masked_vals = values[mask]
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
    V = skg.Variogram(masked_coords, masked_vals, n_lags=30, bin_func='uniform')
    #V.fit_method ='lm'
    #fig = V.plot(show=False)
    #fig.savefig('test_variogram_'+file+'.png')
    #fig2 = V.distance_difference_plot()
    #fig2.savefig('test_dist_diff_'+file+'.png')

    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    print('plotting full frame')
    ax[0,0].scatter(long[::50, ::50], lat[::50, ::50], c=vel[::50, ::50], cmap='bwr_r', s=3)
    #ax[0,0] = plt.scatter(long_masked[::100, ::100], lat_masked[::100, ::100], c=vel_masked[::100, ::100], cmap='bwr_r')
    print('plotting frame subset')
    ax[0,1].scatter(longs_grouped, lats_grouped, c=masked_vals, cmap='bwr_r', s=3)
    #ax[1,0].plot(V.bins, V.experimental, '.b')
    V.plot(axes=ax[1,0], show=False)
    V.distance_difference_plot(ax=ax[1,1], show=False)
    fig.savefig('inspect_variogram_'+file+'.png')

    print(V.parameters)

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

construct_variogram('/home/conradb/Downloads/'+file)
