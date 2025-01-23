import h5py as h5
import numpy as np
import skgstat as skg
from scipy.optimize import curve_fit

file = '096A_13547_NZ_GNS_rural.h5'

def construct_variogram(filepath):
	f = h5.File(filepath, 'r')
	datasetNames = [n for n in f.keys()]
	for n in datasetNames:
		print(n)
		print(type(f[n]))
	vel = f['Velocity']
	rows, cols = vel.shape[0], vel.shape[1]

	#if rows > cols:
		#rand_rows = np.random.randint(0, rows, round(0.8*cols,))
		#rand_cols = np.random.randint(0, cols, round(0.8*cols,))
	#else:
		#rand_rows = np.random.randint(0, rows, round(0.8*rows,))
		#rand_cols = np.random.randint(0, cols, round(0.8*rows))
	

	#rand_rows = np.random.randint(0, rows, (round(0.01*(rows*cols)),))
	#rand_cols = np.random.randint(0, cols, (round(0.01*(rows*cols)),))
		
	#coords = np.column_stack((rand_rows,rand_cols))

	if rows > cols:
		coords = np.random.randint(0, 2000, (round(0.02*(2000**2)),2))
	else:
		coords = np.random.randint(0, 2000, (round(0.02*(2000**2)),2))
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

	# assert value in original array equal masked array
	#assert vel[masked_rows[0], masked_cols[0]] == masked_vals[0]

	V = skg.Variogram(masked_coords, masked_vals, n_lags=25, bin_func='even')
	V.fit_method ='lm'
	fig = V.plot(show=False)
	fig.savefig('test_variogram_'+file+'.png')
	fig2 = V.distance_difference_plot()
	fig2.savefig('test_dist_diff_'+file+'.png')

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
