import h5py as h5
import numpy as np
import skgstat as skg

def construct_variogram(filepath):
	f = h5.File(filepath, 'r')
	datasetNames = [n for n in f.keys()]
	for n in datasetNames:
		print(n)
		print(type(f[n]))
	vel = f['Velocity']
	rows, cols = vel.shape[0], vel.shape[1]
		
	rand_rows = np.random.randint(0, rows, (2000,))
	rand_cols = np.random.randint(0, cols, (2000,))
		
	coords = np.column_stack((rand_rows,rand_cols))
	
	# only use values/coords where non-NaN
	mask = ~np.isnan(np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float))
	values = np.fromiter((vel[c[0], c[1]] for c in coords), dtype=float)
	masked_vals = values[mask]
	masked_rows = rand_rows[mask]
	masked_cols = rand_cols[mask]
	masked_coords = np.column_stack((masked_rows,masked_cols))
	print(values)
	print(values.shape)
	print(masked_vals)
	print(masked_vals.shape)
	print(masked_coords.shape)

	# assert value in original array equal masked array
	assert vel[masked_rows[0], masked_cols[0]] == masked_vals[0]

	V = skg.Variogram(masked_coords, masked_vals)
	fig = V.plot(show=False)
	fig.savefig('test_variogram.png')
	fig2 = V.distance_difference_plot()
	fig2.savefig('test_dist_diff.png')


construct_variogram('/home/conradb/Downloads/073D_Auckland_000006_rural_gns.h5')
