import h5py as h5
import munch
import numpy as np
import toml
from scipy.ndimage import generic_filter

def calc_spike_noise(config):
    
    # use toml config for easier management of parameters/args
    config = munch.munchify(toml.load(config))
    frame = config.data.frame # insar frame defines region and resolution
    dims = config.data.image_dims # image size/shape in pixels
    reliable_pixel_screen = config.filters.rel_pix # use reliable pixels?
    nan_thresh = config.filters.nan_thresh # max percentage of the data that can be nans

    f = h5.File(frame, 'r')
    
    # use only reliable pixels
    if reliable_pixel_screen:
        rel_pix_mask = f['Reliable_Pixels'] # grab reliable pixel mask, True = reliable
        vel = f['Velocity']
        vel_ma = np.ma.masked_array(vel, mask=~np.array(rel_pix_mask)) # use masked array to preserve 2d shape, True = unreliable = masked values
        vel_ma = vel_ma.filled(fill_value=np.nan) # use nan to replace masked vals for easier filtering later
        rows, cols = vel_ma.shape
        print(f'rows are size {rows}, cols are size {cols}')
        row_strides = rows // dims
        col_strides = cols // dims
        
        # nested for loop to iterate over 2d array in pixel dims x dims slices (e.g. 224 x 224)
        for r_idx in range(row_strides):
            for c_idx in range(col_strides):
                col_start = c_idx * dims
                col_end = (c_idx * dims) + dims
                row_start = r_idx * dims
                row_end = (r_idx * dims) + dims

                vel_slice = vel_ma[row_start:row_end, col_start:col_end]
                nan_count = np.isnan(vel_slice).sum()
                
                # check which slices have less than 65% as nan values
                if nan_count < (0.65*dims**2):

                    filt_vel = generic_filter(vel_slice.astype(np.float32), np.nanmedian, footprint=np.ones((3,3))) # 3x3 pixel median filter
                    N = vel_slice - filt_vel # noise map in mm/yr
                    N = np.nan_to_num(N, nan=0) # reloading a .npy file with nans will return garbage values so we replace with zeros
                    np.save('../test_outputs/spike_noise/noise_map_'+str(frame.split('/')[-1])+'_rows_'+str(row_start)+'-'+str(row_end)+'_cols_'+str(col_start)+'-'+str(col_end), N)

                    decoh_mask = np.isnan(vel_slice)
                    np.save('../test_outputs/decoh_mask/decoh_mask_'+str(frame.split('/')[-1])+'_rows_'+str(row_start)+'-'+str(row_end)+'_cols_'+str(col_start)+'-'+str(col_end), decoh_mask)






if __name__ == '__main__':
    calc_spike_noise('../configs/insar_synthetic_vel.toml')




