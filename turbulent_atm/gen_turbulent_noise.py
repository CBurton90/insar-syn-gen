# refactored into Python by C Burton 2025 from #https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/main_code/runGenTurbulent.m
# TJW 2004 & Nantheera Anantrasirichai 2021

import cv2
import math
import munch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import toml
from cholesky_decomp import create_turb_del

# dummy params from original code - ignore
#halfcrop = 224 // 2 # Alexnet input size 227x227
#img_size = 500 # resolution in pixels

#maxvariance = [i + 7.5 for i in [-2, -1, 0, 0.75, 1.5]]
#alpha = [i * 0.008 for i in  [0.5, 0.75, 1, 1.5, 2]]

def gen_tur_samples(config):

    # use toml config for easier management of parameters/args
    config = munch.munchify(toml.load(config))
    img_dim = config.data.image_dims
    output_dir = config.turbulent_delay.output_path
    param_file = config.turbulent_delay.cov_params
    psizex = config.turbulent_delay.pixel_size_x
    psizey = config.turbulent_delay.pixel_size_y
    rows = config.turbulent_delay.rows
    cols = config.turbulent_delay.cols
    samples = config.turbulent_delay.samples
    wrapped = config.turbulent_delay.wrapped

    df = pd.read_csv(param_file)
    df_filt = df.loc[df['rmse'] < 1.5]
    max_covar = df_filt['a'].max() # max max covariance for covariance function
    min_covar = df_filt['a'].min() # min max covariance for covariance function
    max_decay = df_filt['b'].max() # max spatial decay constant for covariance function
    min_decay = df_filt['b'].min() # min spatial decay constant for covariance function

    # draw samples of maximum covariance and spatial decay constant from uniform distribution to produce individual covariance functions that then can create a covariance matrix to be decomposed
    covar_arr = np.random.uniform(min_covar, max_covar, 10) # alpha in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9181454, max covariance in https://www.sciencedirect.com/science/article/pii/S003442571930183X
    decay_arr = np.random.uniform(min_decay, max_decay, 10) # beta in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9181454, k in https://www.sciencedirect.com/science/article/pii/S003442571930183X
    print(covar_arr)
    print(decay_arr)
    N = samples // (covar_arr.size * decay_arr.size) # number of samples to produce per for loop iteration
    print(N)

    for covar in covar_arr:
        for k in decay_arr:
            turb_atm = create_turb_del(rows, cols, covar, k, 0, N, psizex, psizey)
            for i in range(N):
                # choosing not to use resize and crop as it could alter data, instead choose rows and cols to match image size
                #full_res = cv2.resize(turb_atm[:, : , i], (img_dim, img_dim)) # note MATLAB imresize is hard to replicate https://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
                #crop_idx = np.arange(-halfcrop, halfcrop)
                #crop = full_res[np.ix_(crop_idx + math.ceil(full_res.shape[0]/2), math.ceil(full_res.shape[1]/2) + crop_idx)]

                crop = turb_atm[:, :, i] / 1000 # convert from mm/yr to m/yr
                print(crop.shape)
                crop = crop/0.028333 * 2 * np.pi # convert to radians for Sentinel-1 C-band half wavelength

                output_path = output_dir + 'unwrapped/set'+str(2-(i % 2))+'/'
                #plt.imsave(output_path+'turb_'+str(var)+'_'+str(a)+'_'+str(i)+'.png', crop, cmap='jet')
                np.save(output_path+'turb_maxcov_'+str(covar)+'_decay_'+str(k)+'_'+str(i), crop)

                if wrapped:
                    #wrapped_crop = np.angle(np.exp(1j * crop))
                    #norm_wrapped = wrapped_crop - np.min(wrapped_crop) / (np.max(wrapped_crop) - np.min(wrapped_crop))
                    wrapped_crop = np.mod(crop, 2*np.pi) - np.pi # wrap between -pi to +pi
                    norm_wrapped = (wrapped_crop - np.min(wrapped_crop)) / np.ptp(wrapped_crop)
                    output_path = output_dir + 'wrapped/set'+str(2-(i % 2))+'/'
                    plt.imsave(output_path+'+pi-pi_turb_maxcov_'+str(covar)+'_decay_'+str(k)+'_'+str(i)+'.png', wrapped_crop, cmap='jet', vmin=-np.pi, vmax=np.pi)
                    plt.imsave(output_path+'turb_maxcov_'+str(covar)+'_decay_'+str(k)+'_'+str(i)+'.png', norm_wrapped, cmap='jet', vmin=0, vmax=1)

if __name__ == '__main__':
    gen_tur_samples('/home/conradb/git/insar-syn-gen/configs/insar_synthetic_vel.toml')
