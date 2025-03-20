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

def gen_tur_samples(config):

    # use toml config for easier management of parameters/args
    config = munch.munchify(toml.load(config))
    img_dim = config.data.image_dims
    output_dir = config.turbulent_delay.output_path
    rescale_size = config.turbulent_delay.rescale
    psizex = config.turbulent_delay.pixel_size_x
    psizey = config.turbulent_delay.pixel_size_y
    rows = config.turbulent_delay.rows
    cols = config.turbulent_delay.cols
    samples = config.turbulent_delay.samples
    wrapped = config.turbulent_delay.wrapped
    maxcovs = config.turbulent_delay.maxcovs
    alphas = config.turbulent_delay.alphas

    N = samples // len(maxcovs) * len(alphas) # number of samples to produce per for loop iteration

    for covar in maxcovs:
        for a in alphas:
            
            turb_atm = create_turb_del(rows, cols, covar, a, 0, N, psizex, psizey)

            for i in range(N):
                full_res = cv2.resize(turb_atm[:, : , i], (rescale_size, rescale_size)) # note MATLAB imresize is hard to replicate https://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
                halfcrop = img_dim // 2
                crop_idx = np.arange(-halfcrop, halfcrop)
                crop = full_res[np.ix_(crop_idx + math.ceil(full_res.shape[0]/2), math.ceil(full_res.shape[1]/2) + crop_idx)]
                output_path = output_dir + 'unwrapped/set'+str(2-(i % 2))+'/'
                #plt.imsave(output_path+'ifg-turb_'+str(covar)+'_'+str(a)+'_'+str(i)+'.png', crop, cmap='jet')
                np.save(output_path+'ifg-turb_maxcov_'+str(covar)+'_decay_'+str(a)+'_'+str(i), crop)

                if wrapped:
                    wrapped_crop = np.mod(crop, 2*np.pi) - np.pi # wrap between -pi to +pi
                    norm_wrapped = (wrapped_crop - np.min(wrapped_crop)) / np.ptp(wrapped_crop)
                    output_path = output_dir + 'wrapped/set'+str(2-(i % 2))+'/'
                    plt.imsave(output_path+'ifg-turb_maxcov_'+str(covar)+'_decay_'+str(a)+'_'+str(i)+'.png', norm_wrapped, cmap='jet')

if __name__ == '__main__':
    gen_tur_samples('/home/conradb/git/insar-syn-gen/configs/insar_synthetic_ifg.toml')

