# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/main_code/runGenStratified.m
# TJW 2004 & Nantheera Anantrasirichai 2021

import cv2
import glob
import math
import munch
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
import toml
import os

def gen_strat_samples(config):

    # use toml config for easier management of parameters/args
    config = munch.munchify(toml.load(config))
    img_dim = config.data.image_dims
    output_dir = config.stratified.output_path
    rescale_size = config.stratified.rescale
    samples = config.stratified.samples
    incidence = config.stratified.incidence
    wrapped = config.stratified.wrapped

    wavelength = 0.055465
    m2rad = 4.0 * np.pi / wavelength
    rad2m = (wavelength / (4.0 * np.pi))
    zen2los = 1.0 / np.cos(incidence / (180*np.pi))
    halfcrop = img_dim // 2

    count = 0

    dirs = [x[0] for x in os.walk('./gacos')]
    #print(dirs[1:])

    for d in dirs[1:]:
        path = d + '/*.ztd'
        t = glob.glob(path)
        for i in range(len(t)):

            dt = np.dtype('<f')
            print(f'open {t[i]}')
            atmo1 = np.fromfile(t[i], dtype=dt)
            l = atmo1.shape[0]
            atmo1 = atmo1.reshape(int(np.sqrt(l)), int(np.sqrt(l)))
            print(atmo1.shape)
            print(f'open {t[i+1]}')
            atmo2 = np.fromfile(t[i+1], dtype=dt)
            atmo2 = atmo2.reshape(int(np.sqrt(l)), int(np.sqrt(l)))

            print(f'i is {i}')

            for angle in [0, 90, 180, 270]:
                atmo1 = ndimage.rotate(atmo1, angle, reshape=False)
                atmo2 = ndimage.rotate(atmo2, angle, reshape=False)

                atmo = (atmo2-atmo1)*zen2los*m2rad
                atmo = cv2.resize(atmo, (rescale_size, rescale_size))

                print(atmo.shape)
            
                disk = morphology.disk(3)
                mask = morphology.binary_erosion(atmo != 0, disk)
                print(mask)

                crop_idx = np.arange(-halfcrop, halfcrop)
                atmo = atmo[np.ix_(crop_idx + math.ceil(atmo.shape[0]/2), math.ceil(atmo.shape[1]/2) + crop_idx)]
                output_path = output_dir + 'unwrapped/set'+str(2 - (i % 2))+'/'
                
                if count < samples:
                    #plt.imsave(str(t[i]).split('/')[-2]+'_diff_angle_'+str(angle)+'_'+str(t[i+1]).split('/')[-1]+'-'+str(t[i]).split('/')[-1]+'.png', atmo, cmap='jet')
                    np.save(output_path+str(t[i]).split('/')[-2]+'_diff_angle_'+str(angle)+'_'+str(t[i+1]).split('/')[-1]+'-'+str(t[i]).split('/')[-1], atmo)

                    mask = mask[np.ix_(crop_idx + math.ceil(mask.shape[0]/2), math.ceil(mask.shape[1]/2) + crop_idx)]
                    #mask = mask[::-1, :]
                    count += 1
                    
                    #if ~(((np.sum(mask == False) / mask.size) <= 0.1) or ((np.sum(mask == False) / mask.size) >= 0.5)):
                    if wrapped:
                        wrapped_crop = np.mod(atmo, 2*np.pi) - np.pi # wrap between -pi to +pi
                        norm_wrapped = (wrapped_crop - np.min(wrapped_crop)) / np.ptp(wrapped_crop)
                        norm_wrapped = np.ma.MaskedArray(norm_wrapped, mask=~mask)
                        #norm_wrapped = norm_wrapped * mask

                        output_path = output_dir + 'wrapped/set'+str(2 - (i % 2))+'/'
                        plt.imsave(output_path+str(t[i]).split('/')[-2]+'_wrapped_diff_angle_'+str(angle)+'_'+str(t[i+1]).split('/')[-1]+'-'+str(t[i]).split('/')[-1]+'.png', norm_wrapped, cmap='jet')

                else:
                    break


            if i == (len(t) -2):
                break

            else:
                continue

            break

if __name__ == '__main__':
    gen_strat_samples('/home/conradb/git/insar-syn-gen/configs/insar_synthetic_ifg.toml')
