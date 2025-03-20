# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/main_code/runGenStratified.m
# TJW 2004 & Nantheera Anantrasirichai 2021

import cv2
import glob
import math
import munch
import numpy as np
import matplotlib.pyplot as plt
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

            print(f'i is {i}')

            if i == (len(t) -2):
                break

if __name__ == '__main__':
    gen_strat_samples('/home/conradb/git/insar-syn-gen/configs/insar_synthetic_ifg.toml')
