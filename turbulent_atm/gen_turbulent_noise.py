import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
from cholesky_decomp import create_turb_del

output_dir = '../test_outputs/turbulent_atm_noise/'
wrapped = True
rows = 100
cols = 100
psizex = 1
psizey = 1
covmodel = 0
N = 10
halfcrop = 224 // 2 # Alexnet input size 227x227
img_size = 500 # resolution in pixels

maxvariance = [i + 7.5 for i in [-2, -1, 0, 0.75, 1.5]]
alpha = [i * 0.008 for i in  [0.5, 0.75, 1, 1.5, 2]]

for var in maxvariance:
    for a in alpha:
        turb_atm = create_turb_del(rows, cols, var, a, covmodel, N, psizex, psizey)
        for i in range(N):
            full_res = cv2.resize(turb_atm[:, : ,i], (img_size, img_size)) # note MATLAB imresize is hard to replicate https://stackoverflow.com/questions/29958670/how-to-use-matlabs-imresize-in-python
            crop_idx = np.arange(-halfcrop, halfcrop)
            crop = full_res[np.ix_(crop_idx + math.ceil(full_res.shape[0]/2), math.ceil(full_res.shape[1]/2) + crop_idx)]

            output_path = output_dir + 'unwrapped/set'+str(2-(i % 2))+'/'
            #plt.imsave(output_path+'turb_'+str(var)+'_'+str(a)+'_'+str(i)+'.png', crop, cmap='jet')
            np.save(output_path+'turb_'+str(var)+'_'+str(a)+'_'+str(i), crop)

            if wrapped:
                wrapped_crop = np.angle(np.exp(1j * crop))
                norm_wrapped = wrapped_crop - np.min(wrapped_crop) / (np.max(wrapped_crop) - np.min(wrapped_crop))
                output_path = output_dir + 'wrapped/set'+str(2-(i % 2))+'/'
                plt.imsave(output_path+'turb_'+str(var)+'_'+str(a)+'_'+str(i)+'.png', norm_wrapped, cmap='jet')


