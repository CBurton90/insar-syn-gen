from glob import glob
import munch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import toml

def main(config):

    config = munch.munchify(toml.load(config))
    class_samples = config.combine.samples
    root_dir = config.combine.root_dir

    for set in range(1,3):
        unwrapped_dir = 'unwrapped/set'+str(set)+'/'
        wrapped_dir = 'wrapped/set'+str(set)+'/'
        deform_list = glob(root_dir+'deformation/unwrapped/set2/*.npy')
        turbulent_list = glob(root_dir+'turbulent_atm_noise/'+unwrapped_dir+'*.npy')
        stratified_list = glob(root_dir+'stratified_atm_noise/'+unwrapped_dir+'*.npy')

        idx_def = np.random.choice(len(deform_list), class_samples, replace = False)
        idx_trb = np.random.choice(len(turbulent_list), class_samples, replace = False)
        idx_str = np.random.choice(len(stratified_list), class_samples, replace = True)

        output_dir_unwrap = root_dir+'combined/'+unwrapped_dir
        output_dir_wrap = root_dir+'combined/'+wrapped_dir

        # class 0 / set 1 = no deformation
        if set == 1:
            for k in range(class_samples):
                tur_del = np.load(turbulent_list[idx_trb[k]])
                str_del = np.load(stratified_list[idx_str[k]])
                insar_img = tur_del + str_del
                insar_wrapped = np.mod(insar_img, 2*np.pi) - np.pi
                insar_wrapped = (insar_wrapped - np.min(insar_wrapped)) / np.ptp(insar_wrapped) # min-max normalize
                tur_str = turbulent_list[idx_trb[k]].split('/')[-1].split('.npy')[0]
                str_str = stratified_list[idx_str[k]].split('/')[-1].split('.npy')[0]
                np.save(output_dir_unwrap+'unwrapped_ST_'+str_str+'_'+tur_str, insar_img)
                plt.imsave(output_dir_unwrap+'unwrapped_ST_'+str_str+'_'+tur_str+'.png', insar_img, cmap='jet', vmin=-20, vmax=20)
                plt.imsave(output_dir_wrap+'wrapped_ST_'+str_str+'_'+tur_str+'.png', insar_wrapped, cmap='gray', vmin=0, vmax=1)
        
        # class 1 / set 2 = deformation
        else:
            for k in range(class_samples):
                los_grid = np.load(deform_list[idx_def[k]])
                tur_del = np.load(turbulent_list[idx_trb[k]])
                str_del = np.load(stratified_list[idx_str[k]])

                if np.ptp(los_grid) <= 15:
                    los_grid = los_grid*18/np.ptp(los_grid) # scale up small deformation
                elif np.ptp(los_grid) => 50:
                    los_grid = los_grid*40/np.ptp(los_grid) # scale down large deformation
                else:
                    los_grid = los_grid

                insar_img = los_grid + str_del + tur_del
                insar_wrapped = np.mod(insar_img, 2*np.pi) - np.pi
                insar_wrapped = (insar_wrapped - np.min(insar_wrapped)) / np.ptp(insar_wrapped) # min-max normalize
                def_str = deform_list[idx_def[k]].split('/')[-1].split('.npy')[0]
                tur_str = turbulent_list[idx_trb[k]].split('/')[-1].split('.npy')[0]
                str_str = stratified_list[idx_str[k]].split('/')[-1].split('.npy')[0]
                np.save(output_dir_unwrap+'unwrapped_DST_'+def_str+'_'+str_str+'_'+tur_str, insar_img)
                plt.imsave(output_dir_unwrap+'unwrapped_DT_'+def_str+'_'+str_str+'_'+tur_str+'.png', insar_img, cmap='jet', vmin=-20, vmax=20)
                plt.imsave(output_dir_wrap+'wrapped_DT_'+def_str+'_'+str_str+'_'+tur_str+'.png', insar_wrapped, cmap='gray', vmin=0, vmax=1)

if __name__ == '__main__':
    config = '/home/conradb/git/insar-syn-gen/configs/insar_synthetic_ifg.toml'
    main(config)

