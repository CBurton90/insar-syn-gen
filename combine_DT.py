from glob import glob
import munch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import toml
#from skimage.morphology import erosion, disk
#import cv2


def main(config):
    
    config = munch.munchify(toml.load(config))
    class_samples = config.combine.samples
    root_dir = config.combine.root_dir
    

    for set in range(1,3):
        unwrapped_dir = 'unwrapped/set'+str(set)+'/'
        wrapped_dir = 'wrapped/set'+str(set)+'/'

        deform_list = glob(root_dir+'deformation/unwrapped/set2/*.npy')
        turbulent_list = glob(root_dir+'turbulent_atm_noise/'+unwrapped_dir+'*.npy')
        
        noise_list = glob(root_dir+'spike_noise/*.npy')
        decoh_list = glob(root_dir+'decoh_mask/*.npy')

        idx_def = np.random.choice(len(deform_list), class_samples, replace = False)
        print(idx_def)
        idx_trb= np.random.choice(len(turbulent_list), class_samples, replace = False)
        print(idx_trb)

        idx_noise = np.random.choice(len(noise_list), class_samples, replace = True)
        idx_decoh = np.random.choice(len(decoh_list), class_samples, replace = True)

        output_dir_unwrap = root_dir+unwrapped_dir
        output_dir_wrap = root_dir+wrapped_dir
        
        # class 0 / set 1 = no deformation
        if set == 1:
            for k in range(class_samples):
                tur_del = np.load(turbulent_list[idx_trb[k]])
                angle = np.random.randint(0, 360)
                noise = np.load(noise_list[idx_noise[k]])
                noise = (noise / 1000) / 0.028333 * 2 * np.pi # scale noise from mm/yr to m/yr to radians
                noise = rotate(noise, angle, reshape=False, cval=0)
                insar_img = tur_del + noise
                insar_wrapped = np.mod(insar_img, 2*np.pi) - np.pi
                insar_wrapped = (insar_wrapped - np.min(insar_wrapped)) / np.ptp(insar_wrapped) # min-max normalize
                tur_str = turbulent_list[idx_trb[k]].split('/')[-1].split('.npy')[0]
                angle = np.random.randint(0, 360)
                decoh = np.load(decoh_list[idx_decoh[k]])
                decoh = rotate(decoh, angle, reshape=False, cval=False)
                insar_unwrapped = np.ma.MaskedArray(insar_img, mask=decoh)
                insar_unwrapped_filled = np.ma.filled(insar_unwrapped, 0)
                insar_wrapped = np.ma.MaskedArray(insar_wrapped, mask=decoh)
                np.save(output_dir_unwrap+'unwrapped_T_'+tur_str, insar_unwrapped_filled)
                plt.imsave(output_dir_wrap+'wrapped_T_'+tur_str+'.png', insar_wrapped, cmap='jet', vmin=0, vmax=1)
                plt.imsave(output_dir_unwrap+'unwrapped_T_'+tur_str+'.png', insar_unwrapped, cmap='jet', vmin=-20, vmax=20)

        # class 1 / set 2 = deformation
        else:
            for k in range(class_samples):
                
                los_grid = np.load(deform_list[idx_def[k]])

                # TO DO - check why certain los ranges are normalized in:
                # https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/da224d217ae95d0198d4b8514b96b8f0328ca1b9/main_code/runGenCombineSignals.m#L32
                #print(np.ptp(los_grid))

                tur_del = np.load(turbulent_list[idx_trb[k]])
                insar_img = los_grid + tur_del

                #noise = np.random.default_rng().uniform(0.2*np.min(insar_img), 0.3*np.max(insar_img), int(0.03*insar_img.size))
                #zeros = np.zeros(insar_img.size - len(noise))
                #noise = np.concatenate([noise, zeros])
                #np.random.shuffle(noise)

                angle = np.random.randint(0, 360)

                noise = np.load(noise_list[idx_noise[k]])
                noise = (noise / 1000) / 0.028333 * 2 * np.pi # scale noise from mm/yr to m/yr to radians
                noise = rotate(noise, angle, reshape=False, cval=0)

                #insar_img = insar_img + noise.reshape((224,224))
                insar_img = insar_img + noise

                insar_wrapped = np.mod(insar_img, 2*np.pi) - np.pi
                insar_wrapped = (insar_wrapped - np.min(insar_wrapped)) / np.ptp(insar_wrapped) # min-max normalize
                def_str = deform_list[idx_def[k]].split('/')[-1].split('.npy')[0]
                tur_str = turbulent_list[idx_trb[k]].split('/')[-1].split('.npy')[0]

                #dsk = disk(3)
                #test = erosion(((tur_del > -0.5) & (tur_del < 0.5)).astype(int), dsk)
                #test = rotate(test, angle, reshape=False, cval=0)

                angle = np.random.randint(0, 360)
                decoh = np.load(decoh_list[idx_decoh[k]])
                decoh = rotate(decoh, angle, reshape=False, cval=False)


                #mask = np.random.choice([True, False], insar_wrapped.shape, p=[0.6, 0.4])
                #insar_wrapped = np.ma.MaskedArray(insar_wrapped, mask=mask)

                #insar_unwrapped = np.ma.MaskedArray(insar_img, mask=test.astype(bool))
                insar_unwrapped = np.ma.MaskedArray(insar_img, mask=decoh)
                insar_unwrapped_filled = np.ma.filled(insar_unwrapped, 0)
                #insar_wrapped = np.ma.MaskedArray(insar_wrapped, mask=test.astype(bool))
                insar_wrapped = np.ma.MaskedArray(insar_wrapped, mask=decoh)
                np.save(output_dir_unwrap+'unwrapped_DT_'+def_str+'_'+tur_str, insar_unwrapped_filled)

            
                plt.imsave(output_dir_wrap+'wrapped_DT_'+def_str+'_'+tur_str+'.png', insar_wrapped, cmap='jet', vmin=0, vmax=1)
                plt.imsave(output_dir_unwrap+'unwrapped_DT_'+def_str+'_'+tur_str+'.png', insar_unwrapped, cmap='jet', vmin=-20, vmax=20)

if __name__ == '__main__':
    config = '/home/conradb/git/insar-syn-gen/configs/insar_synthetic_vel.toml'
    main(config)
