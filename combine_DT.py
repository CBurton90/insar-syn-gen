from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.morphology import erosion, disk
#import cv2


def main(root_dir, class_samples):

    for set in range(1,3):
        unwrapped_dir = 'unwrapped/set'+str(set)+'/'
        wrapped_dir = 'wrapped/set'+str(set)+'/'

        deform_list = glob(root_dir+'deformation/'+unwrapped_dir+'*.npy')
        print(deform_list)
        turbulent_list = glob(root_dir+'turbulent_atm_noise/'+unwrapped_dir+'*.npy')


        idx_def = np.random.choice(len(deform_list), class_samples, replace = False)
        print(idx_def)
        idx_trb= np.random.choice(len(turbulent_list), class_samples, replace = False)
        print(idx_trb)

        output_dir_unwrap = root_dir+'combined/'+unwrapped_dir
        output_dir_wrap = root_dir+'combined/'+wrapped_dir

        for k in range(class_samples):
            los_grid = np.load(deform_list[idx_def[k]])

            # TO DO - check why certain los ranges are normalized in:
            # https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/da224d217ae95d0198d4b8514b96b8f0328ca1b9/main_code/runGenCombineSignals.m#L32
            #print(np.ptp(los_grid))

            tur_del = np.load(turbulent_list[idx_trb[k]])
            insar_img = los_grid + tur_del

            noise = np.random.default_rng().uniform(0.2*np.min(insar_img), 0.3*np.max(insar_img), int(0.03*insar_img.size))
            zeros = np.zeros(insar_img.size - len(noise))
            noise = np.concatenate([noise, zeros])
            np.random.shuffle(noise)
            insar_img = insar_img + noise.reshape((224,224))

            insar_wrapped = np.mod(insar_img, 2*np.pi) - np.pi
            def_str = deform_list[idx_def[k]].split('/')[-1].split('.npy')[0]
            tur_str = turbulent_list[idx_trb[k]].split('/')[-1].split('.npy')[0]

            dsk = disk(3)
            print(dsk)
            test = erosion(((tur_del > -0.5) & (tur_del < 0.5)).astype(int), dsk)
            print(test.shape)
            angle = np.random.randint(0, 360)
            test = rotate(test, angle, reshape=False, cval=0)
            print(test.shape)

            #mask = np.random.choice([True, False], insar_wrapped.shape, p=[0.6, 0.4])
            #insar_wrapped = np.ma.MaskedArray(insar_wrapped, mask=mask)

            insar_unwrapped = np.ma.MaskedArray(insar_img, mask=test.astype(bool))
            insar_unwrapped_filled = np.ma.filled(insar_unwrapped, 0)
            insar_wrapped = np.ma.MaskedArray(insar_wrapped, mask=test.astype(bool))
            np.save(output_dir_unwrap+'unwrapped_DT_'+def_str+'_'+tur_str, insar_unwrapped_filled)

            
            plt.imsave(output_dir_wrap+'wrapped_DT_'+def_str+'_'+tur_str+'.png', insar_wrapped, cmap='jet')
            plt.imsave(output_dir_unwrap+'unwrapped_DT_'+def_str+'_'+tur_str+'.png', insar_unwrapped, cmap='jet')






if __name__ == '__main__':
    root_dir = 'test_outputs/'
    main(root_dir, 50)
