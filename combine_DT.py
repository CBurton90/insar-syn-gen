import os



def main(root_dir, class_samples):

    for set in range(1,3):
        unwrapped_dir = 'unwrapped/set'+str(set)+'/'
        wrapped_dir = 'wrapped/set'+str(set)+'/'

        deform_list = os.listdir(root_dir+'deformation/'+unwrapped_dir)
        turbulent_list = os.listdir(root_dir+'turbulent_atm_noise/'+unwrapped_dir)

        print(deform_list)
        print(turbulent_list)



if __name__ == '__main__':
    root_dir = 'test_outputs/'
    main(root_dir, '')
