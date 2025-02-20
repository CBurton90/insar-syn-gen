# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/generateDeformation.m
# TJW 2009 & Nantheera Anantrasirichai 2021

import math
import matplotlib.pyplot as plt
import numpy as np
import rngchn_mogi

def gen_def(source, x, y, mogi, heading, incidence):

    # define elastic lame params
    lmbd = 2.3e10 # units = pascals
    mu = 2.3e10
    v = lmbd / (2 * (lmbd + mu)) 
    
    # calc LOS vector from heading & incidence angle
    sat_inc = 90 - incidence
    sat_az = 360 - heading
    los_x = -np.cos(np.deg2rad(sat_az)) * np.cos(np.deg2rad(sat_inc))
    los_y = -np.sin(np.deg2rad(sat_az)) * np.cos(np.deg2rad(sat_inc))
    los_z = np.sin(np.deg2rad(sat_inc))
    los_vec = np.array([los_x, los_y, los_z])

    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))
    coords = np.array([xx.T, yy.T])

    if source == 4:
        xgrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(np.array([1, 0, 0]), (coords.shape[1], 1))) # volume converted to km^3, easting/northing converted from m to km
        ygrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(np.array([0, 1, 0]), (coords.shape[1], 1))) # volume converted to km^3, easting/northing converted from m to km
        zgrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(np.array([0, 0, 1]), (coords.shape[1], 1))) # volume converted to km^3, easting/northing converted from m to km
        los_grid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(los_vec, (coords.shape[1], 1)))
        los_grid = np.reshape(los_grid, (len(y),len(x)))

        los_grid_wrap = (np.mod(los_grid+10000, 0.028333) / 0.028333) *2*np.pi # wrap S1 C-band if grid def in m
        #los_grid_wrap = np.mod(los_grid+100000, 0.028333*1000) # wrap S1 C-band if grid def in mm

    return los_grid_wrap, los_grid

def create_def_samples(source, x, y, mogi, volume_lst, heading_lst, incidence_lst, halfcrop, output_dir, wrapped=True):
    max_num = 5000
    count = 0
    for inc in incidence_lst:
        inc_name = 'incidence_'+str(inc)
        for hdg in heading_lst:
            hdg_name = inc_name+'_heading_'+str(hdg)
            for vol in volume_lst:
                vol_name = hdg_name+'_vol-1e'+str(vol)
                mogi['volume'] = 10**vol # volume in m^3

                # TO DO - create expert adviced volume changes and depths for small subsidence/uplift in NZ
                if vol <=1:
                    depth_range = [10/1000, 20/1000, 30/1000, 40/1000]

                elif vol <=2:
                    depth_range = [10/1000, 20/1000, 30/1000, 40/1000]

                elif vol <=4:
                    depth_range = [45/1000, 65/1000, 80/1000, 90/1000, 100/1000, 200/1000, 300/1000, 400/1000]

                else:
                    print('Volume outside of range')
                    continue

                for depth in depth_range:
                    file_name = vol_name+'_depth_'+str(depth)
                    mogi['depth'] = depth

                    for k, v in mogi.items():
                        print(f'key is {k}, value is {v}')

                    if count < max_num:

                        wrapped_grid, los_grid = gen_def(source, x, y, mogi, hdg, inc)

                        # scaling to radians
                        los_grid = los_grid/0.028333 * 2 * np.pi

                        crop_idx = np.arange(-halfcrop, halfcrop)
                        los_grid = los_grid[np.ix_(crop_idx + math.ceil(los_grid.shape[0]/2), math.ceil(los_grid.shape[1]/2) + crop_idx)]
                        wrapped_grid = wrapped_grid[np.ix_(crop_idx + math.ceil(wrapped_grid.shape[0]/2), math.ceil(wrapped_grid.shape[1]/2) + crop_idx)]

                        if 10 < np.ptp(los_grid) < 60:

                            output_path = output_dir + 'unwrapped/set'+str(2-(count % 2))+'/'
                            #output_path = output_dir + 'unwrapped/set'+str(2-(count % 2))+'/'
                            np.save(output_path+file_name, los_grid)
                            #fig = plt.figure(figsize=(10,10))
                            #plt.imshow(los_grid, cmap='jet', vmin=-10, vmax=10)
                            #plt.colorbar()
                            #plt.imsave(output_path+file_name+'.png', los_grid, cmap='jet')
                            #plt.savefig(output_path+file_name+'.png')

                            if wrapped:
                                #wrapped_crop = np.angle(np.exp(1j * los_grid))
                                #norm_wrapped = wrapped_crop - np.min(wrapped_crop) / (np.max(wrapped_crop) - np.min(wrapped_crop))
                                wrapped_pn_pi = np.mod(los_grid, 2*np.pi) - np.pi # wrap between -pi to +pi
                                output_path = output_dir + 'wrapped/set'+str(2-(count % 2))+'/'
                                fig = plt.figure(figsize=(10,10))
                                #plt.imshow(wrapped_grid, cmap='jet', vmin=0, vmax=0.02833) # if def in m
                                plt.imshow(wrapped_pn_pi, cmap='jet', vmin=-np.pi, vmax=np.pi) # if gdef in mm
                                plt.colorbar()
                                plt.savefig(output_path+file_name+'.png')

                        count += 1






mogi = {
        'depth' : 5, # depth in km
        'volume': 10*1e6, # volume in m^3
        }

volume_lst = [0.3, 0.5, 1, 1.5, 1.6, 1.8, 2, 2.5, 2.7, 3, 3.2, 3.5, 4]
heading_lst = [5, 15, 20, 25, 30]
incidence_lst = [29,30,31]
halfcrop = 224 // 2
output_dir = '../test_outputs/deformation/'

if __name__ == '__main__':
    x = np.arange(-2500, 2500, 10)
    y = np.arange(-2500, 2500, 10)
    #_, los_grid = gen_def(4, x, y, mogi, 20, 34)
    create_def_samples(4, x, y, mogi, volume_lst, heading_lst, incidence_lst, halfcrop, output_dir, wrapped=True)


