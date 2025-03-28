# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/generateDeformation.m
# TJW 2009 & Nantheera Anantrasirichai 2021

import math
import matplotlib.pyplot as plt
import munch
import numpy as np
import toml

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

def create_def_samples(source, x, y, mogi, volume_lst, heading_lst, incidence_lst, halfcrop, output_dir, config):

    max_num = config.deformation.max_samples - 1
    count = 0

    for inc in incidence_lst:
        inc_name = 'incidence_' + str(inc)
        for hdg in heading_lst:
            hdg_name = inc_name + '_heading_' + str(hdg)
            for vol in volume_lst:
                vol_name = hdg_name + '_vol-1e' + str(vol)
                mogi['volume'] = 10 ** vol # volume in m^3

                if vol <= 6:
                    depth_range = [1.5, 2.]
                elif vol <= 6.5:
                    depth_range = [2., 2.5, 2.8, 3.]
                elif vol <= 7:
                    depth_range = [2.5, 2.8, 3., 4., 5., 5.5, 6.]
                else:
                    depth_range = [5., 6., 7., 7.5, 8.]

                for depth in depth_range:
                    file_name = vol_name + '_depth_' + str(depth)
                    mogi['depth'] = depth

                    for k, v in mogi.items():
                        print(f'key is {k}, value is {v}')

                    if count < max_num:

                        wrapped_grid, los_grid = gen_def(source, x, y, mogi, hdg, inc)

                        if config.deformation.min_def_range /1000  < np.ptp(los_grid) < config.deformation.max_def_range / 1000:
                            print('range in mm/yr ', np.ptp(los_grid) * 1000)
                            los_grid = los_grid/0.028333 * 2 * np.pi # scale to radians

                            crop_idx = np.arange(-halfcrop, halfcrop)
                            los_grid = los_grid[np.ix_(crop_idx + math.ceil(los_grid.shape[0]/2), math.ceil(los_grid.shape[1]/2) + crop_idx)]
                            wrapped_grid = wrapped_grid[np.ix_(crop_idx + math.ceil(wrapped_grid.shape[0]/2), math.ceil(wrapped_grid.shape[1]/2) + crop_idx)]

                            output_path = output_dir + 'unwrapped/set2/'
                            np.save(output_path+file_name, los_grid)
                            print(f'generating sample {count} / 10000')
                            count += 1

                            if config.deformation.wrapped:
                                wrapped_pn_pi = np.mod(los_grid, 2*np.pi) - np.pi # wrap between -pi to +pi
                                wrapped_norm = (wrapped_pn_pi - np.min(wrapped_pn_pi)) / np.ptp(wrapped_pn_pi) # min max normalize
                                output_path = output_dir + 'wrapped/set2/'
                                fig = plt.figure(figsize=(10,10))
                                plt.imshow(wrapped_norm, cmap='jet', vmin=0, vmax=1) # if gdef in mm
                                plt.colorbar()
                                plt.savefig(output_path+file_name+'.png')

                        else:
                            continue

if __name__ == '__main__':

    # initial params dictionary which we will overwrite
    mogi = {
            'depth' : 5, # depth in km
            'volume': 10*1e6 # volume in m^3
            }

    config = '/home/conradb/git/insar-syn-gen/configs/insar_synthetic_ifg.toml'
    config = munch.munchify(toml.load(config))
    max_dist = config.deformation.max_dist
    p_size = config.deformation.pixel_size
    x = np.arange(-max_dist, max_dist, p_size)
    y = np.arange(-max_dist, max_dist, p_size)
    source = config.deformation.source # Mogi source
    volume_lst = config.deformation.volumes
    heading_lst = config.deformation.headings
    incidence_lst = config.deformation.incidences
    halfcrop = config.deformation.crop // 2
    output_dir = config.deformation.output_path

    create_def_samples(source, x, y, mogi, volume_lst, heading_lst, incidence_lst, halfcrop, output_dir, config)


