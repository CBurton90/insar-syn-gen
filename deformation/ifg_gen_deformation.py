# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/generateDeformation.m
# TJW 2009 & Nantheera Anantrasirichai 2021

import math
import matplotlib.pyplot as plt
import munch
import numpy as np
import toml

import rngchn_mogi
import disloc3d4

# Hardcoded parameters
# Quakes - top depth (dependent on bottom depth)
# Mogi (point) - depth range (dependent on volume)

# All other parameters can be modified in the TOML config


def scale_crop(los_grid, halfcrop):

    los_grid = los_grid/0.028333 * 2 * np.pi # scale to radians
    crop_idx = np.arange(-halfcrop, halfcrop)
    los_grid = los_grid[np.ix_(crop_idx + math.ceil(los_grid.shape[0]/2), math.ceil(los_grid.shape[1]/2) + crop_idx)]

    return los_grid

def save_ifg(output_dir, fname, los_grid, wrapped=True):
    
    output_path = output_dir + 'unwrapped/set2/'
    np.save(output_path + fname, los_grid)

    if wrapped:
        wrapped_pn_pi = np.mod(los_grid, 2*np.pi) - np.pi # wrap between -pi to +pi
        wrapped_norm = (wrapped_pn_pi - np.min(wrapped_pn_pi)) / np.ptp(wrapped_pn_pi) # min max normalize
        output_path = output_dir + 'wrapped/set2/'
        fig = plt.figure(figsize=(10,10))
        plt.imshow(wrapped_norm, cmap='jet', vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(output_path + fname+'.png')

def gen_def(source, x, y, mogi, quake, dyke, heading, incidence):
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

    if source == 1:
        model = np.array([[1], [1], [quake['strike']], [quake['dip']], [quake['rake']], [quake['slip']], [quake['length']*1000], [quake['top_depth']*1000], [quake['bottom_depth']*1000], [1]])
    elif source == 2:
        model = [1, 1, dyke['strike'], dyke['dip'], 0, dyke['opening'], dyke['length']*1000, dyke['top_depth']*1000, dyke['bottom_depth']*1000, 2]
    elif source == 3:
        model = [1, 1, sill['strike'], sill['dip'], 0, sill['opening'], sill['length']*1000, sill['depth']*1000, sill['width']*1000, 3]
    elif source == 4 or source == 5:
        model = []
    else:
        print('please enter a valid source type')

    if source == 1 or source == 2 or source == 3:
        U, flag = disloc3d4.disloc3d4(model, coords, lmbd, mu)
        xgrid = U[0, :].reshape((y.size, x.size))
        ygrid = U[1, :].reshape((y.size, x.size))
        zgrid = U[2, :].reshape((y.size, x.size))

        los_grid = xgrid*los_vec[0] + ygrid*los_vec[1] + zgrid*los_vec[2]
    
    elif source == 4:
        xgrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(np.array([1, 0, 0]), (coords.shape[1], 1))) # volume converted to km^3, easting/northing converted from m to km
        ygrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(np.array([0, 1, 0]), (coords.shape[1], 1))) # volume converted to km^3, easting/northing converted from m to km
        zgrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(np.array([0, 0, 1]), (coords.shape[1], 1))) # volume converted to km^3, easting/northing converted from m to km
        los_grid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:].T/1000, coords[0,:].T/1000, v, np.tile(los_vec, (coords.shape[1], 1)))
        los_grid = np.reshape(los_grid, (len(y),len(x)))

    else:
        print('please enter a valid source type')

    los_grid_wrap = (np.mod(los_grid+10000, 0.028333) / 0.028333) *2*np.pi # wrap S1 C-band if grid def in m
            
    return los_grid_wrap, los_grid

def create_def_samples(source, x, y, mogi, quake, dyke, halfcrop, config):

    max_num = config.deformation.max_samples - 1
    count = 0

    if source == 1:

        quake['slip'] = config.quake.slip
        heading = config.quake.heading
        incidence = config.quake.incidence

        for strike in config.quake.strikes:
            quake['strike'] = strike
            strk_name = 'strike' + str(strike)

            for dip in config.quake.dips:
                quake['dip'] = dip
                dip_name = strk_name + '_dip' + str(dip)

                for rake in config.quake.rakes:
                    quake['rake'] = rake
                    rake_name = dip_name + '_rake' + str(rake)

                    for length in config.quake.lengths:
                        quake['length'] = length 
                        len_name = rake_name + '_len' + str(length)

                        for bdepth in config.quake.bottom_depths:
                            quake['bottom_depth'] = bdepth
                            bdepth_name = len_name + '_bdepth' + str(bdepth)

                            if bdepth == 6:
                                top_depth_range = [2.5, 3.0, 3.5]
                            else:
                                top_depth_range = [3., 4., 5.]

                            for tdepth in top_depth_range:
                                quake['top_depth'] = tdepth
                                fname = bdepth_name + '_tdepth' + str()

                                if count < max_num:

                                    _, los_grid = gen_def(source, x, y, mogi, quake, dyke, heading, incidence)
                                    los_grid = scale_crop(los_grid, halfcrop)

                                    print('Quake range in radians', np.ptp(los_grid))
                                    if config.quake.rad_min < np.ptp(los_grid) < config.quake.rad_max:
                                        save_ifg(config.deformation.output_path, fname, los_grid, wrapped=True)
                                        count +=1
                                        print(f'Generated Quake sample {count+1} of 5000')
                                    else:
                                        continue
                                else:
                                    return

    if source == 4:
        
        for incidence in config.mogi.incidences:
            inc_name = 'incidence_' + str(incidence)
            for heading in config.mogi.headings:
                hdg_name = inc_name + '_heading_' + str(heading)
                for vol in config.mogi.volumes:
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
                        fname = vol_name + '_depth_' + str(depth)
                        mogi['depth'] = depth

                        for k, v in mogi.items():
                            print(f'key is {k}, value is {v}')

                        if count < max_num:

                            _, los_grid = gen_def(source, x, y, mogi, quake, dyke, heading, incidence)
                            los_grid = scale_crop(los_grid, halfcrop)

                            print('Mogi range in radians', np.ptp(los_grid))
                            if config.mogi.rad_min  < np.ptp(los_grid) < config.mogi.rad_max:
                                save_ifg(config.deformation.output_path, fname, los_grid, wrapped=True)
                                print(f'Generated Mogi sample {count+1} of 5000')
                                count +=1
                            else:
                                continue
                        else:
                            return

if __name__ == '__main__':

    # initial params dictionary which we will overwrite
    
    quake = {
        'slip': 1,
        'strike': 0,
        'dip': 80,
        'rake': -90,
        'top_depth': 3,
        'bottom_depth': 6,
        'length': 2,
        }

    dyke = {
        'strike': 0,
        'dip': 0,
        'opening': 0,
        'length': 1,
        'top_depth': 1,
        'bottom_depth': 1,
        }
  
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
    halfcrop = config.deformation.crop // 2

    create_def_samples(1, x, y, mogi, quake, dyke, halfcrop, config) # Quake
    create_def_samples(4, x, y, mogi, quake, dyke, halfcrop, config) # Mogi


