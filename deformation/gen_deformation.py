# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/generateDeformation.m
# TJW 2009 & Nantheera Anantrasirichai 2021

import numpy as np
import rngchn_mogi

def gen_def(source, x, y, mogi, incidence):

    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))
    coords = np.array([xx.T, yy.T])

    if source == 4:
        xgrid = rngchn_mogi.rngchn_mogi(1/1000, 1/1000, mogi['depth'], -mogi['volume']/1e9, coords[1,:], coords[0,:], v, plook) # volume converted to km^3
        ygrid = rngchn_mogi.rngchn_mogi(n1, e1, depth, del_v, ning, eing, v, plook) # volume converted to km^3
        zgrid = rngchn_mogi.rngchn_mogi(n1, e1, depth, del_v, ning, eing, v, plook) # volume converted to km^3
        los_grid = rngchn_mogi(n1, e1, depth, del_v, ning, eing, v, plook)

    return

# define elastic lame params
 = 

mogi = {
        'depth' : 5, # depth in km
        'volume': 10*1e6, # volume in m^3
        }

