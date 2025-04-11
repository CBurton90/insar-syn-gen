# Copyright TJW
# https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/disloc3d4.m

import numpy as np
import matplotlib.pyplot as plt
import dc3d4

def disloc3d4(m, x, lmbd, mu):

    deg2rad = (2*np.pi/360)
    nfaults = m.shape[1]
    nx = x.shape[2]

    U = np.zeros((3, nx))

    alpha = (lmbd + mu) / (lmbd + 2*mu)

    for i in range(nfaults):
        
        # convert for dc3d3
        flt_x = np.ones((1,nx))*m[0,i];
        flt_y = np.ones((1,nx))*m[1,i];
        strike = m[2,i]
        dip = m[3,i]
        rake = m[4,i]
        slip = m[5,i]
        length = m[6,i]
        hmin = m[7,i]
        hmax = m[8,i]

        # make special case for sills
        if m[9,i] == 3:
            dip = m[3,i]
            fdepth = hmin;
            aw1= -hmax/2;
            aw2= hmax/2;
            w= hmax;
        else:
            sindip = np.sin(dip*deg2rad)
            w = (hmax-hmin)/sindip
            aw1 = np.ones((1,nx))*hmin/sindip
            aw2 = np.ones((1,nx))*hmax/sindip

        rrake = (rake + 90)*deg2rad
        ud = np.ones((1,nx)) * slip * np.cos(rrake)
        us = np.ones((1,nx)) * -slip * np.sin(rrake)
        opening = np.ones((1,nx)) * slip
        halflen = length/2
        al2 = np.ones((1,nx)) * halflen
        al1 = -al2
        
        #reject data which breaks the surface
        if np.sum(hmin < 0) > 0:
            print('ERROR: Fault top above ground surface')

        hmin= hmin + (hmin == 0)*0.00001
    
        sstrike = (strike + 90) * deg2rad
        ct = np.cos(sstrike)
        st = np.sin(sstrike)
    
        #loop over points
    
        X= ct * (-flt_x + x[0, :]) - st * (-flt_y + x[1, :])
        Y= ct * (-flt_y + x[1, :]) + st * (-flt_x + x[0, :])

        if m[9, i] == 1:
             ux, uy, uz, err = dc3d4.dc3d4(alpha, X, Y, -dip, al1, al2, aw1, aw2, us, ud)

        if err != 0:
            print('error with dc3d3')
            flag = err
            return None

        else:
            flag = 0

        U[0, :] = U[0, :] + ct*ux + st*uy
        U[1, :] = U[1, :] -st*ux + ct*uy
        U[2, :] = U[2, :] + uz

    return U, flag

if __name__ == '__main__':

    quake = {
    'strike': 180,
    'dip': 70,
    'rake': -90,
    'slip': 5,
    'length': 30,
    'top_depth': 1,
    'bottom_depth': 30,
    }

    m = np.array([[1], [1], [quake['strike']], [quake['dip']], [quake['rake']], [quake['slip']], [quake['length']*1000], [quake['top_depth']*1000], [quake['bottom_depth']*1000], [1]])
    x = np.arange(-25000, 25000, 100)
    y = np.arange(-25000, 25000, 100)
    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape((-1, 1))
    yy = yy.reshape((-1, 1))
    coords = np.array([xx.T, yy.T])

    # define elastic lame params
    lmbd = 2.3e10 # units = pascals
    mu = 2.3e10

    U, flag = disloc3d4(m, coords, lmbd, mu)
    xgrid = U[0, :].reshape((y.size, x.size))
    ygrid = U[1, :].reshape((y.size, x.size))
    zgrid = U[2, :].reshape((y.size, x.size))

    # calc LOS vector from heading & incidence angle
    incidence = 29
    heading = 20
    sat_inc = 90 - incidence
    sat_az = 360 - heading

    los_x = -np.cos(np.deg2rad(sat_az)) * np.cos(np.deg2rad(sat_inc))
    los_y = -np.sin(np.deg2rad(sat_az)) * np.cos(np.deg2rad(sat_inc))
    los_z = np.sin(np.deg2rad(sat_inc))
    los_vec = np.array([los_x, los_y, los_z])
    los_grid = xgrid*los_vec[0] + ygrid*los_vec[1] + zgrid*los_vec[2]

    fig = plt.figure(figsize=(10,10))
    plt.imshow(los_grid, cmap='viridis')
    plt.colorbar()
    plt.savefig('quake_test.png')




