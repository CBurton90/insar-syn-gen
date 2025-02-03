# Copyright (C) 2015
# %     Email: eedpsb@leeds.ac.uk or davidbekaert.com
# %     With permission by Peter Cervelli, Jessica Murray

# Modified for Python by C Burton 2025
# https://github.com/dbekaert/TRAIN/blob/master/matlab/llh2local.m

# %Converts from longitude and latitude to local coorindates
# %given an origin.  llh (lon; lat; height) and origin should
# %be in decimal degrees. Note that heights are ignored and
# %that xy is in km.

import numpy as np

def llh2local(llh, origin):

    # Set ellipsoid constants (WGS84)
    a=6378137.0
    e=0.08209443794970

    # Convert to radians
    llh = np.float64(np.deg2rad(llh))
    print(llh)
    origin = np.float64(np.deg2rad(origin))

    # Do the projection
    z = llh[1, :] != 0
    print(z)

    dlambda = llh[0, z] - origin[0]

    M = a * ((1-np.exp(2)/4-3 * np.exp(4)/64-5 * np.exp(6)/256) * llh[1,z] - (3*np.exp(2)/8+3 * np.exp(4)/32+45 * np.exp(6)/1024)*np.sin(2*llh[1,z]) + (15*np.exp(4)/256 +45*np.exp(6)/1024)*np.sin(4*llh[1,z]) - (35*np.exp(6)/3072)*np.sin(6*llh[1,z]))

    M0 =a*((1-np.exp(2)/4-3*np.exp(4)/64-5*np.exp(6)/256)*origin[1] - (3*np.exp(2)/8+3*np.exp(4)/32+45*np.exp(6)/1024)*np.sin(2*origin[1]) + (15*np.exp(4)/256 +45*np.exp(6)/1024)*np.sin(4*origin[1]) - (35*np.exp(6)/3072)*np.sin(6*origin[1]))

    print(np.sin(llh[1,z]**2))
    print(np.exp(2) * np.sin(llh[1,z]**2))
    print(1 - (np.exp(2) * np.sin(llh[1,z]**2)))
    N = a / (np.sqrt(-1*(1 - np.exp(2) * np.sin(llh[1,z])**2)))
    print(N)
    E = dlambda * np.sin(llh[1,z])
    print(E)

    cot = 1 / np.tan(llh[1,z])

    xy = np.zeros_like(llh)

    xy[0, z] = N * cot * np.sin(E)
    xy[1, z] = M - M0 + N * cot *(1-np.cos(E))

    # Handle special case of latitude = 0

    xy[0, ~z] = a * dlambda[~z]
    xy[1, ~z] = -M0

    return xy
