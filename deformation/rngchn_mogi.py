# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/rngchn_mogi.m
# TJW 2009 & Nantheera Anantrasirichai 2021

#  INPUT: n1 = local north coord of center of Mogi source (km)
#         e1 = local east coord of center of Mogi source (km)
#         depth = depth of Mogi source (km) for points to calculate
#                  range change. This vector is the depth of the
#                  mogi source plus the elevation of the point
#                  taken from the DEM.
#         del_v = Volume change of Mogi source (km^3)
#         ning = north coord's of points to calculate range change
#         eing = east coord's of points to calculate range change
#         v = Poisson's ration of material

#  OUTPUT: del_rng = range change at coordinates given in ning and eing.
#                    If ning and eing are vectors with the same dimensions,
#                    del_rng is a vector. If ning is a row vector and eing
#                    is a column vecor, del_rng is a matrix of deformation
#                    values...compliant with Feigle and Dupre's useage.

import numpy as np
import sys

def rngchn_mogi(n1, e1, depth, del_v, ning, eing, v, plook):

    m, n = ning.shape
    mm, nn = eing.shape

    # Poisson's ration v (tjw nov 09)
    dsp_coef = 1000000 * del_v * (1-v)/np.pi
    
    #easting are single row and northings are single column
    if mm == 1 and n ==1:
        print('Calculating a matrix of rngchg values for easting row vector and northing column vector')

        del_rng = np.zeros((m, nn)) #2d easting/northing matrix
        del_d = del_rng
        del_f = del_rng
        tmp_n = del_rng
        tmp_e = del_rng

        for i in range(m):
            tmp_e[i, :] = eing
        for i in range(nn):
            tmp_n[:, i] = ning

        d_mat = np.sqrt((tmp_n - n1)**2 + (tmp_e - e1)**2)
        tmp_hyp = ((d_mat**2 + depth**2)**1.5)
        del_d = dsp_coef * d_mat / tmp_hyp
        del_f = dsp_coef * depth / tmp_hyp
        azim = np.arctan2((tmp_e - e1),(tmp_n - n1))
        e_disp = np.sin(azim) * del_d
        n_disp = np.cos(azim) * del_d

        for i in range(nn):
            del_rng[:, i] = np.array([e_disp[:, i], n_disp[:, i], del_f[:, i]]) * plook.T

        return del_rng

    #easting are single row/column and northings are single row/column
    elif (mm ==1 and m == 1) | (n == 1 and nn == 1):
        if n != nn:
            print('Coord vectors not equal length')
            sys.exit()
        print('Calculating a matrix of rngchg values for matching easting/northing row/column vectors')
        del_rng = np.zeros(m)
        del_d = del_rng
        del_f = del_rng
        d_mat = np.sqrt((ning - n1)**2 + (eing - e1)**2)
        tmp_hyp = ((d_mat**2 + depth**2)**1.5)
        del_d = dsp_coef * d_mat / tmp_hyp
        del_f = dsp_coef * depth / tmp_hyp
        azim = np.arctan2((eing - e1),(ning - n1))
        e_disp = np.sin(azim) * del_d
        n_disp = np.cos(azim) * del_d
        del_rng = np.column_stack((e_disp, n_disp, del_f)) * plook
        print(del_rng.shape)
        del_rng = np.sum(del_rng.T, axis=0)
        print(del_rng.shape)
        del_rng = del_rng[:, None]
        print(del_rng.shape)
        del_rng = -1.0 * del_rng / 1000 # convert from mm to m
        #del_rng = -1.0 * del_rng #mm

        return del_rng

    else:
        print('Coord vectors make no sense')
        sys.exit()
