# Copyright TJW
# https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/deform/dc3d4.m

import numpy as np

def dc3d4(alpha, x, y, dip, al1, al2, aw1, aw2, disl1, disl2):

    # constants

    F0 = 0
    F1 = 1
    F2 = 2
    PI2 = 2*np.pi
    EPS = 1.0e-6

    woo, nx = x.shape

    # Other vector gubbins 
    
    disl1_3 = np.concatenate([disl1, disl1, disl1], axis=0)
    disl2_3 = np.concatenate([disl2, disl2, disl2], axis=0)
    
    u = np.zeros((3,nx))
    dub = np.zeros((3,nx))
    
    c0_alp3 = (F1 - alpha) / alpha
    
    pl8 = PI2 / 360
    c0_sd = np.sin(dip * pl8)
    c0_cd = np.cos(dip*pl8)
    
    if (abs(c0_cd) < EPS):
        c0_cd = F0
        if (c0_sd > F0):
            c0_sd = F1
        else:
            c0_sd = -F1
            
    c0_cdcd = c0_cd*c0_cd
    c0_sdcd = c0_sd*c0_cd
    
    p=y*c0_cd
    q=y*c0_sd
    
    jxi = (((x-al1) * (x-al2)) <= F0)
    jet = (((p-aw1) * (p-aw2)) <= F0)
    
    for k in range(1,3):
        if (k == 1 ):
            et = p-aw1
        else:
            et = p-aw2 
        
        for j in range(1,3):
            if (j == 1):
                xi = x-al1
            else:
                xi = x-al2

            # dccon2 "subroutine"
            # calculates station geometry constants for finite source
        
            etq_max= max(abs(et.all()), abs(q.all()))  
            dc_max = max(abs(xi.all()), etq_max.all())

            xi= xi - (abs(xi / dc_max) < EPS) * xi
            xi= xi - (abs(xi) < EPS) * xi
            et = et - (abs(et / dc_max) < EPS) * et
            et = et - (abs(et) < EPS) * et
            q = q - (abs(q / dc_max) < EPS) * q
            q = q - (abs(q) < EPS) * q

            dc_xi = xi
            dc_et = et
            dc_q = q

            c2_r = np.sqrt(dc_xi**2 + dc_et**2 + dc_q**2)

            if (np.sum(c2_r == F0) > 0):
                print('singularity error')
                ux=0
                uy=0
                uz=0
                error = 1

                return ux, uy, uz, error

            c2_y = dc_et * c0_cd + dc_q * c0_sd
            c2_d = dc_et * c0_sd - dc_q * c0_cd

            c2_tt = np.atan(dc_xi * dc_et /(dc_q * c2_r))
            c2_tt = c2_tt - c2_tt * (dc_q == F0)

            c2_x11 = (np.ones((1,nx))*F1) / (c2_r * (c2_r + dc_xi))
            c2_x11 = c2_x11 - ((dc_xi < F0) * (dc_q == 0) * (dc_et == 0)) * c2_x11

            c2_ale_msk1 = ((dc_et < F0) * (dc_q == F0) * (dc_xi ==F0))
            c2_ale_msk2 = 1 - c2_ale_msk1

            ret1 = c2_r - dc_et
            ret2 = c2_r + dc_et
            
            c2_ale = (c2_ale_msk1 * -np.log(ret1)) + (c2_ale_msk2 * np.log(ret2))
            c2_y11 = (np.ones((ret2.shape) * F1) / (c2_r * ret2))

            if (np.sum( (q == F0) * (((jxi == 1) * (et == F0)) + ((jet == 1) * (xi == F0))) + (c2_r == F0) ) > 0):
                ux = 0 
                uy = 0
                uz = 0
                error = 2
                # singular problems: return error code
                return ux, uy, uz, error

            # ub "subroutine"
            # part B of displacement and strain at depth due to buried fauls in semi-infinite medium

            rd = c2_r + c2_d

            if (np.sum(rd < 1e-14) > 0): 
                print('ub')
                print(rd ,c2_r, c2_d,xi,et,q)
            
            if (c0_cd != F0):
                xx = np.sqrt(xi**2 + q**2)
                ai4 = (xi != 0) * ((F1 / c0_cdcd) * (xi /rd * c0_sdcd + F2 * np.atan((et * (xx + q * c0_cd) + xx *(c2_r + xx) * c0_sd) / (xi * (c2_r + xx) * c0_cd))))
                ai3 = (c2_y * c0_cd / rd - c2_ale + c0_sd * np.log(rd)) / c0_cdcd
                
            else:
                rd2 = rd * rd
                ai3 = (et / rd + c2_y * q / rd2 - c2_ale) / F2
                ai4 = (xi * c2_y / rd2) / F2

            ai1 = -xi /rd * c0_cd - ai4 * c0_sd
            ai2 = np.log(rd) + ai3 * c0_sd
            qx = q * c2_x11
            qy = q * c2_y11

            # strike-slip contribution
            if (disl1.all() !=0):
                du2 = np.zeros((3, nx))
                du2[0, :] = - xi * qy - c2_tt - c0_alp3 * ai1 *c0_sd
                du2[1 ,:] = - q / c2_r + c0_alp3 * c2_y / rd * c0_sd
                du2[2, :] =  q * qy - c0_alp3 * ai2 * c0_sd

                dub = (disl1_3 * du2) / PI2

            else:
                dub = np.zeros((3,nx))

            # dip-slip contribution
            if (disl2.all() !=F0):
                du2 = np.zeros((3, nx))
                du2[0, :] = -q / c2_r + c0_alp3 * ai3 * c0_sdcd
                du2[1, :] = -et * qx - c2_tt - c0_alp3 * xi / rd * c0_sdcd
                du2[2, :] = q * qx + c0_alp3 * ai4 * c0_sdcd
            
                dub = dub + (disl2_3 / PI2) * du2

            du = np.zeros((3, nx))
            du[0, :] = dub[0, :]
            du[1, :] = dub[1, :] * c0_cd - dub[2, :] * c0_sd
            du[2, :] = dub[1, :] * c0_sd + dub[2, :] * c0_cd

            if ((j+k) != 3):
                u = u + du
            else:
                u = u - du

    ux = u[0, :] 
    uy = u[1, :] 
    uz = u[2, :] 
    err = 0

    return ux, uy, uz, err

    
        


















