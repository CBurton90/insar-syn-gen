# refactored into Python by C Burton 2025 from https://github.com/pui-nantheera/Synthetic_InSAR_image/blob/main/turbulent/pcmc_atm.m
# TJW 2004 & Nantheera Anantrasirichai 2021

# useful - https://www.sfu.ca/sasdoc/sashtml/stat/chap58/sect13.htm

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

# for a given spatial grid use a distance-covariance function to produce a covariance matrix and decompose with Cholesky decomposition to produce spatially correlated noise
def create_turb_del(rows, cols, maxvariance, alpha, covmodel, rv_N, psizex, psizey):

    n = rows * cols

    row_vals = np.linspace(1, rows, rows)
    col_vals = np.linspace(1, cols, cols)

    yy, xx = np.meshgrid(row_vals, col_vals)

    xxv = np.reshape(xx, (n, 1))
    yyv = np.reshape(yy, (n, 1))

    xxv = xxv * psizex
    yyv = yyv * psizey

    dx = np.tile(xxv, (1,n)) - np.tile(xxv.T, (n,1))
    dy = np.tile(yyv, (1,n)) - np.tile(yyv.T, (n,1))

    rgrid = np.sqrt(dx**2 + dy**2)

    del xx, yy, xxv, yyv, dx, dy

    if covmodel == 0:
        # use exp func to calculate variance-covariance matrix
        vcm = maxvariance * np.exp(-alpha * rgrid)
    elif covmodel == 1:
        # use expcos func to calculate variance-covariance matrix
        vcm = maxvariance * np.exp(-alpha * rgrid) * np.cos(beta * rgrid)
    elif covmodel == 2:
        # use ebessel model to calculate variance-covariance matrix
        # TO DO define ebessel
        print('Bessel covariance model not yet defind')
    else:
        print('Covariance model to compute cov matrix not defined')

    print(vcm)

    # calculate correlated turbulent atmosphere noise using Cholesky Decomposition
    # first create a matrix of N (rv_N) gaussian/standard normal distributed noise vectors of length n
    Z = np.random.randn(rv_N, n)
    # perform Chol Decomp on vcm
    V = cholesky(vcm, lower=False) # return upper triangular part of Cholesky V.T*V = vcm
    # create matrix X containing rv_N correlated noise vectors of length n
    X = Z @ V
    X = X.T #transpose to make it the right way up

    # write out
    t_noise_stack = np.empty((rows, cols, rv_N))
    for i in range(rv_N):
        t_noise_grid = np.reshape(X[:,i], (cols, rows))
        t_noise_grid_tp = t_noise_grid.T
        #plt.imsave("TPS_"+str(i)+'.png', t_noise_grid_tp, cmap='jet')
        t_noise_stack[..., i] = t_noise_grid_tp

    return t_noise_stack

if __name__ == '__main__':
    test = create_turb_del(100, 100, 5.5, 0.008, 0, 10, 1, 1)
    print(test.shape)
