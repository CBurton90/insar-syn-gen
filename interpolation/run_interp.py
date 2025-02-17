import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.interpolate import LinearNDInterpolator
import svt_solver

U = np.random.randn(20, 5)
V = np.random.randn(15, 5)
R = np.random.randn(20, 15) + np.dot(U, V.T)

mask = np.round(np.random.rand(20, 15))
print(mask)
print(R.dtype, mask.dtype)
R_hat = svt_solver.svt_solve(R, mask)

def test_synthetic_completion(file, tol=None, K=None):

    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    mask = (img != 0)

    #img = img.astype(float)
    #img[img == 0] = np.nan
    print(img[106:118, 122:162])
    img = generic_filter(img, np.nanmedian, footprint=np.ones((3,3)))
    print(img[106:118, 122:162])
    
    x = np.arange(0, 224, 1)
    y = np.arange(0, 224, 1)
    xx, yy = np.meshgrid(x,y)
    xx_valid = xx[mask]
    yy_valid = yy[mask]
    z = img[mask]
    pp = np.vstack((xx_valid,yy_valid)).T
    interp = LinearNDInterpolator(pp, z)
    Z = interp(xx, yy)
    print(Z.shape)

    import scipy
    d = scipy.spatial.Delaunay(pp)
    r = abs(d.plane_distance(np.array([116, 140])))
    print(np.min(r), np.max(r))

    plt.imsave('test.png', Z, cmap='jet')

    #img = img.astype(float)
    #mask = mask.astype(float)
    #print(img.dtype, mask.dtype)
    #X = svt_solver.svt_solve(img, mask, epsilon=tol, max_iterations=K)
    #plt.imsave('test.png', X, cmap='jet')


test_synthetic_completion('../test_outputs/combined/wrapped/set1/wrapped_DT_incidence_31_heading_5_vol-1e3.5_depth_0.09_turb_5.5_0.008_1.png', tol=1e-2, K=200)

