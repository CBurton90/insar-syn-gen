import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.interpolate import LinearNDInterpolator
from skimage.morphology import convex_hull_image
import alphashape
from descartes import PolygonPatch

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
    y = np.arange(224, 0, -1)
    xx, yy = np.meshgrid(x,y)
    print(xx)
    print(yy)
    xx_valid = xx[mask]
    yy_valid = yy[mask]
    z = img[mask]
    pp = np.vstack((xx_valid,yy_valid)).T #rows/cols
    interp = LinearNDInterpolator(pp, z)
    Z = interp(xx, yy)
    print(Z.shape)

    import scipy
    d = scipy.spatial.Delaunay(pp)
    r = abs(d.plane_distance(np.array([116, 140])))
    print(np.min(r), np.max(r))

    #np.random.seed(0)
    #x = 3.0 * np.random.rand(1000)
    #y = 2.0 * np.random.rand(1000) - 1.0
    #inside = ((x ** 2 + y ** 2 > 1.0) & ((x - 3) ** 2 + y ** 2 > 1.0) & ((x-1.5) ** 2 + y ** 2 > 0.09))
    #points = np.vstack([x[inside], y[inside]]).T
    #tri = scipy.spatial.Delaunay(points)
    
    thresh = 10.0
    small_edges = set()
    sim_ss = []
    for tr in d.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]
            if (edge_idx1, edge_idx0) in small_edges:
                continue  # already visited this edge from other side
            p0 = pp[edge_idx0]
            p1 = pp[edge_idx1]
            if np.linalg.norm(p1 - p0) <  thresh:
                small_edges.add((edge_idx0, edge_idx1))

    #plt.plot(pp[:, 0], pp[:, 1], '.', markersize=0.2)
    for i, j in small_edges:
        plt.plot(pp[[i, j], 0], pp[[i, j], 1], 'c', linewidth=0.1)
        
    

    plt.savefig('deltest.png')



    points = [(x, y) for x,y in zip(xx_valid, yy_valid)]
    print(points)
    alpha = 0.2
    alpha_shape = alphashape.alphashape(points, alpha)
    print(alpha_shape)

    fig, ax = plt.subplots()
    # Plot input points
    ax.scatter(*zip(*points))
    # Plot alpha shape
    ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
    plt.savefig('testalpha.png')

    #img = img.astype(float)
    #mask = mask.astype(float)
    #print(img.dtype, mask.dtype)
    #X = svt_solver.svt_solve(img, mask, epsilon=tol, max_iterations=K)
    #plt.imsave('test.png', X, cmap='jet')


test_synthetic_completion('../test_outputs/combined/wrapped/set1/wrapped_DT_incidence_29_heading_20_vol-1e3.5_depth_0.09_turb_8.25_0.004_145.png', tol=1e-2, K=200)

