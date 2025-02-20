import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter
from scipy.interpolate import LinearNDInterpolator, griddata
from skimage.morphology import convex_hull_image
import alphashape
from descartes import PolygonPatch
import shapely as shp
from shapely import Polygon, Point

from ConcaveHull import ConcaveHull
import svt_solver

U = np.random.randn(20, 5)
V = np.random.randn(15, 5)
R = np.random.randn(20, 15) + np.dot(U, V.T)

mask = np.round(np.random.rand(20, 15))
print(mask)
print(R.dtype, mask.dtype)
R_hat = svt_solver.svt_solve(R, mask)

def tr_area(x1,y1,x2,y2,x3,y3):
    # calculate area of a triangle defined by the coordinate vertices x1/y1,x2/y2,x3/y3
    return abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2.0)

def inside_tr(x1,y1,x2,y2,x3,y3,x,y):
    # calculate whether point x/y (P) lies within the triangle defined by the coordinate vertices x1/y1,x2/y2,x3/y3 (ABC)
    
    # area of ABC
    A = tr_area(x1,y1,x2,y2,x3,y3)

    # area of PBC
    A1 = tr_area(x, y, x2, y2, x3, y3)

    # area of PAC
    A2 = tr_area(x1, y1, x, y, x3, y3)

    # area of PAB
    A3 = tr_area(x1, y1, x2, y2, x, y)

    # check if  A1 + A2 + A3 == A
    if (A == A1 + A2+ A3):
        return True
    else:
        return False

def bary_inside_tr(A,B,C,P):
    # compute vectors
    v0 = C - A
    v1 = B - A
    v2 = P - A

    # compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # compute barycentric coordinates
    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # check if point is in triangle
    if (u >= 0) and (v >= 0) and (u + v < 1):
        return True
    else:
        return False

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
    xx_valid = xx[mask]
    yy_valid = yy[mask]
    xx_invalid = xx[~mask]
    yy_invalid = yy[~mask]
    all_points = np.column_stack((xx_invalid, yy_invalid))
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

    ch = ConcaveHull()
    ch.loadpoints(pp)
    ch.calculatehull(tol=0.1)
    print(ch.boundary)
    fig, ax = plt.subplots()
    x, y = ch.boundary.exterior.xy
    plt.plot(x,y)
    plt.savefig('shapelytest.png')
    
    thresh = 15.0
    small_edges = set()
    sim_ss = []
    for tr in d.simplices:
        single_tri_set = []
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]
            #if (edge_idx1, edge_idx0) in small_edges:
                #continue  # already visited this edge from other side
            p0 = pp[edge_idx0]
            p1 = pp[edge_idx1]
            if np.linalg.norm(p1 - p0) >  thresh:
                #small_edges.add((edge_idx0, edge_idx1))
                break
            elif i == 2:
                sim_ss.append(tr)
            else:
                pass

    print('SUBSET')
    sim_ss = np.array(sim_ss)
    print(sim_ss.shape)
    print(d.simplices.shape)

    idx_list = []
    count = 0
    for simplex in sim_ss:
        count += 1
        print(f'count {count} of {sim_ss.shape[0]}')
        #bary = d.find_simplex(all_points)
        idx_list.append(np.where(np.all(d.simplices == simplex, axis=1))[0])
        #in_triang = (bary == np.where(np.all(d.simplices[sim_ss]==simplex, axis=1))[0][0])
        #print(in_triang)
    filt = np.array(idx_list)

    a = d.simplices[filt]



    new_set = []
    tri_poly_set = []
    count = 0
    #for xy in all_points:
        #count += 1
        #print('checking point {count} of {all_points.shape[0]}')
        #print(xy)
        #x, y = xy[0], xy[1]
    for triang in pp[a]:
        x1, y1 = triang[0, 0, :]
        x2, y2 = triang[0, 1, :]
        x3, y3 = triang[0, 2, :]
        tri_poly = Polygon([(x1,y1), (x2,y2), (x3,y3)])
        tri_poly_set.append(tri_poly)
        #if inside_tr(x1,y1,x2,y2,x3,y3,x,y):
        #if bary_inside_tr(triang[0, 0, :], triang[0, 1, :], triang[0, 2, :], xy):
            #print('True')
            #new_set.append(xy)
        #else:
            #continue
    joined = shp.ops.unary_union(tri_poly_set)
    print('JOINED')
    print(joined)

    fig, ax = plt.subplots()
    try:
        for poly in joined.geoms:
            x,y = poly.exterior.xy
            ax.plot(x,y)
            for inte in poly.interiors:
                x,y = inte.xy
                ax.plot(x,y)
    except AttributeError:
        x,y = joined.exterior.xy
        ax.plot(x,y)
        for inte in joined.interiors:
            x,y = inte.xy
            ax.plot(x,y)


    plt.savefig('pleeeeeeeeeeeeeeasework.png')
    
    pppp = []
    for c in np.column_stack((xx.flatten(), yy.flatten())):
        print(c)
        point = Point(c[0], c[1])
        print(point)
        print(joined.contains(point))
        if joined.contains(point):
            pppp.append(c)
    
    fig, ax = plt.subplots()
    interp = griddata(pp, z, np.array(pppp), method='cubic')
    print(interp.shape)
    g = np.array(pppp)
    print(g.shape)
    ax.scatter(g[:,0], g[:,1], c=interp, s=0.05, cmap='jet')
    fig.savefig('pleaswork.png')

    final_img = np.zeros((224,224))
    for idx, u in enumerate(g):
        row = 224 - 1 - u[1]
        col = u[0]
        val = interp[idx]
        final_img[row, col] = val

    plt.imsave('final_img.png', final_img, cmap='jet')

    
    
    

    #plt.plot(pp[:, 0], pp[:, 1], '.', markersize=0.2)
    for i, j in small_edges:
        plt.plot(pp[[i, j], 0], pp[[i, j], 1], 'c', linewidth=0.1)
        
    

    plt.savefig('deltest.png')



    points = [(x, y) for x,y in zip(xx_valid, yy_valid)]
    #print(points)
    alpha = 0.2
    alpha_shape = alphashape.alphashape(points, alpha)
    #print(alpha_shape)

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


test_synthetic_completion('../test_outputs/combined/unwrapped/set1/unwrapped_DT_incidence_29_heading_30_vol-1e4_depth_0.2_turb_6.5_0.004_183.png', tol=1e-2, K=200)

