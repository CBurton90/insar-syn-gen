import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import generic_filter
from scipy.interpolate import LinearNDInterpolator, griddata
import shapely as shp
from shapely import Polygon, Point

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

def test_synthetic_completion(file, tol=10.0, K=None):

    #img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = np.load(file)

    mask = (img == 0)
    print(mask[106:118, 112:124])


    img[np.where(img == 0)] = np.nan
    img = generic_filter(img, np.nanmedian, footprint=np.ones((3,3)))

    sparse = np.ma.MaskedArray(img, mask=mask)

    
    coords_x, coords_y = np.meshgrid(np.arange(0,224,1), np.arange(0,224,1))
    coords_y = np.flipud(coords_y) # 0 on Y axis must begin lower left
    x_valid = coords_x[~mask]
    y_valid = coords_y[~mask]
    coord_set = np.column_stack((x_valid, y_valid))

    d = scipy.spatial.Delaunay(coord_set)

    valid_tris = []
    for tr in d.simplices:
        for i in range(3):
            edge_idx0 = tr[i]
            edge_idx1 = tr[(i+1)%3]
            p0 = coord_set[edge_idx0]
            p1 = coord_set[edge_idx1]
            if np.linalg.norm(p1-p0) > tol:
                break
            elif i == 2:
                valid_tris.append(tr)
            else:
                pass

    valid_tris = np.array(valid_tris)

    #small_edges = set()
    #sim_ss = []
    #for tr in d.simplices:
        #single_tri_set = []
        #for i in range(3):
            #edge_idx0 = tr[i]
            #edge_idx1 = tr[(i+1)%3]
            #if (edge_idx1, edge_idx0) in small_edges:
                #continue  # already visited this edge from other side
            #p0 = pp[edge_idx0]
            #p1 = pp[edge_idx1]
            #if np.linalg.norm(p1 - p0) >  thresh:
                #small_edges.add((edge_idx0, edge_idx1))
                #break
            #elif i == 2:
                #sim_ss.append(tr)
            #else:
                #pass
    
    plt.triplot(coord_set[:,0], coord_set[:,1], valid_tris)
    plt.savefig("example_tri_mesh.png")
    plt.clf()

    poly_set = []
    valid_tri_pts = coord_set[valid_tris]
    for tri_pts in valid_tri_pts:
        x1,y1 = tri_pts[0, :]
        x2,y2 = tri_pts[1, :]
        x3,y3 = tri_pts[2, :]
        poly = Polygon([(x1,y1),(x2,y2),(x3,y3)])
        poly_set.append(poly)

    poly_union = shp.unary_union(poly_set)

    fig, ax = plt.subplots()
    try:
        for poly in poly_union.geoms:
            x,y = poly.exterior.xy
            ax.plot(x,y)
            try:
                for itr in poly.interiors:
                    x,y = itr.xy
                    ax.plot(x,y)
            except AttributeError:
                pass
    except AttributeError:
        x,y = poly_union.exterior.xy
        ax.plot(x,y)
        try:
            for itr in poly_union.interiors:
                x,y = itr.xy
                ax.plot(x,y)
        except:
            pass
    fig.savefig('example_polygons.png')
    plt.clf()

    full_grid = np.column_stack((coords_x.flatten(), coords_y.flatten()))

    intrpt_xy = []
    for xy in full_grid:
        point = Point(xy[0], xy[1])
        if poly_union.contains(point):
            intrpt_xy.append(xy)

    intrpt_xy = np.array(intrpt_xy)

    intrpt_vals = griddata((coord_set[:,0], coord_set[:,1]), sparse.compressed(), intrpt_xy, method='linear')
    plt.scatter(intrpt_xy[:,0],intrpt_xy[:,1], c=intrpt_vals, s=4)
    plt.savefig('example_interpolated.png')

    intrpt_image = np.zeros((224,244))
    for idx, img_coord in enumerate(intrpt_xy):
        row = 224 - 1 - img_coord[1]
        col = img_coord[0]
        val = intrpt_vals[idx]
        intrpt_image[row, col] = val

    mask = (intrpt_image == 0)
    final = np.ma.MaskedArray(intrpt_image, mask=mask)
    plt.imsave('example_interpolated_image.png', final, cmap='jet', vmin=-20, vmax=20)

test_synthetic_completion('../test_outputs/combined/unwrapped/set1/unwrapped_DT_incidence_29_heading_30_vol-1e3_depth_0.065_turb_6.5_0.008_7.npy', tol=30.0, K=200)
