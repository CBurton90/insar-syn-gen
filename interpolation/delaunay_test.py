import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import generic_filter
from scipy.interpolate import LinearNDInterpolator, griddata
import shapely as shp
from shapely import Polygon, Point


def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]


# Compute filter kernel with radius correlation_scale (can probably be a bit smaller)
correlation_scale = 25
x = np.arange(-correlation_scale, correlation_scale)
y = np.arange(-correlation_scale, correlation_scale)
X, Y = np.meshgrid(x, y)
dist = np.sqrt(X*X + Y*Y)
filter_kernel = np.exp(-dist**2/(2*correlation_scale))

# Generate n-by-n grid of spatially correlated noise
n = 100
noise = np.random.randn(n, n)
noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
#plt.contourf(np.arange(n), np.arange(n), noise)
#plt.savefig("test_noise.png")

print(np.min(noise), np.max(noise))

mask = (noise >= -4) & (noise <= 4)
print(mask)

full_img = func(grid_x, grid_y).T[:100, :]
print(full_img.shape)
plt.imshow(full_img)
plt.savefig("test_img.png")
plt.clf()

sparse = np.ma.MaskedArray(full_img, mask=mask)
plt.imshow(sparse)
plt.savefig("test_sparse.png")
plt.clf()

coords_x, coords_y = np.meshgrid(np.arange(0,100,1), np.arange(0,100,1))
coords_y = np.flipud(coords_y)
x_valid = coords_x[~mask]
y_valid = coords_y[~mask]
coord_set = np.column_stack((x_valid, y_valid))

d = scipy.spatial.Delaunay(coord_set)

thresh = 10.0

valid_tris = []
for tr in d.simplices:
    for i in range(3):
        edge_idx0 = tr[i]
        edge_idx1 = tr[(i+1)%3]
        p0 = coord_set[edge_idx0]
        p1 = coord_set[edge_idx1]
        if np.linalg.norm(p1-p0) > thresh:
            break
        elif i == 2:
            valid_tris.append(tr)
        else:
            pass

valid_tris = np.array(valid_tris)
print(d.simplices.shape)
print(valid_tris.shape)

plt.triplot(coord_set[:,0],coord_set[:,1], valid_tris)
plt.savefig("tri_mesh.png")
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
fig.savefig('test_polygons.png')
plt.clf()

full_grid = np.column_stack((coords_x.flatten(), coords_y.flatten()))

intrpt_xy = []
for xy in full_grid:
    point = Point(xy[0], xy[1])
    if poly_union.contains(point):
        intrpt_xy.append(xy)

intrpt_xy = np.array(intrpt_xy)
print(intrpt_xy.shape)

intrpt_vals = griddata((coord_set[:,0],coord_set[:,1]), sparse.compressed(), intrpt_xy, method='linear')
plt.scatter(intrpt_xy[:,0],intrpt_xy[:,1], c=intrpt_vals, s=4)
plt.savefig('test_interpolated.png')

intrpt_image = np.zeros((100,100))
for idx, img_coord in enumerate(intrpt_xy):
    row = 100 - 1 - img_coord[1]
    col = img_coord[0]
    val = intrpt_vals[idx]
    intrpt_image[row, col] = val

mask = (intrpt_image == 0)
final = np.ma.MaskedArray(intrpt_image, mask=mask)

plt.imsave('test_interpolated_image.png', final)

