import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import generic_filter
from scipy.interpolate import LinearNDInterpolator, griddata
import shapely as shp
from shapely import Polygon, Point


def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2

grid_x, grid_y = np.meshgrid(np.arange(0,224,1), np.arange(0,224,1))

# Compute filter kernel with radius correlation_scale (can probably be a bit smaller)
correlation_scale = 25
x = np.arange(-correlation_scale, correlation_scale)
y = np.arange(-correlation_scale, correlation_scale)
X, Y = np.meshgrid(x, y)
dist = np.sqrt(X*X + Y*Y)
filter_kernel = np.exp(-dist**2/(2*correlation_scale))

# Generate n-by-n grid of spatially correlated noise
n = 224
noise = np.random.randn(n, n)
noise = scipy.signal.fftconvolve(noise, filter_kernel, mode='same')
plt.contourf(np.arange(n), np.arange(n), noise)
plt.savefig("test_noise.png")

print(noise.shape)
