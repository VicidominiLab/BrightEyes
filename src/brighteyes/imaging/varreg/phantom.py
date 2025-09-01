import torch
import numpy as np

from scipy.ndimage import gaussian_filter

from .utils import partial_convolution_rfft,


def disk(radius, shape):
    nx, ny = shape
    x = np.arange(-nx // 2, nx // 2, 1)
    y = np.arange(-nx // 2, nx // 2, 1)

    xx, yy = np.meshgrid(x, y)

    r = np.sqrt(xx ** 2 + yy ** 2)

    return np.where(r <= radius, 1, 0)


def sparse_random_image(shape, num_nonzero, value_range=(0.1, 1.0), dtype=np.float32):
    """
    Generate an array of zeros with `num_nonzero` random pixels set to
    values uniformly drawn from `value_range`.
    """

    total = np.prod(shape)
    if num_nonzero > total:
        raise ValueError(f"num_nonzero={num_nonzero} exceeds total pixels {total}.")

    flat_idx = np.random.choice(total, size=num_nonzero, replace=False)
    values = np.random.uniform(low=value_range[0], high=value_range[1], size=num_nonzero).astype(dtype)

    img = np.zeros(shape, dtype=dtype)
    np.put(img, flat_idx, values)
    return img


def generate_disk_phantom(shape, num, radius, value_range=(0.1, 1.0), smooth: bool = False):
    img = sparse_random_image(shape=shape, num_nonzero=num, value_range=value_range)
    kernel = disk(radius=radius, shape=shape)

    phantom = partial_convolution_rfft(kernel, img, 'xy', 'xy', 'xy', fourier=(0,0))

    if smooth is True:
        phantom = gaussian_filter(phantom, sigma=1)

    return phantom
