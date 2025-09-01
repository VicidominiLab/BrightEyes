import torch
import numpy as np

from scipy.ndimage import gaussian_filter

from .utils import partial_convolution_rfft


def disk(radius, shape):
    nx, ny = shape
    x = np.arange(-nx // 2, nx // 2, 1)
    y = np.arange(-nx // 2, nx // 2, 1)

    xx, yy = np.meshgrid(x, y)

    r = np.sqrt(xx ** 2 + yy ** 2)

    return np.where(r <= radius, 1, 0)


def sparse_random_image(shape, num_nonzero, value_range=(0.1, 1.0), margin=0, dtype=np.float32):
    """
    Generate an array of zeros with `num_nonzero` random pixels set to
    values uniformly drawn from `value_range`. Random pixels are restricted
    to lie at least `margin` pixels away from each border.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the array.
    num_nonzero : int
        Number of nonzero pixels.
    value_range : (float, float)
        Range [low, high] for random values.
    margin : int
        Excluded border width (applied to all dimensions).
    dtype : np.dtype
        Output dtype.

    Returns
    -------
    img : np.ndarray
        Generated sparse image.
    """
    shape = tuple(shape)
    ndim = len(shape)

    # Valid ranges for each dimension
    valid_ranges = [range(margin, s - margin) for s in shape]
    if any(len(r) <= 0 for r in valid_ranges):
        raise ValueError("Margin too large for given shape.")

    # Total number of valid positions
    valid_coords = np.array(np.meshgrid(*valid_ranges, indexing="ij")).reshape(ndim, -1).T
    if num_nonzero > len(valid_coords):
        raise ValueError(f"num_nonzero={num_nonzero} exceeds available {len(valid_coords)} valid positions.")

    # Choose random coordinates
    chosen_idx = np.random.choice(len(valid_coords), size=num_nonzero, replace=False)
    chosen_coords = valid_coords[chosen_idx]

    # Fill image
    img = np.zeros(shape, dtype=dtype)
    values = np.random.uniform(value_range[0], value_range[1], num_nonzero).astype(dtype)
    img[tuple(chosen_coords.T)] = values

    return img


def generate_disk_phantom(shape, num, radius, value_range=(0.1, 1.0), smooth: bool = False):

    img = sparse_random_image(shape=shape, num_nonzero=num, value_range=value_range, margin = radius)
    kernel = disk(radius=radius, shape=shape)

    phantom = partial_convolution_rfft(kernel, img, 'xy', 'xy', 'xy', fourier=(0,0))

    if smooth is True:
        phantom = gaussian_filter(phantom, sigma=1)

    phantom[phantom < 0] = 0

    return phantom
