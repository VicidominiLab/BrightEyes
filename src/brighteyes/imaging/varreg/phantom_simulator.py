"""Utilities for generating simple synthetic 2D fluorescence phantoms.

Array convention
----------------
Ny, Nx : number of image rows and columns, respectively

All public generators return NumPy arrays with shape ``(Ny, Nx)``.
"""

import numpy as np

from scipy.ndimage import gaussian_filter

from .utils import partial_convolution_rfft


def _get_rng(seed):
    """Return the global NumPy RNG or a local legacy RNG initialized with ``seed``."""
    return np.random if seed is None else np.random.RandomState(seed)


def _normalize_unit_interval(image):
    """Normalize an array in place to [0, 1], leaving a constant zero array unchanged."""
    image -= image.min()
    max_value = image.max()
    if max_value > 0:
        image /= max_value
    return image


def disk(radius, shape):
    """Create a centered binary disk.

    Parameters
    ----------
    radius : float
        Disk radius in pixels.
    shape : tuple[int, int]
        Output shape ``(Ny, Nx)``.

    Returns
    -------
    np.ndarray
        Binary image with shape ``(Ny, Nx)`` and value 1 inside the disk.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if len(shape) != 2 or any(size <= 0 for size in shape):
        raise ValueError("shape must contain two positive dimensions (Ny, Nx).")

    ny, nx = shape
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(xx**2 + yy**2)

    return np.where(r <= radius, 1, 0)


def sparse_random_image(shape, num_nonzero, value_range=(0.1, 1.0), margin=0, dtype=np.float32):
    """Generate a sparse image with randomly positioned nonzero pixels.

    Parameters
    ----------
    shape : tuple[int, ...]
        Output array shape. The function supports any number of dimensions.
    num_nonzero : int
        Number of distinct nonzero pixels.
    value_range : tuple[float, float], default=(0.1, 1.0)
        Inclusive sampling range ``(low, high)`` for the nonzero values.
    margin : int, default=0
        Number of pixels excluded from both borders along every axis.
    dtype : np.dtype, default=np.float32
        Output dtype.

    Returns
    -------
    np.ndarray
        Sparse array with shape ``shape``.
    """
    shape = tuple(shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("shape must contain positive dimensions.")
    if not isinstance(num_nonzero, (int, np.integer)):
        raise TypeError("num_nonzero must be an integer.")
    if num_nonzero < 0:
        raise ValueError("num_nonzero must be non-negative.")
    if not isinstance(margin, (int, np.integer)):
        raise TypeError("margin must be an integer.")
    if margin < 0:
        raise ValueError("margin must be non-negative.")
    if value_range[0] > value_range[1]:
        raise ValueError("value_range must satisfy low <= high.")

    ndim = len(shape)
    valid_ranges = [range(margin, size - margin) for size in shape]
    if any(len(valid_range) <= 0 for valid_range in valid_ranges):
        raise ValueError("margin is too large for the requested shape.")

    # Materialize all admissible coordinates, then sample distinct locations.
    valid_coords = np.array(np.meshgrid(*valid_ranges, indexing="ij")).reshape(ndim, -1).T
    if num_nonzero > len(valid_coords):
        raise ValueError(f"num_nonzero={num_nonzero} exceeds the {len(valid_coords)} available positions.")

    chosen_idx = np.random.choice(len(valid_coords), size=num_nonzero, replace=False)
    chosen_coords = valid_coords[chosen_idx]

    img = np.zeros(shape, dtype=dtype)
    values = np.random.uniform(value_range[0], value_range[1], num_nonzero).astype(dtype)
    img[tuple(chosen_coords.T)] = values

    return img


def generate_disk_phantom(shape, num, radius, value_range=(0.1, 1.0), smooth: bool = False):
    """Generate a 2D phantom containing randomly positioned filled disks.

    ``num`` weighted point sources are placed at least ``radius`` pixels from
    the image border and circularly convolved with a binary disk kernel.

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape ``(Ny, Nx)``.
    num : int
        Number of disk centers.
    radius : int
        Disk radius in pixels. Integer-valued floats are also accepted.
    value_range : tuple[float, float], default=(0.1, 1.0)
        Intensity range assigned to the disk centers before convolution.
    smooth : bool, default=False
        Apply a Gaussian filter with ``sigma=1`` to the convolved phantom.

    Returns
    -------
    np.ndarray
        Float phantom with shape ``(Ny, Nx)``. The output type is independent
        of the value of ``smooth``.
    """
    if not float(radius).is_integer():
        raise ValueError("radius must be an integer for disk-center margin handling.")
    margin = int(radius)

    img = sparse_random_image(shape=shape, num_nonzero=num, value_range=value_range, margin=margin)
    kernel = disk(radius=radius, shape=shape)
    phantom = partial_convolution_rfft(kernel, img, dim1="xy", dim2="xy", axis="xy", fourier=(False, False))
    phantom = phantom.detach().cpu().numpy()

    if smooth is True:
        phantom = gaussian_filter(phantom, sigma=1)

    phantom[phantom < 0] = 0
    return phantom


def Fit_gaussian2D(x, y, sx, sy, ux, uy, K):
    """Evaluate an axis-aligned normalized 2D Gaussian.

    Parameters
    ----------
    x, y : float or np.ndarray
        Coordinates at which the Gaussian is evaluated.
    sx, sy : float
        Standard deviations along x and y.
    ux, uy : float
        Gaussian center coordinates.
    K : float
        Integrated Gaussian amplitude.

    Returns
    -------
    float or np.ndarray
        Gaussian values with the broadcast shape of ``x`` and ``y``.
    """
    result = K / (2.0 * np.pi * sx * sy) * np.exp(-((x - ux)**2 / (2.0 * sx**2)) - ((y - uy)**2 / (2.0 * sy**2)))
    return result


def Fit_gaussian2DDraw(w, h, sx, sy, ux, uy, theta, K, pixelSize):
    """Draw a rotated 2D Gaussian on a regular pixel grid.

    Parameters
    ----------
    w, h : int
        Output width ``Nx`` and height ``Ny`` in pixels.
    sx, sy : float
        Gaussian standard deviations in physical-coordinate units.
    ux, uy : float
        Gaussian center in physical-coordinate units.
    theta : float
        Rotation angle in radians.
    K : float
        Integrated Gaussian amplitude.
    pixelSize : float
        Physical size of one pixel.

    Returns
    -------
    np.ndarray
        Gaussian image with shape ``(h, w) == (Ny, Nx)``.
    """
    x = np.arange(w, dtype=float)[None, :] * pixelSize - ux
    y = np.arange(h, dtype=float)[:, None] * pixelSize - uy
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    x_rot = x * cos_theta - y * sin_theta
    y_rot = x * sin_theta + y * cos_theta
    return Fit_gaussian2D(x_rot, y_rot, sx, sy, 0, 0, K)


def Nucleus(shape, noOfGauss, value_range, std_range, ellipse_axes, background_noise=0.01, seed=None):
    """Generate a nucleus-like phantom containing Gaussian intensity domains.

    Parameters
    ----------
    shape : tuple[int, int]
        Output shape ``(Ny, Nx)``.
    noOfGauss : int
        Number of Gaussian domains placed inside the elliptical nucleus.
    value_range : tuple[float, float]
        Range of Gaussian peak amplitudes.
    std_range : tuple[float, float]
        Range of isotropic Gaussian standard deviations in pixels.
    ellipse_axes : tuple[float, float]
        Ellipse radii ``(x_radius, y_radius)`` in pixels.
    background_noise : float, default=0.01
        Amplitude of additive uniform background noise.
    seed : int or None, default=None
        Seed used by a local NumPy random-number generator. ``None`` uses the
        current global NumPy RNG state.

    Returns
    -------
    np.ndarray
        Float phantom with shape ``(Ny, Nx)`` normalized to [0, 1].
    """
    if noOfGauss < 0:
        raise ValueError("noOfGauss must be non-negative.")
    if any(axis <= 0 for axis in ellipse_axes):
        raise ValueError("ellipse_axes must contain positive radii.")
    if std_range[0] <= 0 or std_range[0] > std_range[1]:
        raise ValueError("std_range must contain positive values with min <= max.")

    rng = _get_rng(seed)
    h, w = shape
    y, x = np.indices(shape)
    cy, cx = h / 2, w / 2

    # Binary support of the nucleus; ellipse_axes are ordered as (x_radius, y_radius).
    ellipse_mask = ((x - cx) / ellipse_axes[0])**2 + ((y - cy) / ellipse_axes[1])**2 <= 1
    if not ellipse_mask.any():
        raise ValueError("ellipse_axes do not define any valid pixel inside the image.")

    phantom = np.zeros(shape, dtype=float)

    # Draw Gaussian centers uniformly in the image and retain only points inside the ellipse.
    positions = []
    while len(positions) < noOfGauss:
        px = rng.uniform(0, w)
        py = rng.uniform(0, h)
        if ellipse_mask[int(py % h), int(px % w)]:
            positions.append((px, py))

    for px, py in positions:
        amp = rng.uniform(*value_range)
        std = rng.uniform(*std_range)
        phantom += amp * np.exp(-((x - px)**2 + (y - py)**2) / (2 * std**2))

    phantom *= ellipse_mask
    phantom = gaussian_filter(phantom, sigma=1.0)
    phantom += background_noise * rng.random_sample(shape)

    return _normalize_unit_interval(phantom)


def generate_membrane_phantom(shape, radius, thickness, value_range, noise_level=0.05, blur_sigma=1.0, seed=None):
    """Generate a single fluorescence-like elliptical membrane.

    Parameters
    ----------
    shape : tuple[int, int]
        Output shape ``(Ny, Nx)``.
    radius : tuple[float, float]
        Ellipse radii ``(x_radius, y_radius)`` in pixels.
    thickness : float
        Dimensionless shell-width parameter, internally scaled by 1/100.
    value_range : tuple[float, float]
        Range of multiplicative membrane intensities.
    noise_level : float, default=0.05
        Amplitude of additive uniform background noise.
    blur_sigma : float, default=1.0
        Gaussian blur standard deviation in pixels. Set to 0 to disable blur.
    seed : int or None, default=None
        Seed used by a local NumPy random-number generator.

    Returns
    -------
    np.ndarray
        Float phantom with shape ``(Ny, Nx)`` normalized to [0, 1].
    """
    if any(axis <= 0 for axis in radius):
        raise ValueError("radius must contain positive ellipse radii.")
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    if blur_sigma < 0 or noise_level < 0:
        raise ValueError("blur_sigma and noise_level must be non-negative.")

    rng = _get_rng(seed)
    h, w = shape
    y, x = np.indices(shape)
    cy, cx = h / 2, w / 2
    a, b = radius

    # Gaussian shell around the implicit ellipse boundary ellipse_eq == 1.
    ellipse_eq = ((x - cx) / a)**2 + ((y - cy) / b)**2
    membrane_mask = np.exp(-((ellipse_eq - 1.0)**2) / (2 * (thickness / 100)**2))
    phantom = membrane_mask * rng.uniform(*value_range, size=shape)

    if blur_sigma > 0:
        phantom = gaussian_filter(phantom, sigma=blur_sigma)
    if noise_level > 0:
        phantom += noise_level * rng.random_sample(shape)

    return _normalize_unit_interval(phantom)


def generate_hollow_membrane_phantom(
    shape=(256, 256),
    n_ellipses=40,
    radius_range=(6, 25),
    amplitude_range=(0.5, 1.0),
    thickness=0.15,
    blur_sigma=1.0,
    noise_level=0.05,
    seed=None
):
    """Generate multiple randomly oriented hollow elliptical membranes.

    Parameters
    ----------
    shape : tuple[int, int], default=(256, 256)
        Output shape ``(Ny, Nx)``.
    n_ellipses : int, default=40
        Number of membrane objects.
    radius_range : tuple[float, float], default=(6, 25)
        Sampling range used independently for the x and y radii.
    amplitude_range : tuple[float, float], default=(0.5, 1.0)
        Sampling range for each membrane amplitude.
    thickness : float, default=0.15
        Width of the Gaussian shell in elliptical-distance coordinates.
    blur_sigma : float, default=1.0
        Final Gaussian blur standard deviation in pixels.
    noise_level : float, default=0.05
        Amplitude of additive uniform background noise.
    seed : int or None, default=None
        Seed used by a local NumPy random-number generator.

    Returns
    -------
    np.ndarray
        Float phantom with shape ``(Ny, Nx)`` normalized to [0, 1].
    """
    if n_ellipses < 0:
        raise ValueError("n_ellipses must be non-negative.")
    if radius_range[0] <= 0 or radius_range[0] > radius_range[1]:
        raise ValueError("radius_range must contain positive values with min <= max.")
    if amplitude_range[0] > amplitude_range[1]:
        raise ValueError("amplitude_range must satisfy min <= max.")
    if thickness <= 0:
        raise ValueError("thickness must be positive.")
    if blur_sigma < 0 or noise_level < 0:
        raise ValueError("blur_sigma and noise_level must be non-negative.")

    rng = _get_rng(seed)
    h, w = shape
    y, x = np.indices(shape)
    phantom = np.zeros(shape, dtype=float)

    for _ in range(n_ellipses):
        cx = rng.uniform(0.2 * w, 0.8 * w)
        cy = rng.uniform(0.2 * h, 0.8 * h)
        rx = rng.uniform(*radius_range)
        ry = rng.uniform(*radius_range)
        amp = rng.uniform(*amplitude_range)
        theta = rng.uniform(0, np.pi)

        # Rotate coordinates into the principal-axis frame of the ellipse.
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X = x - cx
        Y = y - cy
        Xr = cos_t * X + sin_t * Y
        Yr = -sin_t * X + cos_t * Y
        ellipse_eq = (Xr / rx)**2 + (Yr / ry)**2

        ring = np.exp(-((ellipse_eq - 1.0)**2) / (2 * thickness**2))
        ring *= amp * rng.uniform(0.8, 1.2, size=shape)
        phantom += ring

    phantom = gaussian_filter(phantom, sigma=blur_sigma)
    phantom += noise_level * rng.random_sample(shape)

    return _normalize_unit_interval(phantom)