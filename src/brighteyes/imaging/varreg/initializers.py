import numpy as np
import torch

from scipy.ndimage import gaussian_filter

from scipy.signal import convolve

from .utils import centered_pad, partial_convolution_rfft

def flat(shape, flux, device):

    init = torch.ones(shape).to(device) * flux / np.prod(shape[:-1])

    return init


def data(dset, flux, device):

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32)

    axis_to_conv = tuple(range(2))
    init = torch.tensor(
        gaussian_filter(dset.numpy(), sigma=1, mode='constant', cval=0, truncate=4, axes=axis_to_conv)).to(device)
    init = init * flux / init.sum()

    return init


def rl_deconvolution(dset: torch.Tensor, psf: torch.Tensor, max_iter: int = 5, epsilon: float = None):
    """
    Richardson-Lucy deconvolution using FFT implementation.

    Parameters
    ----------
    dset : torch.Tensor (Hx x Wy x N)
        Input dataset (stack of images along the last axis).
    psf : torch.Tensor (Hx x Wy)
        PSF of the system.
    max_iter : int
        Number of iterations.
    epsilon : float
        Minimum value of the denominator (default = torch.finfo(dset.dtype).eps).

    Returns
    -------
    img_deconv : torch.Tensor (Hx x Wy x N)
        Deconvolved images.
    """
    device = dset.device
    dtype = torch.float32

    # Collapse stack into a single image (sum over last axis)
    image = dset.sum(dim=-1)

    # Pad and normalize PSF
    psf = centered_pad(psf, target_shape=image.shape)
    h = psf / psf.sum()

    # Fourier transforms of PSF and its flipped version
    otf = torch.fft.rfftn(h)
    otf_conj = torch.conj(otf)

    # Initialize object estimate
    obj = torch.ones_like(image, dtype=dtype, device=device)

    if epsilon is None:
        epsilon = torch.finfo(dtype).eps

    for _ in range(max_iter):
        # Convolution: obj * h

        conv = partial_convolution_rfft(otf, obj, dim1='ab', dim2='ab', axis='ab', fourier=(1, 0))

        # Avoid division by zero
        ratio = image / torch.clamp(conv, min=epsilon)

        # Back-projection: ratio * hT
        correction = partial_convolution_rfft(otf_conj, ratio, dim1='ab', dim2='ab', axis='ab', fourier=(1, 0))
        # Update estimate
        obj = obj * correction

    # Expand back into dataset shape (repeat along last axis)
    n_rep = dset.shape[-1]

    return obj.unsqueeze(-1).repeat(1, 1, n_rep) / n_rep


def initialize(dset, psf, method: str = 'flat', device: str = None, max_iter: int = 10):
    """
    Initialize the reconstruction based on the specified method.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        psf (ndarray): Point Spread Function of shape Px x Py.
        method (str): Initialization method ('flat', 'data', 'rl').
        device (str): Device to use ('cpu' or 'cuda'). Default is None, which uses 'cuda' if available.

    Returns:
        Tensor: Initialized object tensor of shape Nx x Ny x K.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32).to(device)
    else:
        dset = dset.to(device)

    # Axis to sum for flux conservation
    axis_to_sum = tuple(range(dset.ndim - 1))
    flux = dset.sum(axis_to_sum, keepdim=True)

    if method == 'flat':
        init = flat(dset.shape, flux, device)
    elif method == 'data':
        init = data(dset, flux, device)
    elif method == 'rl':
        if not isinstance(psf, torch.Tensor):
            psf = torch.tensor(psf, dtype=torch.float32).to(device)
        else:
            psf = psf.to(device)
        init = rl_deconvolution(dset, psf, max_iter=max_iter)
    else:
        raise ValueError("Initialization method must be 'flat', 'data', or 'rl'.")

    return init