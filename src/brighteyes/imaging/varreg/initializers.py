import numpy as np
import torch

from scipy.ndimage import gaussian_filter

from scipy.signal import convolve

from .utils import centered_pad, partial_convolution_rfft, forward_model

def flat(shape, flux, device):

    init = torch.ones(shape).to(device) * flux / np.prod(shape[:-1])

    return init


def data(dset, flux, device):

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32)
    
    conv_dims = 2
    axis_to_conv = tuple(range(conv_dims))
    
    if dset.ndim - conv_dims == 2: #if I have channels
        dset = dset.sum(dim=-2)  # Sum over channels Nx x Ny x K
    
    init = torch.tensor(
        gaussian_filter(dset.numpy(), sigma=1, mode='constant', cval=0, truncate=4, axes=axis_to_conv)).to(device)
    init = init * flux / init.sum()

    return init


def rl_deconvolution(dset: torch.Tensor, psf: torch.Tensor, max_iter: int = 5, epsilon: float = None):
    """
    Richardson-Lucy deconvolution using FFT implementation.

    Parameters
    ----------
    dset : torch.Tensor (Hx x Wy x Nch x K)
        Input dataset (stack of images along the last axis).
    psf : torch.Tensor (Hx x Wy)
        PSF of the system.
    max_iter : int
        Number of iterations.
    epsilon : float
        Minimum value of the denominator (default = torch.finfo(dset.dtype).eps).

    Returns
    -------
    img_deconv : torch.Tensor (Hx x Wy x K)
        Deconvolved images.
    """
    device = dset.device
    dtype = torch.float32

    conv_dims = 2

    # Initialization
    if dset.ndim - conv_dims == 1: #if I do not have channels
        dset = dset.unsqueeze(-2)  # Add a channel dimension #Nx x Ny x 1 x K
        psf = psf.unsqueeze(-1)  # Add a channel dimension #Mx x My x 1

    # Sum dataset over repetition axis
    image = dset.sum(dim=-1)    # Nx x Ny x 1
    
    axis_to_conv = tuple(range(conv_dims))

    # Pad and normalize PSF
    psf = centered_pad(psf, target_shape=image.shape)
    h = psf / psf.sum()

    # Fourier transforms of PSF and its flipped version
    otf = torch.fft.rfftn(h, dim=axis_to_conv)
    otf_conj = torch.conj(otf)

    # Initialize object estimate
    obj = torch.ones_like(image.sum(-1), dtype=dtype, device=device) # Nx x Ny

    if epsilon is None:
        epsilon = torch.finfo(dtype).eps

    for _ in range(max_iter):
        # Convolution: obj * h

        conv = forward(otf, obj) # Nx x Ny x 1
        
        # Avoid division by zero
        ratio = image / conv + epsilon #torch.clamp(conv, min=epsilon)

        # Back-projection: ratio * hT
        correction = forward(otf_conj, ratio).sum(-1) # Nx x Ny
        # Update estimate
        obj = obj * correction

    # Expand back into dataset shape (repeat along last axis)
    n_rep = dset.shape[-1]
    size = tuple([1]*(conv_dims) + [n_rep])

    return obj.unsqueeze(-1).repeat(size) / n_rep


def initialize(dset, psf, method: str = 'flat', device: str = None, max_iter: int = 10):
    """
    Initialize the reconstruction based on the specified method.

    Parameters:
        dset (ndarray): Input dataset of shape Nz x Nx x Ny x Nch x K.
        psf (ndarray): Point Spread Function of shape Px x Py.
        method (str): Initialization method ('flat', 'data', 'rl').
        device (str): Device to use ('cpu' or 'cuda'). Default is None, which uses 'cuda' if available.

    Returns:
        Tensor: Initialized object tensor of shape Nz x Nx x Ny x K.
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
    conv_dims = 2
    axis_to_sum = tuple(range(dset.ndim - 1))
    flux = dset.sum(axis_to_sum, keepdim=True)

    if method == 'flat':
        # if dset.ndim - conv_dims == 2: #if I have channels
        #     dset = dset.sum(dim=-2)  # Sum over channels, output Nx x Ny x K
        dset = dset.sum(dim=-2)  # Sum over channels, output Nx x Ny x K
        init_shape = (psf.shape[0],) + tuple(dset.shape) #dset.squeeze().shape
        init = flat(init_shape, flux, device)
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