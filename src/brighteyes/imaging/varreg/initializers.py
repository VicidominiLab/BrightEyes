import numpy as np
import torch

from scipy.ndimage import gaussian_filter

from scipy.signal import convolve


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


def rl_deconvolution(dset, psf, max_iter=5, epsilon=None):
    """
    Richardson-Lucy deconvolution, performed using FFT

    Parameters
    ----------
    h : np.array (Nx x Ny)
        PSF of the system
    i : np.array(Mx x My)
        Image
    max_iter : float
        Number of iterations
    epsilon : float
        Minimum value of the denominator.
        Used to avoid division by zero (default = float.eps)
    reg : float
        Regularization parameter (Tikhonov)

    Returns
    -------
    img_deconv : np.array(Mx x My)
        Deconvolved image

    """

    n_rep = dset.shape[-1]
    i = dset.numpy().sum(-1)
    h = psf.numpy()

    if epsilon is None:
        epsilon = np.finfo(float).eps

    h = h / np.sum(h)  # PSF normalization
    hT = np.flip(h)
    obj = np.ones(i.shape)  # Initialization

    k = 0
    while k < max_iter:

        conv = convolve(obj, h, mode='same')
        A = np.where(conv < epsilon, 0, i / conv)
        B = convolve(A, hT, mode='same')
        C = obj
        obj = B * C

        k += 1

    return torch.tensor(obj).unsqueeze(-1).repeat(1, 1, n_rep) / n_rep


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