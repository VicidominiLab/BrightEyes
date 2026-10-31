# Array-dimension conventions used throughout this module:
# Nx, Ny : lateral image dimensions
# Mx, My : unpadded PSF dimensions
# Nz     : axial object/PSF planes
# Nch    : detector or measurement channels
# K      : independent noise realizations

import numpy as np
import torch

from scipy.ndimage import gaussian_filter

from .utils import as_float_tensor, centered_pad, standardize_dataset, standardize_psf


def flat(shape, flux, device):
    """Create a spatially uniform initialization with realization-wise flux.

    Parameters
    ----------
    shape : tuple[int, int, int, int]
        Requested object shape ``(Nz, Nx, Ny, K)``.
    flux : torch.Tensor or array-like
        Photon flux for each realization. The expected broadcastable shape is
        ``(1, 1, 1, K)``.
    device : str or torch.device
        Device on which to allocate the initialization.

    Returns
    -------
    torch.Tensor
        Uniform object with shape ``(Nz, Nx, Ny, K)``. Summing over ``Nz``,
        ``Nx``, and ``Ny`` recovers the supplied flux for every realization.
    """
    shape = tuple(shape)
    flux = as_float_tensor(flux, device, dtype=torch.float32)
    return torch.ones(shape, dtype=flux.dtype, device=device) * flux / np.prod(shape[:-1])


def data(dset, flux, device):
    """Create a Gaussian-smoothed, data-driven initialization.

    Detector channels are summed before applying a Gaussian filter over the
    two lateral axes. Each realization is then independently rescaled to its
    measured photon flux.

    Parameters
    ----------
    dset : torch.Tensor or array-like
        Dataset shaped ``(Nx, Ny, K)`` or ``(Nx, Ny, Nch, K)``.
    flux : torch.Tensor or array-like
        Per-realization photon flux, usually shaped ``(1, 1, 1, K)``.
    device : str or torch.device
        Output device. Smoothing is performed by SciPy on the CPU and the
        result is moved back to this device.

    Returns
    -------
    torch.Tensor
        Initialization shaped ``(1, Nx, Ny, K)``. The singleton leading axis
        is the axial object dimension ``Nz``.
    """
    dset = standardize_dataset(dset, device, dtype=torch.float32).sum(dim=-2)  # (Nx, Ny, K)
    smoothed = gaussian_filter(dset.detach().cpu().numpy(), sigma=1, mode='constant', cval=0, truncate=4, axes=(0, 1))
    init = torch.as_tensor(smoothed, dtype=dset.dtype, device=device)

    flux = as_float_tensor(flux, device, dtype=torch.float32).reshape(-1)
    if flux.numel() != init.shape[-1]:
        raise ValueError(f"Flux and dataset realization counts differ: {flux.numel()} != {init.shape[-1]}.")

    spatial_sum = init.sum(dim=(0, 1), keepdim=True)
    scale = torch.where(spatial_sum > 0, flux.view(1, 1, -1) / spatial_sum, torch.zeros_like(spatial_sum))
    return (init * scale).unsqueeze(0)


def _rl_forward(otf: torch.Tensor, obj: torch.Tensor, output_shape: tuple) -> torch.Tensor:
    """Project ``(Nz, Nx, Ny)`` object planes to ``(Nx, Ny, Nch)`` data space."""
    obj_fft = torch.fft.rfftn(obj, dim=(1, 2))
    data_fft = torch.einsum('zxyc,zxy->xyc', otf, obj_fft)
    return torch.fft.irfftn(data_fft, s=output_shape, dim=(0, 1))


def _rl_backproject(otf_conj: torch.Tensor, ratio: torch.Tensor, output_shape: tuple) -> torch.Tensor:
    """Back-project a ``(Nx, Ny, Nch)`` ratio to ``(Nz, Nx, Ny)`` object space."""
    ratio_fft = torch.fft.rfftn(ratio, dim=(0, 1))
    object_fft = torch.einsum('zxyc,xyc->zxy', otf_conj, ratio_fft)
    return torch.fft.irfftn(object_fft, s=output_shape, dim=(1, 2))


def rl_deconvolution(dset: torch.Tensor, psf: torch.Tensor, max_iter: int = 5, epsilon: float = None):
    """Create an initialization with Richardson-Lucy deconvolution.

    The independent realizations are first summed to form the full-flux
    measurement. RL is then applied with the same multi-plane and
    multi-channel image-formation model used by the reconstruction code. The
    resulting full-flux object is copied equally across the ``K``
    realization-specific estimates.

    Parameters
    ----------
    dset : torch.Tensor
        Dataset shaped ``(Nx, Ny, K)`` or ``(Nx, Ny, Nch, K)``.
    psf : torch.Tensor
        PSF shaped ``(Mx, My)``, ``(Mx, My, Nch)``, or ``(Nz, Mx, My, Nch)``.
    max_iter : int, default=5
        Number of RL iterations.
    epsilon : float or None, default=None
        Positive denominator floor. The machine epsilon of the working dtype
        is used when omitted.

    Returns
    -------
    torch.Tensor
        RL-based initialization shaped ``(Nz, Nx, Ny, K)``.
    """
    if not isinstance(max_iter, int) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer.")

    device = dset.device if isinstance(dset, torch.Tensor) else torch.device('cpu')
    dset = standardize_dataset(dset, device, dtype=torch.float32)
    psf = standardize_psf(as_float_tensor(psf, device, dtype=torch.float32))

    image = dset.sum(dim=-1)  # (Nx, Ny, Nch): full-flux measurement
    spatial_shape = tuple(image.shape[:2])
    psf = centered_pad(psf, target_shape=spatial_shape)
    psf_sum = psf.sum(dim=(1, 2, 3), keepdim=True)
    if torch.any(psf_sum <= 0):
        raise ValueError("Each PSF plane must have a strictly positive sum.")

    # Move the centered PSF origin to index zero before forming its OTF.
    psf = torch.fft.ifftshift(psf / psf_sum, dim=(1, 2))
    otf = torch.fft.rfftn(psf, dim=(1, 2))
    otf_conj = torch.conj(otf)
    obj = torch.ones((psf.shape[0], *spatial_shape), dtype=dset.dtype, device=device)

    if epsilon is None:
        epsilon = torch.finfo(dset.dtype).eps
    if epsilon <= 0:
        raise ValueError("epsilon must be strictly positive.")

    target_flux = image.sum()
    if target_flux <= 0:
        obj.zero_()
    else:
        obj = obj * target_flux / obj.numel()
        for _ in range(max_iter):
            prediction = _rl_forward(otf, obj, spatial_shape)
            ratio = image / prediction.clamp_min(epsilon)
            correction = _rl_backproject(otf_conj, ratio, spatial_shape)
            obj = (obj * correction).clamp_min(0)
            obj = obj * target_flux / obj.sum().clamp_min(epsilon)

    n_rep = dset.shape[-1]
    return obj.unsqueeze(-1).repeat(1, 1, 1, n_rep) / n_rep


def initialize(dset, psf, method: str = 'flat', device: str = None, max_iter: int = 10):
    """Initialize the realization-specific reconstruction variables.

    Parameters
    ----------
    dset : torch.Tensor or np.ndarray
        Dataset shaped ``(Nx, Ny, K)`` or ``(Nx, Ny, Nch, K)``.
    psf : torch.Tensor or np.ndarray
        PSF shaped ``(Mx, My)``, ``(Mx, My, Nch)``, or ``(Nz, Mx, My, Nch)``.
    method : {'flat', 'data', 'rl'}, default='flat'
        Initialization strategy. ``flat`` distributes each realization flux
        uniformly, ``data`` smooths the corresponding measurement, and ``rl``
        deconvolves the summed full-flux measurement before sharing it equally
        across realizations.
    device : str, torch.device, or None, default=None
        Processing device. CUDA is selected when available if omitted.
    max_iter : int, default=10
        Number of RL iterations used only when ``method='rl'``.

    Returns
    -------
    torch.Tensor
        Initial object estimates shaped ``(Nz, Nx, Ny, K)``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
    dset = standardize_dataset(dset, device, dtype=torch.float32)
    psf = standardize_psf(as_float_tensor(psf, device, dtype=torch.float32))

    flux = dset.sum(dim=(0, 1, 2), keepdim=True)  # (1, 1, 1, K)
    method = method.lower()

    if method == 'flat':
        init_shape = (psf.shape[0], dset.shape[0], dset.shape[1], dset.shape[-1])
        init = flat(init_shape, flux, device)
    elif method == 'data':
        init = data(dset, flux, device)
        if psf.shape[0] > 1:
            init = init.repeat(psf.shape[0], 1, 1, 1) / psf.shape[0]
    elif method == 'rl':
        init = rl_deconvolution(dset, psf, max_iter=max_iter)
    else:
        raise ValueError("Initialization method must be 'flat', 'data', or 'rl'.")

    return init