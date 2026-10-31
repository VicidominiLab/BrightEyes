# Origin: VicidominiLab/Regularization-by-Noise, src/varreg/loss.py at bb6d0c6.
"""Data-fidelity and realization-consistency losses used by ``varreg``.

Array convention
----------------
``Nx, Ny`` are the lateral image dimensions, ``Nz`` is the number of object
planes, ``Nch`` the detector-channel axis, ``K`` the independent noise
realizations, and ``Niter`` an optional leading reconstruction-history axis.
"""

import torch
from torch.fft import rfftn

from .regularizers import compute_centroid
from .utils import (
    forward_model, kl_divergence, standardize_dataset, standardize_fidelity_weight,
    standardize_object_history, standardize_psf,
)


def fidelity_loss(dataset, obj_set, psf, eps=None, weight_map: torch.Tensor = 1, full_KL=False, k_loss=True):
    """Compute Poisson data fidelity for one object or a reconstruction history.

    Parameters
    ----------
    dataset : torch.Tensor or array-like
        Measured photon counts shaped ``(Nx, Ny, Nch, K)``. Singleton channel
        or realization axes may be omitted.
    obj_set : torch.Tensor or array-like
        One reconstruction ``(Nz, Nx, Ny, K)`` or a history
        ``(Niter, Nz, Nx, Ny, K)``. Reduced single-channel/single-realization
        layouts are accepted and standardized internally.
    psf : torch.Tensor or array-like
        Real PSF ``(Nz, Mx, My, Nch)`` or precomputed complex OTF
        ``(Nz, Nx, Ny_rfft, Nch)``. Reduced 2D and 3D layouts are accepted.
    eps : float or None, default=None
        Positive offset used in the KL division and logarithm.
    weight_map : scalar or array-like, default=1
        Optional point-wise fidelity weight broadcastable over ``Nx, Ny, Nch, K``.
    full_KL : bool, default=False
        Include the linear term ``prediction - data``. With per-realization
        flux conservation, its spatial sum is zero and the logarithmic term is
        equivalent up to numerical precision.
    k_loss : bool, default=True
        Preserve the realization axis ``K``. This is required by the package's
        realization-wise Armijo line search; set to ``False`` to sum over ``K``.

    Returns
    -------
    torch.Tensor
        Per-realization losses shaped ``(Niter, K)`` when ``k_loss=True`` or
        total losses shaped ``(Niter,)`` otherwise. Singleton axes are squeezed.
    """
    device = obj_set.device if isinstance(obj_set, torch.Tensor) else None
    obj_probe = torch.as_tensor(obj_set, device=device)
    dtype = obj_probe.dtype if obj_probe.is_floating_point() else torch.float32

    psf_fft = standardize_psf(psf, device=device, dtype=dtype, target_shape=torch.as_tensor(dataset).shape[:2])
    obj_history = standardize_object_history(obj_probe, psf_fft.shape[0], device)
    prediction = torch.stack([forward_model(psf_fft, obj).sum(dim=0) for obj in obj_history], dim=0)
    dataset = standardize_dataset(dataset, device=prediction.device, dtype=prediction.dtype)

    if eps is None:
        eps = torch.finfo(prediction.dtype).eps
    weight = standardize_fidelity_weight(weight_map, prediction)
    if weight.ndim == 0 and weight.item() == 1:
        return kl_divergence(dataset, prediction, normalize_entries=False, eps=eps, full_KL=full_KL, k_loss=k_loss)

    data = dataset.unsqueeze(0)
    fidelity = data * torch.log((data + eps) / (prediction + eps))
    if full_KL:
        fidelity = fidelity + prediction - data
    fidelity = fidelity * weight
    axes = (1, 2, 3) if k_loss else (1, 2, 3, 4)
    return torch.squeeze(fidelity.sum(dim=axes))


def regularization_loss(obj_set: torch.Tensor, regularizer: str = 'l2', weight_map: torch.Tensor = 1, otf=None,
                        iteration_axis: bool = False, eps: float = None, fourier: bool = False):
    """Compute a realization-consistency penalty while preserving the ``K`` axis.

    Parameters
    ----------
    obj_set : torch.Tensor
        Reconstruction ensemble ``(..., Nx, Ny, K)``. When ``iteration_axis``
        is true, the leading axis is interpreted as ``Niter``.
    regularizer : {'l2', 'l1', 'kl'}, default='l2'
        Point-wise distance from the centroid across ``K``. ``kl`` denotes the
        generalized KL divergence and is available only in the spatial domain.
    weight_map : scalar or array-like, default=1
        Spatial regularization weight broadcastable with ``obj_set``.
    otf : torch.Tensor or None, default=None
        Optional Fourier-domain weight. Its magnitude must be broadcastable
        with the Fourier-domain residuals.
    iteration_axis : bool, default=False
        Preserve the first axis as ``Niter`` when summing object dimensions.
    eps : float or None, default=None
        Positive numerical floor used by the KL branch and MTF normalization.
    fourier : bool, default=False
        Evaluate the L2 residual after an RFFT over the two lateral axes.

    Returns
    -------
    torch.Tensor
        Penalty per realization, shaped ``(K,)`` or ``(Niter, K)``.
    """
    obj_set = torch.as_tensor(obj_set)
    regularizer = regularizer.lower()
    if eps is None:
        eps = 1e-8
    if eps <= 0:
        raise ValueError("eps must be positive.")
    if regularizer not in {'l1', 'l2', 'kl'}:
        raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")

    if fourier:
        if regularizer != 'l2':
            raise ValueError("Fourier-domain regularization currently supports only the L2 norm.")
        obj_set = rfftn(obj_set, norm='forward', dim=(-3, -2))

    axis_to_sum = tuple(range(1, obj_set.ndim-1)) if iteration_axis else tuple(range(obj_set.ndim-1))
    centroid = compute_centroid(obj_set, regularizer, eps=eps)

    if regularizer == 'l2':
        residuals = torch.abs(obj_set - centroid)**2
    elif regularizer == 'l1':
        residuals = torch.abs(obj_set - centroid)
    else:
        safe_obj = obj_set.clamp_min(eps)
        residuals = safe_obj - centroid + centroid * torch.log(centroid / safe_obj)

    weight = torch.as_tensor(weight_map, dtype=residuals.real.dtype, device=residuals.device)
    residuals = residuals * weight

    if fourier and otf is not None:
        mtf = torch.abs(torch.as_tensor(otf, device=residuals.device))
        mtf = mtf / mtf.max().clamp_min(eps)
        residuals = residuals / mtf.clamp_min(1e-2)

    return residuals.sum(dim=axis_to_sum)