# Origin: VicidominiLab/Regularization-by-Noise, src/varreg/optimizer.py at bb6d0c6.
"""Iterative optimizers for variance-regularized reconstruction.

All modalities are converted once to the most general internal layout:

``dset``
    ``(Nx, Ny, Nch, K)``
``psf``
    ``(Nz, Mx, My, Nch)``
``obj``
    ``(Nz, Nx, Ny, K)``

Confocal and MID inputs are handled by inserting the missing singleton axes.
The optimization code can therefore use one common implementation.
"""

import math
from numbers import Real

import numpy as np
import torch
from torch.fft import rfftn
from tqdm import tqdm

from . import initializers as init
from .loss import fidelity_loss, regularization_loss
from .regularizers import grad_kl, grad_l1, grad_l2
from .utils import (
    centered_pad,
    forward_model,
    standardize_dataset,
    standardize_object_weight,
    standardize_psf,
    tensor_scalar_product,
)


def _regularizer_terms(obj: torch.Tensor, regularizer: str):
    """Return the positive/negative terms of the selected regularizer gradient.

    Parameters
    ----------
    obj : torch.Tensor
        Reconstruction ensemble with shape ``(Nz, Nx, Ny, K)``.
    regularizer : {'l2', 'l1', 'kl'}
        Regularizer used to couple the ``K`` reconstructions.

    Returns
    -------
    u, v : torch.Tensor
        Terms such that the regularizer gradient can be written as ``v - u``.
    """
    if regularizer == 'l2':
        return grad_l2(obj)
    if regularizer == 'l1':
        return grad_l1(obj)
    if regularizer == 'kl':
        return grad_kl(obj)
    raise ValueError("regularizer must be 'l2', 'l1', or 'kl'.")


def _normalize_flux(obj: torch.Tensor, dset: torch.Tensor) -> torch.Tensor:
    """Rescale each reconstruction to the photon flux of its realization.

    Parameters
    ----------
    obj : torch.Tensor
        Reconstructed objects ``(Nz, Nx, Ny, K)``.
    dset : torch.Tensor
        Measurements ``(Nx, Ny, Nch, K)``.

    Returns
    -------
    torch.Tensor
        Flux-normalized objects with the same shape as ``obj``.
    """
    flux = dset.sum(dim=(0, 1, 2), keepdim=True)       # (1, 1, 1, K)
    obj_flux = obj.sum(dim=(0, 1, 2), keepdim=True)   # (1, 1, 1, K)
    return obj * flux / obj_flux


def compute_step_size_line_search(img, obj, obj_tmp, psf_fft, grad, lam, D_matrix_inv,
                                  normalize=True, theta=0.8, delta=0.5, max_iter=30,
                                  initial_step_size=None, verbose=False, regularizer='l2',
                                  weight=1, eps=1e-8):
    """Compute the step size by backtracking independently over the ``K`` realizations.

    Parameters
    ----------
    img : torch.Tensor
        Measured data ``(Nx, Ny, Nch, K)``.
    obj : torch.Tensor
        Current objects ``(Nz, Nx, Ny, K)``.
    obj_tmp : torch.Tensor
        Projected candidate point before applying the line-search step.
    psf_fft : torch.Tensor
        OTF ``(Nz, Nx, Ny_rfft, Nch)``.
    grad : torch.Tensor
        Gradient at the current iterate, shape ``(Nz, Nx, Ny, K)``.
    lam : float
        Regularization parameter.
    D_matrix_inv : scalar or torch.Tensor
        Inverse SGP metric. For SGP it has the same shape as ``obj``.
    normalize : bool, default=True
        Apply the same flux normalization used by the reconstruction update.
    theta : float, default=0.8
        Backtracking reduction factor.
    delta : float, default=0.5
        Sufficient-decrease parameter.
    max_iter : int, default=30
        Maximum number of backtracking iterations.
    initial_step_size : torch.Tensor or None
        Initial step for every realization, normally shape ``(K,)``.
    regularizer : {'l2', 'l1', 'kl'}, default='l2'
        Regularizer used when evaluating the trial loss.
    weight : scalar or torch.Tensor, default=1
        Spatial regularization weight.
    eps : float, default=1e-8
        Numerical floor.

    Returns
    -------
    step_size : torch.Tensor
        Accepted step sizes, shape ``(K,)``.
    should_update : torch.Tensor
        Boolean mask indicating which realizations found an accepted step.
    """
    K = obj.shape[-1]
    device = obj.device

    if initial_step_size is None:
        initial_step_size = torch.ones(K, dtype=obj.dtype, device=device) * theta

    step_size = torch.clamp(initial_step_size / theta, max=1)
    # Track ks that ever satisfy the descent lemma
    satisfied_once = torch.zeros(K, dtype=torch.bool, device=device)
    # Track which k have already fixed their step size
    frozen = torch.zeros(K, dtype=torch.bool, device=device)

    # Loss at the current iterate, evaluated independently for each realization.
    fidelity = fidelity_loss(img, obj, psf_fft, eps=eps, full_KL=False, k_loss=True)
    regularization = regularization_loss(
        obj, regularizer=regularizer, weight_map=weight, iteration_axis=False, eps=eps
    )
    loss_value = fidelity + lam * regularization

    direction = obj_tmp - obj
    axis_to_sum = (0, 1, 2)
    min_step_size = 1e-10

    for i in range(max_iter):
        # Freeze ks whose step is below min threshold
        below_thr = step_size < min_step_size
        step_size = torch.where(below_thr, torch.full_like(step_size, min_step_size), step_size)
        frozen |= below_thr
        if frozen.all():
            break

        # Realizations that already satisfied the condition are kept fixed.
        active_step = step_size.clone()
        active_step[frozen] = 0
        obj_new = obj + active_step * direction

        if normalize:
            obj_new = _normalize_flux(obj_new, img)

        #Compute new loss
        fidelity_new = fidelity_loss(img, obj_new, psf_fft, eps=eps, full_KL=False, k_loss=True)
        regularization_new = regularization_loss(
            obj_new, regularizer=regularizer, weight_map=weight, iteration_axis=False, eps=eps
        )
        loss_value_new = fidelity_new + lam * regularization_new

        # Descent condition used in the original implementation.
        diff = obj_new - obj
        lhs = loss_value_new - loss_value - tensor_scalar_product(diff, grad, axis_to_sum=axis_to_sum)
        rhs = (delta / step_size) * tensor_scalar_product(
            diff, D_matrix_inv**(-1) * diff, axis_to_sum=axis_to_sum
        )
        # Which ks satisfy descent lemma?
        satisfied = lhs <= rhs

        # Mark ks that satisfied at least once
        satisfied_once |= satisfied
        # Only freeze ks that are not frozen yet
        frozen |= (~frozen) & satisfied
        # For ks that are not frozen yet, shrink η_k
        step_size = torch.where(frozen, step_size, step_size * theta)

        if verbose:
            print(
                f"Line search iteration {i}: accepted={int(satisfied_once.sum())}/{K}, "
                f"mean step={step_size.mean().item():.3e}"
            )

    return torch.clamp(step_size, max=1), satisfied_once


def update_fft(img, obj, psf_fft, psf_m_fft, lam, weight, eps, m,
               method: str, regularizer: str, normalize: bool, step_size_search: str,
               initial_step_size=1.0, line_search_params=(10, 2), verbose=False):
    """Perform one reconstruction update in Fourier space.

    Parameters
    ----------
    img : torch.Tensor
        Measured data ``(Nx, Ny, Nch, K)``.
    obj : torch.Tensor
        Current reconstruction ensemble ``(Nz, Nx, Ny, K)``.
    psf_fft, psf_m_fft : torch.Tensor
        OTF and conjugate OTF ``(Nz, Nx, Ny_rfft, Nch)``.
    lam : float
        Regularization parameter.
    weight : scalar or torch.Tensor
        Spatial regularization weight broadcastable with ``obj``.
    eps : float
        Numerical floor.
    m : int
        Zero-based outer iteration index.
    method : {'osl', 'abs', 'sgm', 'pgd', 'sgp'}
        Optimization update.
    regularizer : {'l2', 'l1', 'kl'}
        Consistency regularizer. The RbN reconstruction uses ``l2``.
    normalize : bool
        Apply photon-flux normalization after the update.
    step_size_search : {'fixed', 'line_search', 'rl'} or positive scalar
        Step-size rule for PGD/SGP. In ``rl`` mode the full current object is
        used as step, as in the original implementation.
    initial_step_size : scalar or torch.Tensor
        Step used to initialize the current update.
    line_search_params : tuple
        SGP metric-truncation parameters ``(s1, s2)``.

    Returns
    -------
    obj_new : torch.Tensor
        Updated objects ``(Nz, Nx, Ny, K)``.
    step_size : torch.Tensor
        Step size used by the update. Usually ``(K,)``; in ``rl`` mode it has
        the complete object shape.
    """
    method = method.lower()
    regularizer = regularizer.lower()

    # Forward projection H u_k and RL-like back-projection H^T[v_k / (H u_k)].
    img_estimate_k = forward_model(psf_fft, obj).sum(0)        # (Nx, Ny, Nch, K)
    fraction = img / (img_estimate_k + eps)
    del img_estimate_k
    update_k = forward_model(psf_m_fft, fraction).sum(-2)      # (Nz, Nx, Ny, K)
    del fraction
    
    if method == 'osl':
        obj_mean = obj.mean(dim=-1, keepdim=True)
        denom = 1 + 2 * lam * weight * (obj - obj_mean)
        obj_update = update_k / (denom + eps)
        obj_update.clamp_(min=1e-8)
        obj_new = obj * obj_update
        step_size = torch.ones(obj.shape[-1], dtype=obj.dtype, device=obj.device)

    elif method == 'abs':
        obj_mean = obj.mean(dim=-1, keepdim=True)
        denom = 1 + 2 * lam * weight * torch.abs(obj - obj_mean)
        obj_new = obj * update_k / (denom + eps)
        step_size = torch.ones(obj.shape[-1], dtype=obj.dtype, device=obj.device)

    elif method == 'sgm':
        u, v = _regularizer_terms(obj, regularizer)
        obj_new = obj * (update_k + lam * weight * u) / (1 + lam * weight * v)
        step_size = torch.ones(obj.shape[-1], dtype=obj.dtype, device=obj.device)

    elif method in {'pgd', 'sgp'}:
        obj_mean = obj.mean(dim=-1, keepdim=True)
        grad = 1 - update_k + 2 * lam * weight * (obj - obj_mean)

        if method == 'pgd':
            D_matrix_inv = 1
        else:
            # SGP inverse metric and truncation used in the original code.
            if line_search_params is None:
                line_search_params = (1e18, 2)
            s1, s2 = line_search_params
            D_matrix_inv = obj / (1 + 2 * lam * weight * obj)
            thr = math.sqrt(1 + s1 / ((m + 1)**s2))
            D_matrix_inv = torch.clamp(D_matrix_inv, min=1 / thr, max=thr)

        # Projection onto the non-negative orthant.
        obj_tmp = torch.where(obj - D_matrix_inv * grad > 0, obj - D_matrix_inv * grad, 0)

        if isinstance(step_size_search, Real):
            step_size = torch.ones(obj.shape[-1], dtype=obj.dtype, device=obj.device) * float(step_size_search)
            search_mode = 'fixed'
        else:
            search_mode = step_size_search.lower()
            step_size = initial_step_size

        should_update = torch.ones(obj.shape[-1], dtype=torch.bool, device=obj.device)

        if search_mode == 'rl':
            # Keep the original RL-like choice: the complete current object is the step.
            step_size = obj
            obj_new = obj + step_size * (obj_tmp - obj)

        elif search_mode == 'fixed':
            obj_new = obj + step_size * (obj_tmp - obj)

        elif search_mode == 'line_search':
            step_size, should_update = compute_step_size_line_search(
                img, obj, obj_tmp, psf_fft, grad, lam, D_matrix_inv,
                normalize=normalize, initial_step_size=step_size, verbose=verbose,
                regularizer='l2', weight=weight, eps=eps,
            )
            obj_new = obj.clone()
            obj_new[..., should_update] = (
                obj[..., should_update]
                + step_size[should_update] * (obj_tmp[..., should_update] - obj[..., should_update])
            )

        else:
            raise ValueError("step_size_search must be 'rl', 'line_search', 'fixed', or a positive scalar.")

    else:
        raise ValueError("method must be 'osl', 'abs', 'sgm', 'pgd', or 'sgp'.")

    if normalize:
        obj_new = _normalize_flux(obj_new, img)

    return obj_new, step_size


def iterate(dset, psf, max_iter, lam, weight, stop_tol, method, device, regularizer_norm, initialization,
            normalize, stop_criteria, rec_to_save, step_size_search, line_search_params, idx_in_focus=0,
            return_pointwise_sol=False, verbose=False):
    """Run the outer reconstruction loop.

    Parameters
    ----------
    dset : torch.Tensor
        Canonical data ``(Nx, Ny, Nch, K)``.
    psf : torch.Tensor
        Canonical real PSF ``(Nz, Mx, My, Nch)``.
    max_iter : int
        Maximum number of reconstruction iterations.
    lam : float
        Regularization parameter.
    weight : scalar or torch.Tensor
        Spatial regularization weight broadcastable with the object.
    stop_tol : float or None
        Early-stopping tolerance. ``None`` disables early stopping.
    method : {'osl', 'abs', 'sgm', 'pgd', 'sgp'}
        Optimization method.
    device : torch.device
        Device used for the active reconstruction tensors.
    regularizer_norm : {'l2', 'l1', 'kl'}
        Consistency regularizer.
    initialization : {'flat', 'data', 'rl'}
        Object initialization method.
    normalize : bool
        Apply photon-flux normalization after each update.
    stop_criteria : {'rel_diff', 'step_size'}
        Early-stopping criterion.
    rec_to_save : {'last', 'all'}
        Store only the current/final reconstruction or the full iteration history.
    step_size_search : {'fixed', 'line_search', 'rl'} or positive scalar
        Step-size rule.
    line_search_params : tuple or None
        SGP metric-truncation parameters.
    idx_in_focus : int, default=0
        Axial plane used for the relative-change criterion.
    return_pointwise_sol : bool, default=False
        Sum over ``K`` before storing each reconstruction.

    Returns
    -------
    obj_all : torch.Tensor
        Saved reconstructions. Full histories are deliberately stored on CPU;
        active optimization tensors remain on ``device``.
    step_size_history : list[torch.Tensor]
        Step sizes for completed iterations, moved to CPU before being stored.
    """
    eps = 1e-8

    # Pad/normalize the PSF once and precompute the OTFs once.
    psf_pad = centered_pad(psf, target_shape=dset.shape[:2])
    psf_norm = psf_pad / psf_pad.sum(dim=(1, 2, 3), keepdim=True)
    psf_fft = rfftn(psf_norm, dim=(1, 2))
    psf_m_fft = torch.conj(psf_fft)

    # The active object stays on the computation device.
    obj_set = init.initialize(dset, psf_norm, method=initialization, device=device)

    if return_pointwise_sol:
        saved_shape = tuple(obj_set.shape[:-1])
    else:
        saved_shape = tuple(obj_set.shape)
    
    if rec_to_save == 'all':
        n_saved = max_iter + 1
        history_device = torch.device('cpu')
    else:
        n_saved =1
        history_device = device

    obj_all = torch.empty((n_saved,) + saved_shape, dtype=obj_set.dtype, device=history_device)
    if return_pointwise_sol:
        obj_all[0] = obj_set.sum(-1).detach().to(history_device)
    else:
        obj_all[0] = obj_set.detach().to(history_device)

    # Initialize the step size. A scalar input is interpreted as a fixed step.
    initial_step_size = torch.ones(obj_set.shape[-1], dtype=obj_set.dtype, device=device)
    if isinstance(step_size_search, Real):
        if step_size_search <= 0:
            raise ValueError("step_size_search must be a positive number.")
        initial_step_size = initial_step_size * float(step_size_search)
        step_size_search = 'fixed'

    if line_search_params is None:
        line_search_params = (1e18, 2)

    # Step-size history is small for fixed/line-search modes, but it is still
    # moved to CPU so that the Python list does not retain CUDA allocations.
    step_size_history = []
    idx = 0

    # Reconstruction loop.
    for m in tqdm(range(max_iter), desc=method):
        previous_obj_set = obj_set.clone()
        obj_set, initial_step_size = update_fft(
            dset, previous_obj_set, psf_fft, psf_m_fft, lam, weight, eps, m,
            method, regularizer_norm, normalize, step_size_search,
            initial_step_size=initial_step_size, line_search_params=line_search_params,
            verbose=verbose,
        )
        step_size_history.append(initial_step_size.detach().cpu())

        # Save either the whole history on CPU or overwrite the single output slot.
        idx = m + 1 if rec_to_save == 'all' else 0
        
        if return_pointwise_sol:
            obj_all[idx] = obj_set.sum(-1).detach().to(history_device)
        else:
            obj_all[idx] = obj_set.detach().to(history_device)

        if stop_tol is not None:
            if stop_criteria == 'rel_diff':
                criteria = (
                    torch.abs(obj_set[idx_in_focus] - previous_obj_set[idx_in_focus])
                    / (previous_obj_set[idx_in_focus] + eps)
                ).sum()
            elif stop_criteria == 'step_size':
                criteria = initial_step_size.max()
            else:
                raise ValueError("stop_criteria must be 'rel_diff' or 'step_size'.")

            if criteria < stop_tol:
                if rec_to_save == 'all':
                    obj_all = obj_all[:idx + 1]
                break

    return obj_all, step_size_history


def variance_reconstruction(dset, psf, lambda_value, weight_map=1, max_iter=100, stop_tol=None, method='sgp',
                            regularizer_norm='l2', initialization='flat', normalize=True, process='gpu',
                            stop_criteria='rel_diff', idx_in_focus=0, rec_to_save='last',
                            return_pointwise_sol=False, step_size_search='fixed', line_search_params=None,
                            verbose=False):
    """Reconstruct confocal, MID, or s2ISM data using one common internal layout.

    Parameters
    ----------
    dset : np.ndarray or torch.Tensor
        Confocal ``(Nx, Ny, K)`` or multichannel ``(Nx, Ny, Nch, K)`` data.
    psf : np.ndarray or torch.Tensor
        Confocal ``(Mx, My)``, MID ``(Mx, My, Nch)``, or s2ISM
        ``(Nz, Mx, My, Nch)`` PSF.
    lambda_value : float
        Variance-regularization parameter.
    weight_map : scalar or array-like, default=1
        Spatial regularization weight.
    max_iter : int, default=100
        Maximum number of iterations.
    stop_tol : float or None, default=None
        Early-stopping tolerance.
    method : {'sgp', 'sgm', 'pgd', 'osl', 'abs'}, default='sgp'
        Optimization method.
    regularizer_norm : {'l2', 'l1', 'kl'}, default='l2'
        Consistency regularizer.
    initialization : {'flat', 'data', 'rl'}, default='flat'
        Initialization method.
    normalize : bool, default=True
        Apply flux normalization after every update.
    process : {'cpu', 'gpu'}, default='gpu'
        Processing device. GPU falls back to CPU when CUDA is unavailable.
    stop_criteria : {'rel_diff', 'step_size'}, default='rel_diff'
        Early-stopping criterion.
    idx_in_focus : int, default=0
        Axial plane used by ``rel_diff``.
    rec_to_save : {'last', 'all'}, default='last'
        Save only the current/final reconstruction or every iteration.
    return_pointwise_sol : bool, default=False
        Sum over the realization axis ``K`` before saving the reconstruction.
    step_size_search : {'fixed', 'line_search', 'rl'} or positive scalar
        Step-size strategy.
    line_search_params : tuple or None
        Parameters ``(s1, s2)`` of the SGP metric truncation.
    verbose : bool, default=False
        Print device and line-search information.

    Returns
    -------
    obj_all : torch.Tensor
        Reconstruction with a leading save axis.
    step_size_history : list[torch.Tensor]
        Step sizes for each completed iteration.

    Notes
    -----
    Inputs are standardized once to:

    - ``dset -> (Nx, Ny, Nch, K)``
    - ``psf  -> (Nz, Mx, My, Nch)``

    so the optimizer itself always works in the full s2ISM dimensional layout.
    """
    device = torch.device('cuda:0' if process == 'gpu' and torch.cuda.is_available() else 'cpu')
    
    if verbose:
        print(f"Computing reconstruction on device: {device}")

    # Standardize the three supported acquisition layouts only once.
    dset = standardize_dataset(dset, device=device, dtype=torch.float32)
    psf = standardize_psf(psf, device=device, dtype=torch.float32)
    object_shape = (psf.shape[0], dset.shape[0], dset.shape[1], dset.shape[-1])
    weight = standardize_object_weight(weight_map, object_shape, device=device, dtype=dset.dtype)

    with torch.no_grad():
        return iterate(
            dset, psf, max_iter, lambda_value, weight, stop_tol, method, device,
            regularizer_norm, initialization, normalize, stop_criteria, rec_to_save,
            step_size_search, line_search_params, idx_in_focus, return_pointwise_sol, verbose,
        )


def _batch_starts(length: int, window: int, overlap: int) -> list[int]:
    """Return tile start positions covering one image axis.

    Parameters
    ----------
    length : int
        Full axis length.
    window : int
        Tile size along the axis.
    overlap : int
        Number of overlapping pixels between adjacent tiles.

    Returns
    -------
    list[int]
        Starting indices of all tiles.
    """
    if window >= length:
        return [0]
    stride = window - overlap
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def _crop_weight_map(weight_map, x_slice, y_slice, image_shape):
    """Crop a spatial weight map to the current reconstruction tile.

    Scalars or weights without the full image spatial shape are returned
    unchanged.
    """
    weight = torch.as_tensor(weight_map) if isinstance(weight_map, torch.Tensor) else np.asarray(weight_map)
    if weight.ndim >= 2 and tuple(weight.shape[:2]) == tuple(image_shape):
        return weight[x_slice, y_slice, ...]
    return weight_map


def batch_reconstruction(dset: np.ndarray, psf: np.ndarray, batch_size: list, overlap: int,
                         lambda_value, weight_map=1, max_iter=100, stop_tol=None, method='sgp',
                         regularizer_norm='l2', initialization='flat', normalize=True, process='gpu',
                         stop_criteria='rel_diff', return_pointwise_sol=False, rec_to_save='last',
                         step_size_search='fixed', line_search_params=None, idx_in_focus=0, verbose=False):
    """Reconstruct a large field of view using overlapping spatial tiles.

    Parameters
    ----------
    dset : np.ndarray
        Dataset ``(Nx, Ny, K)`` or ``(Nx, Ny, Nch, K)``.
    psf : np.ndarray
        Confocal, MID, or s2ISM PSF.
    batch_size : sequence of int
        Tile size ``[Nx_batch, Ny_batch]``.
    overlap : int
        Spatial overlap between adjacent tiles.
    lambda_value : float
        Variance-regularization parameter.
    weight_map : scalar or array-like, default=1
        Spatial regularization weight. Full-size maps are cropped per tile.
    max_iter, stop_tol, method, regularizer_norm, initialization, normalize,
    process, stop_criteria, return_pointwise_sol, rec_to_save,
    step_size_search, line_search_params, idx_in_focus, verbose
        Passed directly to :func:`variance_reconstruction`.

    Returns
    -------
    np.ndarray
        Reconstructed image/volume assembled from the central part of each tile.
    """
    wx, wy = map(int, batch_size)
    dset_array = np.asarray(dset)
    nx, ny = dset_array.shape[:2]
    x_starts = _batch_starts(nx, min(wx, nx), overlap)
    y_starts = _batch_starts(ny, min(wy, ny), overlap)

    reconstruction = None
    n_batches = len(x_starts) * len(y_starts)
    batch_index = 0

    for x0 in x_starts:
        for y0 in y_starts:
            batch_index += 1
            if verbose:
                print(f"Batch {batch_index}/{n_batches}")

            x1, y1 = min(x0 + wx, nx), min(y0 + wy, ny)
            x_slice, y_slice = slice(x0, x1), slice(y0, y1)
            crop_weight = _crop_weight_map(weight_map, x_slice, y_slice, (nx, ny))

            recon, _ = variance_reconstruction(
                dset_array[x_slice, y_slice, ...], psf, lambda_value,
                weight_map=crop_weight, max_iter=max_iter, stop_tol=stop_tol,
                method=method, regularizer_norm=regularizer_norm,
                initialization=initialization, normalize=normalize, process=process,
                stop_criteria=stop_criteria, idx_in_focus=idx_in_focus,
                rec_to_save=rec_to_save, return_pointwise_sol=return_pointwise_sol,
                step_size_search=step_size_search, line_search_params=line_search_params,
                verbose=verbose,
            )
            recon = recon.detach().cpu().numpy()

            if reconstruction is None:
                output_shape = list(recon.shape)
                output_shape[2], output_shape[3] = nx, ny
                reconstruction = np.zeros(output_shape, dtype=recon.dtype)

            # Keep only the non-overlapping central portion of each tile.
            trim_left = overlap // 2 if x0 > 0 else 0
            trim_right = overlap - overlap // 2 if x1 < nx else 0
            trim_top = overlap // 2 if y0 > 0 else 0
            trim_bottom = overlap - overlap // 2 if y1 < ny else 0

            out_x = slice(x0 + trim_left, x1 - trim_right)
            out_y = slice(y0 + trim_top, y1 - trim_bottom)
            patch_x = slice(trim_left, recon.shape[2] - trim_right)
            patch_y = slice(trim_top, recon.shape[3] - trim_bottom)
            reconstruction[:, :, out_x, out_y, ...] = recon[:, :, patch_x, patch_y, ...]

    return reconstruction