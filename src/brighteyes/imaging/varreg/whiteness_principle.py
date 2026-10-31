# Origin: VicidominiLab/Regularization-by-Noise, src/varreg/whiteness_principle.py at bb6d0c6.

import numpy as np
import torch
from torch.fft import fftn
from typing import Callable, Optional

from .optimizer import variance_reconstruction
from .utils import forward_model, standardize_dataset, standardize_psf


class VarianceReconstructor:
    """Picklable wrapper around :func:`variance_reconstruction`.

    The object stores all reconstruction parameters except ``lambda_value``.
    This makes it convenient to evaluate the same reconstruction problem over
    a grid of candidate regularization values.

    Parameters
    ----------
    dset : np.ndarray or torch.Tensor
        Confocal ``(Nx, Ny, K)`` or multichannel ``(Nx, Ny, Nch, K)`` data.
    psf : np.ndarray or torch.Tensor
        Confocal ``(Mx, My)``, MID ``(Mx, My, Nch)``, or s2ISM
        ``(Nz, Mx, My, Nch)`` PSF.
    weight_map : scalar or array-like
        Spatial regularization weight.
    stop_tol, stop_criteria, max_iter, method, regularizer_norm, initialization,
    idx_in_focus, normalize, process, rec_to_save, step_size_search,
    line_search_params, verbose
        Parameters forwarded unchanged to :func:`variance_reconstruction`.
    """

    def __init__(self, dset, psf, weight_map=1, stop_tol=1e-5, stop_criteria: str = 'step_size',
                 max_iter: int = 50, method: str = 'sgp', regularizer_norm: str = 'l2',
                 initialization: str = 'flat', idx_in_focus: int = 0, normalize: bool = True,
                 process: str = 'gpu', rec_to_save: str = 'last', step_size_search: str = 'fixed',
                 line_search_params=None, verbose=False):
        self.dset = dset
        self.psf = psf
        self.weight_map = weight_map
        self.stop_tol = stop_tol
        self.stop_criteria = stop_criteria
        self.max_iter = max_iter
        self.method = method
        self.regularizer_norm = regularizer_norm
        self.initialization = initialization
        self.idx_in_focus = idx_in_focus
        self.normalize = normalize
        self.process = process
        self.rec_to_save = rec_to_save
        self.step_size_search = step_size_search
        # Keep the historical whiteness-wrapper default.
        self.line_search_params = line_search_params if line_search_params is not None else (10, 2)
        self.verbose = verbose

    def __call__(self, lambda_value: float):
        """Run one reconstruction for the requested regularization value."""
        return variance_reconstruction(
            self.dset, self.psf, lambda_value,
            weight_map=self.weight_map,
            max_iter=self.max_iter,
            stop_tol=self.stop_tol,
            stop_criteria=self.stop_criteria,
            method=self.method,
            process=self.process,
            regularizer_norm=self.regularizer_norm,
            initialization=self.initialization,
            normalize=self.normalize,
            idx_in_focus=self.idx_in_focus,
            rec_to_save=self.rec_to_save,
            step_size_search=self.step_size_search,
            line_search_params=self.line_search_params,
            verbose=self.verbose,
        )


def make_variance_reconstructor(dset, psf, weight_map=1, stop_tol=1e-5,
                                stop_criteria: str = 'step_size', max_iter: int = 50,
                                method: str = 'sgp', regularizer_norm: str = 'l2',
                                initialization: str = 'flat', normalize: bool = True,
                                process: str = 'gpu', idx_in_focus: int = 0,
                                rec_to_save: str = 'last', step_size_search: str = 'fixed',
                                line_search_params=None, verbose=False) -> Callable:
    """Create a callable solver for a PRWP search over ``lambda_value``.

    The returned :class:`VarianceReconstructor` accepts one scalar lambda and
    returns the same ``(reconstruction, step_sizes)`` pair as
    :func:`variance_reconstruction`.
    """
    return VarianceReconstructor(
        dset=dset, psf=psf, weight_map=weight_map,
        stop_tol=stop_tol, stop_criteria=stop_criteria, max_iter=max_iter,
        method=method, regularizer_norm=regularizer_norm,
        initialization=initialization, normalize=normalize, process=process,
        idx_in_focus=idx_in_focus, rec_to_save=rec_to_save,
        step_size_search=step_size_search, line_search_params=line_search_params,
        verbose=verbose,
    )


def T_func(y: torch.Tensor) -> torch.Tensor:
    """Compute ``T(y) = 1 / (1 - exp(-y))`` for the masked PRWP.

    Parameters
    ----------
    y : torch.Tensor
        Predicted channel-integrated photon counts.

    Returns
    -------
    torch.Tensor
        ``T(y)`` evaluated in float64.
    """
    y = y.type(dtype=torch.float64)
    return 1.0 / (1.0 - torch.exp(-y))


def V_func(y: torch.Tensor) -> torch.Tensor:
    """Compute the ``V(y)`` factor used by the masked PRWP.

    Parameters
    ----------
    y : torch.Tensor
        Predicted channel-integrated photon counts.

    Returns
    -------
    torch.Tensor
        ``V(y)`` evaluated in float64.
    """
    y = y.type(dtype=torch.float64)
    exp_neg_y = torch.exp(-y)
    return (1.0 - (1.0 + y) * exp_neg_y) / ((1.0 - exp_neg_y) ** 2)


def compute_z_variable(obj: torch.Tensor, psf_fft: torch.Tensor, dset: torch.Tensor,
                       masked: bool = False, remove_border: bool = False) -> torch.Tensor:
    """Compute the standardized residual used by the whiteness criterion.

    Parameters
    ----------
    obj : torch.Tensor
        Reconstructed object ``(Nz, Nx, Ny, K)``.
    psf_fft : torch.Tensor
        OTF in canonical layout ``(Nz, Nx, Ny_rfft, Nch)``. Reduced confocal
        or MID layouts are accepted and expanded with singleton axes.
    dset : torch.Tensor
        Measured data. Confocal ``(Nx, Ny, K)`` and multichannel
        ``(Nx, Ny, Nch, K)`` layouts are accepted.
    masked : bool, default=False
        Use the zero-truncated Poisson residual when true.
    remove_border : bool or int, default=False
        If true, remove five pixels from each border. If an integer is given,
        remove that many pixels.

    Returns
    -------
    torch.Tensor
        Standardized residual ``(Nx_crop, Ny_crop, K)``.

    """
    eps = 1e-10

    # Put confocal/MID inputs into the same maximum-dimensional layout used by
    # s2ISM. From this point onward there is only one code path.
    dset = standardize_dataset(dset, device=obj.device)
    psf_fft = standardize_psf(psf_fft, device=obj.device)

    # Forward the reconstructed object and sum first over axial planes, then
    # over detector channels. Both measured and predicted data are therefore
    # reduced to (Nx, Ny, K).
    y_pred = forward_model(psf_fft, obj).sum(axis=0)
    y_pred = y_pred.sum(-2)
    dset = dset.sum(-2)

    # Optional crop used to suppress convolution border effects in the PRWP.
    if remove_border is True:
        crop = 5
    elif remove_border is False:
        crop = 0
    elif isinstance(remove_border, int):
        if remove_border > 0:
            crop = remove_border
        else:
            raise ValueError("remove_border must be a positive integer.")
    else:
        raise ValueError("remove_border must be a boolean or an integer.")

    if crop > 0:
        y_pred = y_pred[crop:-crop, crop:-crop, ...]
        dset = dset[crop:-crop, crop:-crop, ...]

    # Standard or zero-truncated Poisson residual.
    if masked:
        num = dset - y_pred * T_func(y_pred)
        den = torch.sqrt(y_pred * V_func(y_pred))
    else:
        num = dset - y_pred
        den = torch.sqrt(y_pred)

    # Preserve the original active-pixel rule and epsilon placement.
    z_var = torch.where(dset > eps, num / (den + eps), 0.0)
    return z_var


def compute_whiteness_measure(z: torch.Tensor) -> float:
    """Compute the autocorrelation-based whiteness measure in Fourier space.

    Parameters
    ----------
    z : torch.Tensor
        Standardized residual ``(Nx, Ny, K)``.

    Returns
    -------
    float
        Sum over realizations of
        ``sum(|FFT(z)|^4) / sum(|FFT(z)|^2)^2``.

    """
    
    corr_axis = (0, 1)
    zhat = fftn(z, dim=corr_axis)
    num = torch.sum(torch.abs(zhat) ** 4, dim=corr_axis)
    energy = torch.sum(torch.abs(zhat) ** 2, dim=corr_axis)
    denom = energy ** 2
    autocor = num / denom
    whiteness = autocor.real.sum()
    return float(whiteness)


def choose_lambda_by_WP(solver_fn: Callable[[float], torch.Tensor], lambdas,
                         dset: torch.Tensor, psf: torch.Tensor,
                         masked: bool = False, remove_border: bool = False,
                         device: Optional[torch.device] = None,
                         return_all_iters: bool = False,
                         return_all_solutions: bool = False,
                         return_z_variables: bool = False,
                         return_whiteness_list: bool = False,
                         idx_in_focus: Optional[int] = 0,
                         verbose: bool = False):
    """Evaluate candidate lambdas and select the minimum residual whiteness.

    Parameters
    ----------
    solver_fn : callable
        Function taking one lambda and returning
        ``(obj_history, step_size_history)``. ``obj_history`` has canonical
        shape ``(Nsave, Nz, Nx, Ny, K)``.
    lambdas : iterable
        Candidate regularization values.
    dset : torch.Tensor
        Confocal ``(Nx, Ny, K)`` or multichannel ``(Nx, Ny, Nch, K)`` data.
    psf : torch.Tensor
        Real PSF or complex OTF in confocal, MID, or s2ISM layout.
    masked : bool, default=False
        Use the masked zero-truncated Poisson residual.
    remove_border : bool or int, default=False
        Border crop passed to :func:`compute_z_variable`.
    device : torch.device or None
        Device used for residual and whiteness calculations.
    return_all_iters : bool, default=False
        Return the saved reconstruction history for every lambda. This can be
        extremely large and is intended only for diagnostic runs.
    return_all_solutions : bool, default=False
        Return the final in-focus reconstruction for every lambda.
    return_z_variables : bool, default=False
        Also return standardized residuals and predicted detector data.
    return_whiteness_list : bool, default=False
        Return only the whiteness values for all lambdas.
    idx_in_focus : int, default=0
        Axial plane stored in diagnostic outputs.
    verbose : bool, default=False
        Print progress information.

    Returns
    -------
    Depending on the selected return flags, either diagnostic lists or
    ``(best_lambda, best_obj, best_whiteness)``.

    """
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if verbose:
        print(f"Using device: {device}")

    # Standardize once at the public entry point. The rest of the function can
    # assume the same full-dimensional representation used by s2ISM.
    dset = standardize_dataset(dset, device=device)
    psf_fft = standardize_psf(psf, device=device, dtype=dset.dtype, target_shape=dset.shape[:2])

    best_lambda = None
    best_obj = None
    best_whiteness = float('inf')

    obj_list = []
    whiteness_measures_list = []
    obj_lam_all = []
    y_pred_list = []
    z_list = []

    for idx_n, lam in enumerate(lambdas):
        if verbose:
            print(f"{idx_n + 1}/{len(lambdas)}: Testing lambda={lam}")

        # No autograd graph is required: both the optimizer and PRWP use
        # explicitly implemented updates/statistics.
        with torch.inference_mode():
            obj_lam, _ = solver_fn(lam)

            # optimizer.py keeps rec_to_save='all' histories on CPU. Move only
            # the final reconstruction to the working device for the PRWP.
            if return_all_iters:
                obj_lam_all.append(obj_lam[:, idx_in_focus, ...].detach().cpu().numpy())
            obj_last = obj_lam[-1].detach().to(device)

            if return_all_solutions or return_z_variables:
                obj_list.append(obj_last[idx_in_focus].detach().cpu().numpy())

            # Release the solver history before computing/retaining further
            # diagnostics. With large images this prevents an unnecessary
            # overlap between the reconstruction history and PRWP arrays.
            del obj_lam

            z = compute_z_variable(obj_last, psf_fft, dset, masked=masked, remove_border=remove_border)
            whiteness_measure = compute_whiteness_measure(z) * z[..., 0].numel()

            if return_whiteness_list or return_all_solutions or return_all_iters or return_z_variables:
                whiteness_measures_list.append([whiteness_measure])

            if return_z_variables:
                y_pred = forward_model(psf_fft, obj_last).sum(0)
                y_pred_list.append(y_pred.detach().cpu().numpy())
                z_list.append(z.detach().cpu().numpy().squeeze())

            if verbose:
                print(f"  Whiteness={whiteness_measure:.8g}")

            if not return_all_solutions and not return_all_iters and not return_z_variables:
                if abs(whiteness_measure) < best_whiteness:
                    best_whiteness = whiteness_measure
                    best_lambda = lam
                    # Keep the best solution off-GPU while the next candidate
                    # lambda is reconstructed.
                    best_obj = obj_last.detach().cpu().clone()

        # Drop per-lambda GPU tensors before starting the next reconstruction.
        del obj_last, z
        if return_z_variables:
            del y_pred

    if return_z_variables:
        return obj_list, whiteness_measures_list, z_list, y_pred_list
    if return_all_iters:
        return obj_lam_all, whiteness_measures_list
    if return_all_solutions:
        return obj_list, whiteness_measures_list
    if return_whiteness_list:
        return whiteness_measures_list

    return float(best_lambda), best_obj, float(best_whiteness)