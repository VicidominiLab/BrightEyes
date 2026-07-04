import torch
import numpy as np
from typing import Callable, Optional, Tuple
from functools import partial

from .loss import fidelity_loss
from .sgm_optimizer import variance_reconstruction
from .utils import compute_otf_from_psf, forward_model

from torch.fft import fftn, rfftn, irfftn, ifftshift
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("Warning: joblib not available. Install with 'pip install joblib' for parallel execution.")

# Reconstructor class for serialization compatibility with multiprocessing
class VarianceReconstructor:
    """
    Callable class for variance reconstruction that is picklable for multiprocessing.
    
    This replaces the nested function approach in make_variance_reconstructor to enable
    parallel processing across multiple lambda values.
    """
    def __init__(self, 
                 dset: np.ndarray,
                 psf: np.ndarray,
                 alpha: int = 1,
                 weight_map=1,
                 stop_tol=1e-5,
                 stop_criteria: str = 'step_size',
                 max_iter: int = 50,
                 method: str = 'sgp',
                 regularizer_norm: str = 'l2',
                 initialization: str = 'flat',
                 idx_in_focus: int = 0,
                 normalize: bool = True,
                 process: str = 'gpu',
                 rec_to_save: str = 'last',
                 step_size_search: str = 'fixed',
                 line_search_params=None,
                 verbose=False):
        self.dset = dset
        self.psf = psf
        self.alpha = alpha
        self.weight_map = weight_map
        self.stop_tol = stop_tol
        self.stop_criteria = stop_criteria
        self.max_iter = max_iter
        self.method = method
        self.regularizer_norm = regularizer_norm
        self.initialization = initialization
        self.normalize = normalize
        self.process = process
        self.rec_to_save = rec_to_save
        self.idx_in_focus = idx_in_focus
        self.step_size_search = step_size_search
        self.line_search_params = line_search_params if line_search_params is not None else (10, 2)
        self.verbose = verbose
    
    def __call__(self, lambda_value: float) -> torch.Tensor:
        """Call the reconstructor with a specific lambda value."""
        obj = variance_reconstruction(
            self.dset, self.psf, lambda_value, 
            alpha_value=self.alpha, 
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
            verbose=self.verbose
        )
        return obj


# Example reconstructor wrapper using the repository `variance_reconstruction`:
def make_variance_reconstructor(dset: np.ndarray,
                                psf: np.ndarray,
                                alpha: int = 1,
                                weight_map=1,
                                stop_tol = 1e-5,
                                stop_criteria: str = 'step_size',
                                max_iter: int = 50,
                                method: str = 'sgp',
                                regularizer_norm: str = 'l2',
                                initialization: str = 'flat',
                                normalize: bool = True,
                                process: str = 'gpu',
                                idx_in_focus: int = 0,
                                rec_to_save: str = 'last',
                                step_size_search: str = 'fixed',
                                line_search_params=None,
                                verbose=False) -> Callable[[float], torch.Tensor]:
    """
    Create a callable that takes `lambda_value` and returns the reconstructed object
    using the `variance_reconstruction` function in this repo.

    Note: inputs `dset` and `psf` are numpy arrays and will be converted inside
    `variance_reconstruction` to torch tensors. This wrapper avoids repeated
    preprocessing outside the reconstructor.
    
    Returns a VarianceReconstructor instance which is picklable for multiprocessing.
    """
    
    if line_search_params is None:
        line_search_params = (10, 2)

    return VarianceReconstructor(
        dset=dset, psf=psf, alpha=alpha, weight_map=weight_map,
        stop_tol=stop_tol, max_iter=max_iter, method=method, regularizer_norm=regularizer_norm,
        initialization=initialization, normalize=normalize, process=process,
        rec_to_save=rec_to_save, stop_criteria=stop_criteria,
        step_size_search=step_size_search, line_search_params=line_search_params,
        verbose=verbose, idx_in_focus=idx_in_focus
    )

def T_func(y: torch.Tensor) -> torch.Tensor:
    """
    T(y) function from eq. 7.
    y: forwarded predicted data (Nx, Ny, Nch, K)
    Returns: T(y)
    """
    
    y = y.type(dtype=torch.float64)
    T = 1.0 / (1.0 - torch.exp(-y))
    return T   

def V_func(y: torch.Tensor) -> torch.Tensor:
    """
    V(y) function from eq. 8.
    y: forwarded predicted data (Nx, Ny, Nch, K)
    Returns: V(y)
    """
    
    y = y.type(dtype=torch.float64)
    exp_neg_y = torch.exp(-y)
    V = (1.0 - (1.0 + y) * exp_neg_y) / ((1.0 - exp_neg_y)**2)
    return V
    

def compute_z_variable(obj: torch.Tensor, psf_fft: torch.Tensor, dset: torch.Tensor, masked: bool = False, remove_border: bool = False) -> torch.Tensor:
    """
    Compute the z-transform of the reconstructed object.
    obj: reconstructed object (Nz, Nx, Ny, Nch, K)
    psf_fft: precomputed FFT of the PSF in frequency domain
    Returns: z-transform (Nx, Ny, K)
    """
    
    eps = 1e-10  # torch.finfo(torch.float32).eps
    
    y_pred = forward_model(psf_fft, obj).sum(axis=0)  # predicted data, summing over z
        
    y_pred = y_pred.sum(-2)
    dset = dset.sum(-2)
            
    if remove_border is True:
        # Remove border effects by cropping the predicted data and the dataset
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
        
    if masked:
        num = dset - y_pred * T_func(y_pred)
        den = torch.sqrt(y_pred * V_func(y_pred))
        #z_var = torch.where((dset > eps) & (den > eps), num / den, 0.0)
    else:
        num = dset - y_pred
        den = torch.sqrt(y_pred)
        #z_var = torch.where((dset > eps) & (den > eps), num / den, 0.0)
    
    z_var = torch.where(dset > eps, num / (den + eps), 0.0)
        
    return z_var

def compute_whiteness_measure(z: torch.Tensor) -> float:
    """
    Compute W(Z) with autocorrelation-based measure of whiteness for the z-transform.
    z: z-transform of the residual (Nx, Ny, K)
    """
    corr_axis = (0,1) # autocorrelation over x,y
    zhat = fftn(z, dim=corr_axis)  # Compute the Fourier transform along the x,y axes
    num = torch.sum(torch.abs(zhat)**4, dim=corr_axis) # sum over x,y
    energy = torch.sum(torch.abs(zhat)**2, dim=corr_axis) # sum over x,y
    denom = energy**2
    autocor = num / denom 
    whiteness = autocor.real.sum()  # sum over k and c axis 
    
    return float(whiteness)


def choose_lambda_by_WP(solver_fn: Callable[[float], torch.Tensor],
                         lambdas,
                         dset: torch.Tensor,
                         psf: torch.Tensor,
                         masked: bool = False,
                         remove_border: bool = False,
                         device: Optional[torch.device] = None,
                         return_all_iters: bool = False,
                         return_all_solutions: bool = False,
                         return_z_variables: bool = False,
                         idx_in_focus: Optional[int] = 0,
                         verbose: bool = False):
    """
    Search over candidate lambdas and pick the one minimizing the whiteness measure.
    solver_fn(λ) -> reconstructed object o(λ)
    lambdas: iterable of candidate λ values
    img: observed dataset (C, H, W) or (H, W)
    psf: PSF (H, W) or its FFT in frequency domain
    device: torch device to use
    return_all_iters: if True, return the full objective history for the best λ
    return_all: if True, return lists of all objectives and discrepancies for all λ
    verbose: if True, print progress
    parallel: if True, use multiprocessing to parallelize over lambdas
    max_workers: maximum number of parallel workers (None = number of CPUs)
    Returns:
    
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dset = dset.to(device)
    psf = psf.to(device)
    
    best_lambda = None
    best_obj = None
    best_whiteness = float('inf')
    
    # Ensure correct shapes of PSF w.r.t. dataset
    if dset.ndim <= 2:
        raise Exception('The dataset should have at least 3 dimension: Nx x Ny x K, where K are noise repetitions')
    
    if psf.ndim == dset.ndim -1: # PSF has not Nz dim
        psf = psf.unsqueeze(0)  # Add a z dimension #1 x Mx x My (x Ch)
        
    if psf.ndim == 3:
        psf = psf.unsqueeze(-1)  # Add a channel dimension at the end # Mz x Mx x My x 1 
    if dset.ndim == 3:
        dset = dset.unsqueeze(-2)  # Add a channel dimension at the end # Nx x Ny x 1 x K
        
    # psf = psf / psf.sum(dim=tuple(range(1, psf.ndim)), keepdim=True)  # Normalize PSF
    # print(f'sum of psf: {psf.sum().item()}')
        
    if psf.is_floating_point():         # If I have a PSF, I need to compute its FFT
        num_axis_to_conv = 2
        sz = tuple(dset.shape[:num_axis_to_conv])
        psf_fft = compute_otf_from_psf(psf, target_shape=sz)
    elif torch.is_complex(psf):
        psf_fft = psf
    else:
        raise ValueError(f"Unsupported PSF dtype: {psf.dtype}")
    
    if return_all_solutions:
        obj_list = []
        whiteness_measures_list = []

    obj_lam_all = []
    y_pred_list = []
    z_list = []

    for idx_n, lam in enumerate(lambdas):
        if verbose:
            print(f'{idx_n+1}/{len(lambdas)}: Testing lambda={lam}')

        obj_lam, _ = solver_fn(lam) # obj_lam shape: (num_iters, Nz, Nx, Ny, Nch, K)
        z = compute_z_variable(obj_lam[-1].to(device), psf_fft, dset.to(device), masked=masked, remove_border=remove_border)
        whiteness_measure = compute_whiteness_measure(z) * z[..., 0].numel()
        
        if return_z_variables:
            y_pred = forward_model(psf_fft, obj_lam[-1].to(device)).sum(0) #sum over z
            y_pred_list.append(y_pred)
                    
        if return_all_iters:
            obj_lam_all.append(obj_lam[:,idx_in_focus,...].cpu().numpy())
            
        if return_all_solutions:
            obj_list.append(obj_lam[-1,idx_in_focus].cpu().numpy())
            whiteness_measures_list.append([whiteness_measure])
            
        if return_z_variables:
            z_list.append(z.cpu().numpy().squeeze())
            
        if verbose:
            print(f'  Whiteness={whiteness_measure:.8g}')
            
        if return_all_solutions is False and return_all_iters is False and return_z_variables is False:    
            if abs(whiteness_measure) < best_whiteness:
                best_whiteness = whiteness_measure
                best_lambda = lam
                if return_all_iters:
                    best_obj = obj_lam
                else:
                    best_obj = obj_lam[-1]
    
    if return_all_solutions:
        results = ( obj_list, whiteness_measures_list )
    if return_all_iters:
        results = ( obj_lam_all, whiteness_measures_list )
    if return_z_variables:
        results = (obj_list, whiteness_measures_list, z_list, y_pred_list)
    else:
        results = (float(best_lambda), best_obj, float(best_whiteness))
    
    return results