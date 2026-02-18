import torch
import numpy as np
from typing import Callable, Optional, Tuple
from functools import partial

from .loss import fidelity_loss
from .sgm_optimizer import variance_reconstruction
from .utils import forward, kl_divergence, centered_pad

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
                 max_iter: int = 50,
                 method: str = 'sgp',
                 regularizer_norm: str = 'l2',
                 initialization: str = 'flat',
                 normalize: bool = True,
                 process: str = 'gpu',
                 rec_to_save: str = 'last',
                 binomial: bool = False,
                 step_size_search: str = 'fixed',
                 line_search_params=None,
                 verbose=False):
        self.dset = dset
        self.psf = psf
        self.alpha = alpha
        self.weight_map = weight_map
        self.stop_tol = stop_tol
        self.max_iter = max_iter
        self.method = method
        self.regularizer_norm = regularizer_norm
        self.initialization = initialization
        self.normalize = normalize
        self.process = process
        self.rec_to_save = rec_to_save
        self.binomial = binomial
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
            method=self.method, 
            process=self.process,
            regularizer_norm=self.regularizer_norm, 
            initialization=self.initialization,
            normalize=self.normalize, 
            rec_to_save=self.rec_to_save, 
            binomial=self.binomial,
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
                                max_iter: int = 50,
                                method: str = 'sgp',
                                regularizer_norm: str = 'l2',
                                initialization: str = 'flat',
                                normalize: bool = True,
                                process: str = 'gpu',
                                rec_to_save: str = 'last',
                                binomial: bool = False,
                                step_size_search: str = 'fixed',
                                line_search_params=None,
                                verbose=False,
                                device: Optional[torch.device] = None) -> Callable[[float], torch.Tensor]:
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
        rec_to_save=rec_to_save, binomial=binomial, 
        step_size_search=step_size_search, line_search_params=line_search_params,
        verbose=verbose
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
    

def compute_z_variable(obj: torch.Tensor, psf_fft: torch.Tensor, dset: torch.Tensor, masked: bool = False) -> torch.Tensor:
    """
    Compute the z-transform of the reconstructed object.
    obj: reconstructed object (Nx, Ny, Nch, K)
    psf_fft: precomputed FFT of the PSF in frequency domain
    Returns: z-transform
    """
    
    eps = 1e-10  # torch.finfo(torch.float32).eps
    
    y_pred = forward(psf_fft, obj) + eps  # predicted data
    idx_pos = dset > eps
    if masked:
        num = dset - y_pred * T_func(y_pred)
        den = torch.sqrt(y_pred * V_func(y_pred))
        z_var = torch.where(idx_pos & (den > eps), num / den, 0.0)
    else:
        num = dset - y_pred
        den = torch.sqrt(y_pred)
        z_var = torch.where(den > eps, num / den, 0.0)
    
    return z_var

def compute_whiteness_measure(z: torch.Tensor) -> float:
    """
    Compute W(Z) from eq. 10 via eq. 11 for a 2D real/complex tensor Z.
    Assumes periodic boundary conditions. Returns a non-negative scalar.
    """
    #energy = torch.sum(torch.abs(z)**2, axis=(0,1))
    zhat = fftn(z, dim=(0,1))  # unnormalized DFT
    num = torch.sum(torch.abs(zhat)**4) #, axis=(0,1))
    energy = torch.sum(torch.abs(zhat)**2) #, axis=(0,1))
    denom = energy**2
    autocor = (num / denom).real #.sum()
    return float(autocor)

def _evaluate_single_lambda(lam: float, 
                           solver_fn: Callable[[float], torch.Tensor],
                           img: torch.Tensor,
                           psf_fft: torch.Tensor,
                           masked: bool,
                           device: torch.device,
                           return_all_iters: bool,
                           verbose: bool) -> Tuple:
    """
    Helper function to evaluate a single lambda value.
    Returns: (lam, obj_lam, whiteness_measure, obj_lam_last)
    """
    import gc
    
    if verbose:
        print(f'Testing lambda={lam}')
    
    # Force CPU and clear any GPU cache
    torch.set_num_threads(1)  # Prevent nested parallelism
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    obj_lam, _ = solver_fn(lam)
    z = compute_z_variable(obj_lam[-1].to(device), psf_fft, img.to(device), masked=masked)
    whiteness_measure = compute_whiteness_measure(z) * z.numel()
    
    if verbose:
        print(f'  Whiteness={whiteness_measure:.6g}')
    
    # Extract results and clean up immediately
    obj_lam_last = obj_lam[-1].cpu().clone()
    obj_lam_full = obj_lam.cpu().clone() if return_all_iters else None
    
    # Force cleanup
    del obj_lam, z
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return (lam, obj_lam_full, whiteness_measure, obj_lam_last)


def choose_lambda_by_WP(solver_fn: Callable[[float], torch.Tensor],
                         lambdas,
                         img: torch.Tensor,
                         psf: torch.Tensor,
                         masked: bool = False,
                         device: Optional[torch.device] = None,
                         return_all_iters: bool = False,
                         return_all: bool = False,
                         verbose: bool = False,
                         parallel: bool = False,
                         max_workers: Optional[int] = None):
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

    img = img.to(device)
    psf = psf.to(device)
    best_lambda = None
    best_obj = None
    best_whiteness = float('inf')
    
    if psf.is_floating_point():
        conv_dims = 2
        axis_to_conv = tuple(range(conv_dims))
        sz = img.shape[:conv_dims]
        psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape
        # Normalize PSF
        psf_norm = psf_pad / psf_pad.sum()
        # Precompute FFTs
        psf_fft = rfftn(psf_norm, dim=axis_to_conv)
    elif torch.is_complex(psf):
        psf_fft = psf
        
    if return_all:
        obj_list = []
        whiteness_measures_list = []

    if parallel:
        if not JOBLIB_AVAILABLE:
            raise ImportError("joblib is required for parallel execution. Install with: pip install joblib")
        
        # Parallel execution using joblib (better handling of Python objects)
        if verbose:
            print(f'Running in parallel with n_jobs={max_workers if max_workers else -1}')
        
        # Use joblib's Parallel with 'loky' backend (works well on Windows)
        results = Parallel(n_jobs=max_workers if max_workers else -1, 
                          backend='loky',
                          verbose=10 if verbose else 0)(
            delayed(_evaluate_single_lambda)(
                lam, solver_fn, img.cpu(), psf_fft.cpu(), 
                masked, torch.device('cpu'), return_all_iters, False
            ) for lam in lambdas
        )
        
        # Process results
        for lam, obj_lam_full, whiteness_measure, obj_lam_last in results:
            if return_all:
                obj_list.append(obj_lam_last.numpy())
                whiteness_measures_list.append([whiteness_measure])
            
            if verbose:
                print(f'Lambda={lam}: Whiteness={whiteness_measure:.6g}')
            
            if abs(whiteness_measure) < best_whiteness:
                best_whiteness = whiteness_measure
                best_lambda = lam
                if return_all_iters:
                    best_obj = obj_lam_full
                else:
                    best_obj = obj_lam_last
    else:
        # Sequential execution (original code)
        for idx_n, lam in enumerate(lambdas):
            if verbose:
                print(f'{idx_n+1}/{len(lambdas)}: Testing lambda={lam}')

            obj_lam, _ = solver_fn(lam)
            z = compute_z_variable(obj_lam[-1].to(device), psf_fft, img.to(device), masked=masked)
            whiteness_measure = compute_whiteness_measure(z) * z.numel()
            
            if return_all_iters:
                obj_lam_all = []
                obj_lam_all.append(obj_lam.cpu().numpy())
                
            if return_all:
                obj_list.append(obj_lam[-1].cpu().numpy())
                whiteness_measures_list.append([whiteness_measure])
                
            if verbose:
                print(f'  Whiteness={whiteness_measure:.6g}')
            if abs(whiteness_measure) < best_whiteness:
                best_whiteness = whiteness_measure
                best_lambda = lam
                if return_all_iters:
                    best_obj = obj_lam
                else:
                    best_obj = obj_lam[-1]

    if best_lambda is None:
        raise ValueError('No lambdas were provided or evaluation failed.')
    if return_all:
        return (float(best_lambda), best_obj, float(best_whiteness), obj_list, whiteness_measures_list)
    else:
        return float(best_lambda), best_obj, float(best_whiteness)