import torch
import numpy as np
from typing import Callable, Optional, Tuple

from .loss import fidelity_loss
from .sgm_optimizer import variance_reconstruction
from .utils import forward, kl_divergence, centered_pad

from torch.fft import fftn, rfftn, irfftn, ifftshift
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

# Example reconstructor wrapper using the repository `variance_reconstruction`:
def make_variance_reconstructor(dset: np.ndarray,
                                psf: np.ndarray,
                                alpha: int = 1,
                                weight_map=1,
                                tol = 1e-5,
                                max_iter: int = 50,
                                method: str = 'sgp',
                                regularizer: str = 'l2',
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
    """
    
    if line_search_params is None:
        line_search_params = (10, 2)

    def recon(lambda_value: float) -> torch.Tensor:
        obj = variance_reconstruction(dset, psf, lambda_value, alpha_value=alpha, weight_map=weight_map,
                                      max_iter=max_iter, tol=tol, method=method, process=process,
                                      regularizer=regularizer, initialization=initialization,
                                      normalize=normalize, rec_to_save=rec_to_save, binomial=binomial,
                                      step_size_search=step_size_search, line_search_params=line_search_params, verbose=verbose)
        return obj

    return recon

def T_func(y: torch.Tensor) -> torch.Tensor:
    """
    T(y) function from eq. 7.
    y: forwarded predicted data (Nx, Ny, Nch, K)
    Returns: T(y)
    """
    
    T = 1.0 / (1.0 - torch.exp(-y))
    return T   

def V_func(y: torch.Tensor) -> torch.Tensor:
    """
    V(y) function from eq. 8.
    y: forwarded predicted data (Nx, Ny, Nch, K)
    Returns: V(y)
    """
    
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
    
    y_pred = forward(psf_fft, obj)  # predicted data
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

def choose_lambda_by_WP(solver_fn: Callable[[float], torch.Tensor],
                         lambdas,
                         img: torch.Tensor,
                         psf: torch.Tensor,
                         masked: bool = False,
                         device: Optional[torch.device] = None,
                         return_all_iters: bool = False,
                         return_all: bool = False,
                         verbose: bool = False):
    """
    Search over candidate lambdas and pick the one minimizing |D(λ)-ΔNE(λ)|.
    solver_fn(λ) -> reconstructed object o(λ)
    lambdas: iterable of candidate λ values
    img: observed dataset (C, H, W) or (H, W)
    psf: PSF (H, W) or its FFT in frequency domain
    device: torch device to use
    return_all_iters: if True, return the full objective history for the best λ
    return_all: if True, return lists of all objectives and discrepancies for all λ
    verbose: if True, print progress
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