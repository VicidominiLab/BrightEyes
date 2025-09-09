import torch
import string
import numpy as np

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift
from torch import real, einsum
from tqdm import tqdm

from .utils import partial_convolution_rfft, centered_pad

from .regularizers import grad_l2, grad_l1, grad_kl

from .preprocessing import binomial_split


def forward(otf, obj):
    """
    Performs the convolution between the OTF and an array with the same dimensionality or less (object or dataset).
    The dimension K is optional.

    Parameters:
        obj (Tensor): Object (Nx x Ny x K) or fraction (Nx x Ny x K) or (Nx x Ny x K x Nch).
        otf (Tensor): Fourier transform of the PSF (Nx x Ny x K) or (Nx x Ny x K x Nch).

    Returns:
        Tensor: Result of the convolution (Nx x Ny x K) or (Nx x Ny x K x Nch).
    """

    conv_dims = 2  # to generalize to 3D later

    alphabet = string.ascii_lowercase

    str_psf = alphabet[:otf.ndim]
    str_obj = alphabet[:obj.ndim]
    axis_str = alphabet[:conv_dims]

    dset = partial_convolution_rfft(otf, obj, dim1=str_psf, dim2=str_obj, axis=axis_str, fourier=(1, 0))

    return dset


def update_fft(img, obj, psf_fft, psf_m_fft, lam, alpha, weight, eps, method: str, regularizer: str):
    """
    Performs a single iteration update for SGM with variance regularization.

    Parameters:
        img (Tensor): Input data of shape Nx x Ny x K.
        obj (Tensor): Current estimate of the object of shape Nx x Ny x K.
        psf_fft (Tensor): Fourier transform of the PSF.
        psf_m_fft (Tensor): Fourier transform of the flipped PSF.
        lam (float): Variance regularization parameter.
        alpha (float): Fidelity term regularization parameter.
        weight (float): Regularization weight.
        eps (float): Epsilon to avoid division by zero.
        method (str): 'osl' or 'abs' or 'sgm'.
        regularizer (str): distance to measure the differences between o_k and o_mean: 'variance', 'kl', or 'l1'.

    Returns:
        Tensor: Updated object tensor of shape Nx x Ny x K.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === forward convolution ===
    img_estimate_k = forward(psf_fft, obj)
    fraction = img / (img_estimate_k + eps)

    del img_estimate_k

    update_k = alpha * forward(psf_m_fft, fraction)

    del fraction

    # === Metodo OSL ===
    if method.lower() == 'osl':
        obj_mean = obj.mean(dim=-1, keepdim=True)

        denom = 1 + 2 * lam * weight * (obj - obj_mean)
        obj_update = update_k / (denom + eps)

        obj_update.clamp_(min=1e-8)

        obj_new = obj * obj_update

    # === Metodo ABS ===
    elif method.lower() == 'abs':
        obj_mean = obj.mean(dim=-1, keepdim=True)

        denom = 1 + 2 * lam * weight * torch.abs(obj - obj_mean)
        obj_update = update_k / (denom + eps)
        obj_new = obj * obj_update

    # === Metodo SGM ===
    elif method.lower() == 'sgm':

        if regularizer == 'l2':
            u, v = grad_l2(obj)
        elif regularizer == 'kl':
            u, v = grad_kl(obj)
        elif regularizer == 'l1':
            u, v = grad_l1(obj)
        else:
            raise ValueError("Regularizer must be 'l2', 'kl', or 'l1'.")

        update_k = update_k + lam * weight * u

        denom = alpha * 1 + lam * weight * v
        obj_update = update_k / denom
        obj_new = obj * obj_update

    else:
        raise ValueError("Metodo non valido. Usare 'osl' o 'sgm'.")

    return obj_new


def iterate(dset, psf_fft, psf_m_fft, max_iter, lam, alpha, weight, tol, method, device, regularizer, initialization,
            normalize, rec_to_save, binomial):
    """
    Performs a single iteration update for the reconstruction process.

    Parameters:
        dset (Tensor): Input data of shape Nx x Ny x K x Ch.
        psf_fft (Tensor): Fourier transform of the PSF of shape Nx x Ny x K x Ch.
        psf_m_fft (Tensor): Fourier transform of the flipped PSF of shape Nx x Ny x K x Ch.
        max_iter (int): Maximum number of iterations.
        lam (float): Variance regularization parameter.
        alpha (float): Fidelity term regularization parameter.
        weight (float): Regularization weight a scalar or tensor of shape Nx x Ny.
        tol (float): Convergence tolerance to stop iterations.
        method (str): algorithm 'osl' or 'abs' or 'sgm'.
        device (str): device to perform computation on ('cpu' or 'cuda').
        regularizer (str): distance to measure the differences between o_k and o_mean: 'l2', 'kl', or 'l1'.
        initialization (str): initialization method for the object estimate ('flat' or 'data').
        normalize (bool): whether to normalize the object estimate at each iteration.
        rec_to_save (str): 'last' to return last iteration, 'all' to return all iterations.

    Returns:
        Tensor: Updated object tensor of shape Nx x Ny x K.
    """

    if rec_to_save == 'last' and tol is not None:
        raise ValueError("If rec_to_save is 'last', tol must be None.")

    eps = torch.finfo(torch.float32).eps

    if binomial is True:
        img = dset.sum(dim=-1)
        n_rep = dset.shape[-1]

    # Axis to sum for flux conservation
    axis_to_sum = tuple(range(dset.ndim - 1))
    flux = dset.sum(axis_to_sum, keepdim=True)

    # Initialization
    size = list(dset.shape)
    if rec_to_save == 'last':
        size = [1] + size
    elif rec_to_save == 'all':
        size = [max_iter + 1] + size

    obj_all = torch.empty(size).to(device)

    if initialization == 'flat':
        obj_all[0] = torch.ones(dset.shape).to(device) * flux / np.prod(dset.shape[:-1])
    elif initialization == 'data':
        axis_to_conv = tuple(range(2))
        obj_all[0] = torch.tensor(
            gaussian_filter(dset.numpy(), sigma=1, mode='constant', cval=0, truncate=4, axes=axis_to_conv)).to(device)
        obj_all[0] = obj_all[0] * flux / obj_all[0].sum()
    else:
        raise ValueError("Initialization must be 'flat' or 'data'.")

    idx = 0
    # Reconstruction loop
    for m in tqdm(range(max_iter), desc=method):
        obj_set = update_fft(dset, obj_all[idx], psf_fft, psf_m_fft, lam, alpha, weight, eps, method, regularizer)

        # flux conservation
        if normalize is True:
            obj_set = obj_set * flux / obj_set.sum(axis_to_sum, keepdim=True)

        if rec_to_save == 'all':
            idx = m + 1
        elif rec_to_save == 'last':
            idx = -1
        else:
            raise ValueError("rec_to_save must be 'last' or 'all'.")
        obj_all[idx] = obj_set

        if tol is not None:
            rel_diff = torch.norm(obj_all[m + 1] - obj_all[m]) / torch.norm(obj_all[m])
            if rel_diff.item() < tol:
                print(f"Converged at iteration {m + 1}")
                obj_all = obj_all[:m + 1]
                break

        if binomial is True:
            dset = binomial_split(img, n_splits=n_rep, p=0.5, device=device)
    return obj_all


def variance_reconstruction(dset, psf, lambda_value, alpha_value, weight_map=1, max_iter=50, tol=None, method='sgm',
                            regularizer='l2', initialization='flat', normalize=False, process='gpu',
                            rec_to_save='last', binomial: bool = False):
    """
    Full reconstruction using variance-regularized iterative update with automatic padding, GPU check, and flexible initialization.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K x Ch.
        psf (ndarray): PSF kernel of shape Mx x My x K x Ch.
        lambda_value (float): Variance regularization parameter.
        alpha_value (float): Fidelity term regularization parameter.
        weight_map (float): Regularization weight a scalar or tensor of shape Nx x Ny.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        method (str): Update method ('osl', 'abs', or 'sgm').
        regularizer (str): Regularization type to measure distance ('l2', 'kl', or 'l1').
        initialization (str): Initialization method for the object estimate ('flat' or 'data').
        normalize (bool): Whether to normalize the flux at each iteration.
        process (str): Processing mode ('cpu' or 'gpu').
        rec_to_save (str): 'last' to return last iteration, 'all' to return all iterations.

    Returns:
        Tensor: Reconstructed volume of shape Nx x Ny x K.
    """

    # Ensure tensors are on the correct device
    if process == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dset = torch.from_numpy(dset).float().to(device)
    psf = torch.from_numpy(psf).float().to(device)

    axis_to_sum = 2
    sz = dset.shape[:axis_to_sum]

    psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape

    # Normalize PSF
    psf_norm = psf_pad / psf_pad.sum()
    psf_m = torch.flip(psf_norm, dims=[-2, -1])

    # Precompute FFTs
    psf_fft = rfftn(psf_norm, dim=(-2, -1))
    psf_m_fft = rfftn(psf_m, dim=(-2, -1))

    obj_all = iterate(dset, psf_fft, psf_m_fft, max_iter, lambda_value, alpha_value, weight_map, tol, method, device,
                      regularizer, initialization, normalize, rec_to_save, binomial)
    if rec_to_save == 'last':
        return obj_all[-1]
    elif rec_to_save == 'all':
        return obj_all
    else:
        raise ValueError("rec_to_save must be 'last' or 'all'.")