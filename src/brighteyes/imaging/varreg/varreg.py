import torch
import string
import numpy as np
import torch.optim as optim

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift
from torch import real, einsum
from tqdm import tqdm

from .utils import partial_convolution_rfft, centered_pad
from .loss import fidelity_loss, regularization_loss

from .regularizers import grad_l2, grad_l1, grad_kl


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

    conv_dims = 2 # to generalize to 3D later

    alphabet = string.ascii_lowercase

    str_psf = alphabet[:otf.ndim]
    str_obj = alphabet[:obj.ndim]
    axis_str = alphabet[:conv_dims]

    dset = partial_convolution_rfft(otf, obj, dim1=str_psf, dim2=str_obj, axis=axis_str, fourier=(1, 0))

    return dset


def update_fft(img, obj, psf_fft, psf_m_fft, lam, weight, eps, method: str, regularizer: str):
    """
    Performs a single iteration update for SGM with variance regularization.

    Parameters:
        img (Tensor): Input data of shape Nx x Ny x K.
        obj (Tensor): Current estimate of the object of shape Nx x Ny x K.
        psf_fft (Tensor): Fourier transform of the PSF.
        psf_m_fft (Tensor): Fourier transform of the flipped PSF.
        lam (float): Variance regularization parameter.
        weight (float): Regularization weight.
        method (str): 'osl' or 'abs' or 'sgm'.
        eps (float): Epsilon to avoid division by zero.

    Returns:
        Tensor: Updated object tensor of shape Nx x Ny x K.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # === forward convolution ===
    img_estimate_k = forward(psf_fft, obj)
    # img_estimate_k = partial_convolution_rfft(psf_fft, obj, dim1=str_psf, dim2=str_first, axis=axis_str, fourier=(1, 0))
    fraction = img / (img_estimate_k + eps)

    del img_estimate_k

    update_k = forward(psf_m_fft, fraction)

    # update_k = partial_convolution_rfft(psf_m_fft, fraction, dim1=str_psf, dim2=str_second, axis=axis_str, fourier=(1, 0)

    del fraction

    # === Metodo OSL ===
    if method.lower() == 'osl':
        obj_mean = obj.mean(dim=-1, keepdim=True)

        denom = 1 + 2 * lam * weight * (obj - obj_mean)
        obj_update = update_k / (denom + eps)
        obj_new = obj * obj_update

    # === Metodo ABS ===
    elif method.lower() == 'abs':
        obj_mean = obj.mean(dim=-1, keepdim=True)

        denom = 1 + 2 * lam * weight * torch.abs(obj - obj_mean)
        obj_update = update_k / (denom + eps)
        obj_new = obj * obj_update

    # === Metodo SGM ===
    elif method.lower() == 'sgm':

        if regularizer == 'variance':
            u, v = grad_l2(obj)
        elif regularizer == 'kl':
            u, v = grad_kl(obj)
        elif regularizer == 'l1':
            u, v = grad_l1(obj)
        else:
            raise ValueError("Regularizer must be 'variance', 'kl', or 'l1'.")

        update_k = update_k + lam * weight * u

        denom = 1 + lam * weight * v
        obj_update = update_k / denom
        obj_new = obj * obj_update

    else:
        raise ValueError("Metodo non valido. Usare 'osl' o 'sgm'.")

    return obj_new


def iterate(dset, psf_fft, psf_m_fft, max_iter, lam, weight, tol, method, device, regularizer, normalize):
    eps = torch.finfo(torch.float32).eps

    # Axis to sum for flux conservation
    axis_to_sum = tuple(range(dset.ndim - 1))
    flux = dset.sum(axis_to_sum, keepdim=True)

    # Initialization
    size = [max_iter + 1] + list(dset.shape)
    obj_all = torch.empty(size).to(device)
    obj_all[0] = torch.ones(dset.shape).to(device) * flux / np.prod(dset.shape[:-1])

    # Reconstruction loop
    for m in tqdm(range(max_iter), desc=method):
        obj_all[m + 1] = update_fft(dset, obj_all[m], psf_fft, psf_m_fft, lam, weight, eps, method, regularizer)

        #flux conservation
        if normalize is True:
            obj_all[m + 1] = obj_all[m + 1] * flux / obj_all[m + 1].sum(axis_to_sum, keepdim = True)

        if tol is not None:
            rel_diff = torch.norm(obj_all[m + 1] - obj_all[m]) / torch.norm(obj_all[m])
            if rel_diff.item() < tol:
                print(f"Converged at iteration {m + 1}")
                obj_all = obj_all[:m + 1]
                break

    return obj_all


def variance_reconstruction(dset, psf, lam, W=1, max_iter=50, tol=None, rec_to_save='last', method='osl', process='gpu',
                            regularizer='variance', normalize=False):
    """
    Full reconstruction using variance-regularized iterative update with automatic padding, GPU check, and flexible initialization.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        psf (ndarray): PSF kernel.
        W (float): Regularization weight.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
        process (str): Processing mode ('cpu' or 'gpu').
        regularizer (str): Regularization type ('variance', 'kl', or 'l1').
        method (str): Update method ('osl', 'abs', or 'sgm').
        rec_to_save (str): 'last' to return last iteration, 'all' to return all iterations.
        normalize (bool): Whether to normalize the flux at each iteration.

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

    ny, nx, nrep = dset.shape

    psf_pad = centered_pad(psf, target_shape=(ny, nx))  # Centered padding of PSF to match dataset shape

    # Normalize PSF
    psf_norm = psf_pad / psf_pad.sum()
    psf_m = torch.flip(psf_norm, dims=[-2, -1])

    # Precompute FFTs
    psf_fft = rfftn(psf_norm, dim=(-2, -1))
    psf_m_fft = rfftn(psf_m, dim=(-2, -1))

    obj_all = iterate(dset, psf_fft, psf_m_fft, max_iter, lam, W, tol, method, device, regularizer, normalize)

    if rec_to_save == 'last':
        return obj_all[-1]
    elif rec_to_save == 'all':
        return obj_all
    else:
        raise ValueError("rec_to_save must be 'last' or 'all'.")


def Adam_optimize_obj(dset, psf, num_iter=500, lambda_value=0.01, alpha_value = 0.01, initialization : str = 'flat',
                     regularizer='l2', lr=1e-3,  process = 'gpu', full_KL=False, verbose=True):
    """
    Ottimizza obj_set minimizzando fidelity_loss + reg_weight * regularization_loss con Adam.

    Parameters:
        dataset (Tensor): Tensore dei dati di dimensione (Nx, Ny) o (Nx, Ny, K).
        psf (Tensor): Point Spread Function, dimensione (Mx, My) o (Mx, My, K).
        regularizer (str): Tipo di regolarizzazione ('l1', 'l2', 'kl').
        lr (float): Learning rate di Adam.
        num_iter (int): Numero di iterazioni.
        full_KL (bool): Se True, usa la forma completa della KL.
        verbose (bool): Se True, stampa i progressi.

    Returns:
        obj_set (Tensor): Tensore ottimizzato di dimensioni (Nx, Ny, K).
    """

    # Ensure tensors are on the correct device
    if process == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dset = torch.from_numpy(dset).float().to(device)
    psf = torch.from_numpy(psf).float().to(device)
    
    # Axis to sum for flux conservation
    axis_to_sum = tuple(range(dset.ndim - 1))
    flux = dset.sum(axis_to_sum, keepdim=True)

    # Initialization
    size = [num_iter + 1] + list(dset.shape)
    obj_all = torch.empty(size).to(device)

    if initialization == 'flat':
        obj_all[0] = torch.ones(dset.shape).to(device) * flux / np.prod(dset.shape[:-1])
    elif initialization == 'data':
        axis_to_conv = tuple(range(2))
        obj_all[0] = torch.tensor(gaussian_filter(dset.numpy(), sigma=1, mode='constant', cval=0, truncate=4, axes = axis_to_conv)).to(device)
        obj_all[0] = obj_all[0] * flux / obj_all[0].sum()
    else:
        raise ValueError("Initialization must be 'flat' or 'data'.")

    obj_set = obj_all[0]
    obj_set.requires_grad = True

    # Dictionary with losses
    losses = {
        'total': [],
        'fidelity': [],
        'regularization': []
    }


    optimizer = optim.Adam([obj_set], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    for it in range(num_iter):
        optimizer.zero_grad()

        # Calcolo dei termini di perdita
        loss_fidelity = alpha_value * fidelity_loss(dset, obj_set, psf, full_KL=full_KL)
        loss_reg = lambda_value * regularization_loss(obj_set, regularizer=regularizer)

        loss = loss_fidelity + loss_reg

        # Backprop + update
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        # Imporre positività (opzionale per KL)
        with torch.no_grad():
            obj_set.clamp_(min=1e-8)
            
            
        # Save solution
        obj_all[it+1] = obj_set.detach()
        # Save losses
        losses['total'].append(loss.item())
        losses['fidelity'].append(loss_fidelity.item())
        losses['regularization'].append(loss_reg.item())

        if verbose and (it % max(1, num_iter // 10) == 0 or it == num_iter - 1):
            print(f"Iter {it+1}/{num_iter} | Loss: {loss.item():.6f} "
                  f"| Fidelity: {loss_fidelity.item():.9f} | Reg: {loss_reg.item():.9f}"
                  f" | LR: {optimizer.param_groups[0]['lr']:.6f}")

    return obj_all.detach(), losses
