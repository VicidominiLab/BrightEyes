from tabnanny import verbose
import torch
import numpy as np

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift
from torch import real, einsum
from tqdm import tqdm

from .loss import fidelity_loss, regularization_loss

from .utils import partial_convolution_rfft, centered_pad, forward
from .regularizers import grad_l2, grad_l1, grad_kl
from .preprocessing import binomial_split

from . import initializers as init

def tensor_scalar_product(tensor1, tensor2, axis_to_sum=None):
    """
    Compute the scalar product between two tensors.

    Parameters:
    tensor1 (Tensor): First input tensor.
    tensor2 (Tensor): Second input tensor.

    Returns:
    float: Scalar product result.
    """
    if axis_to_sum is None:
        axis_to_sum = tuple(range(0, tensor1.ndim))
        
    return torch.sum(tensor1 * tensor2, dim=axis_to_sum)

def compute_step_size_line_search(img, obj, obj_tmp, psf_fft, grad, lam, alpha, D_matrix_inv, normalize=True, theta = 0.8, delta = 0.5, max_iter = 30, initial_step_size=None, verbose=False):
    
    
    min_step_size = 1e-10
    K=obj.shape[-1]
    
    
    if initial_step_size is None:
        initial_step_size = 1.0*torch.ones(K)*theta
        
    step_size = initial_step_size / ((theta**1))
    step_size = torch.clamp(step_size, max=1)
    
    # Track ks that ever satisfy the descent lemma
    satisfied_once = torch.zeros(K, dtype=torch.bool, device=obj.device)

    
    #Compute current loss
    fidelity = fidelity_loss(img, obj, psf_fft)
    regularization = regularization_loss(obj, iteration_axis=False)
    loss_value = alpha * fidelity + lam * regularization
    
    direction = obj_tmp - obj
    
    # Track which k have already fixed their step size
    frozen = torch.zeros(K, dtype=torch.bool)

    for i in range(max_iter):
        
        # Freeze ks whose step is below min threshold
        below_thr = (step_size < min_step_size)
        if below_thr.any():
            step_size = torch.where(below_thr, min_step_size, step_size)
            frozen = frozen | below_thr  # freeze these ks
            
        # If all ks are frozen, stop early
        if frozen.all():
            break
            
        # Only update ks where frozen[k] == False
        active_step = step_size.clone()
        active_step[frozen] = 0.0    # unused but keeps shapes consistent
        
        #Compute next point
        obj_new = obj + active_step * direction
        
        # flux conservation
        if normalize is True:
            # Axis to sum for flux conservation
            axis_to_sum = tuple(range(img.ndim - 1))
            flux = img.sum(axis_to_sum, keepdim=True).squeeze(0)
            # Flux conservation
            obj_axis_to_sum = tuple(range(0, obj.ndim - 1))
            obj_new = obj_new * flux / obj_new.sum(obj_axis_to_sum, keepdim=True)
        
        #Compute new loss
        fidelity_new = fidelity_loss(img, obj_new, psf_fft)
        regularization_new = regularization_loss(obj_new, iteration_axis=False)
        loss_value_new = alpha * fidelity_new + lam * regularization_new
                
        #Check descent lemma
        diff = obj_new - obj
        lhs = loss_value_new - loss_value - tensor_scalar_product(diff, grad, axis_to_sum=tuple(range(0, diff.ndim-1)))
        rhs = (delta / step_size) * tensor_scalar_product(diff, D_matrix_inv**(-1) * diff, axis_to_sum=tuple(range(0, diff.ndim-1)))
        if verbose:
            print(f"Line search iteration {i} mean values: lhs = {lhs.mean()}, rhs = {rhs.mean()}, step size = {step_size.mean()}")
        
        # Which ks satisfy descent lemma?
        satisfied = lhs <= rhs
        
        # Mark ks that satisfied at least once
        satisfied_once |= satisfied
        
        # Only freeze ks that are not frozen yet
        newly_frozen = (~frozen) & satisfied
        frozen = frozen | newly_frozen

        # For ks that are not frozen yet, shrink η_k
        step_size = torch.where(frozen, step_size, step_size * theta)
    
    step_size = torch.clamp(step_size, max=1)
    
    # ks that should actually update obj:
    should_update = satisfied_once.clone()
    
    # if verbose:
    #     print(f"Step sizes: {step_size}")
    return step_size, should_update

def update_fft(img, obj, psf_fft, psf_m_fft, lam, alpha, weight, eps, m, method: str, regularizer: str, normalize: bool,  step_size_search:str, initial_step_size: float = 1.0, line_search_params: tuple=(10,2), verbose=False):
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

    update_k = alpha * forward(psf_m_fft, fraction).sum(-2)  # Sum over channels
    del fraction

    step_size = 1.0
    
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

    # === Metodo GD con step size ===
    elif method.lower() == 'pgd':
        obj_mean = obj.mean(dim=-1, keepdim=True)
        grad = (alpha * 1 - update_k + 2 * lam * weight * (obj - obj_mean))
        obj_tmp = torch.where(obj - grad > 0, obj - grad, 0)
        print(f'min obj_tmp: {obj_tmp.min()}, max obj_tmp: {obj_tmp.max()}')
        print(f'grad min: {grad.min()}, grad max: {grad.max()}')
        ### TO DO: Controllare che qua non ci voglia un check obj = obj_tmp per decretare punto fisso vd Bonettini paper ###
        
        if step_size_search == 'rl':
            step_size = obj
        elif step_size_search == 'fixed':
            step_size = initial_step_size    
        elif step_size_search == 'line_search':
            step_size, should_update = compute_step_size_line_search(img, obj, obj_tmp, psf_fft, grad, lam, alpha, D_matrix_inv=1, normalize=normalize, initial_step_size=initial_step_size, verbose=verbose)
            #step_size = step_size[None, None, :]
        else:
            raise ValueError("step_size_search must be 'rl', 'line_search' or 'fixed'.")
            
        obj_new = obj + step_size * (obj_tmp - obj)

    # === Metodo SGP ===
    elif method.lower() == 'sgp':
        should_update = torch.ones(obj.shape[-1], dtype=torch.bool, device=obj.device)
        step_size = initial_step_size
        obj_mean = obj.mean(dim=-1, keepdim=True)
        grad = (alpha * 1 - update_k + 2 * lam * weight * (obj - obj_mean))
        #grad_thr = torch.where( abs(grad) < eps, eps, grad)
        D_matrix_inv = obj/(alpha*1+2*lam*weight*obj)
        s1 = line_search_params[0]
        s2 = line_search_params[1]
        thr = np.sqrt( 1 + (s1/( (m+1)**s2 ) ) )
        D_matrix_inv = torch.clamp(D_matrix_inv, min=1/thr, max=thr)
        
        obj_tmp = torch.where( (obj - D_matrix_inv * grad) > 0, (obj - D_matrix_inv * grad), 0)
        ### TO DO: Controllare che qua non ci voglia un check obj = obj_tmp per decretare punto fisso vd Bonettini paper ###
        
        if step_size_search == 'rl':
            step_size = obj
        elif step_size_search == 'fixed':
            step_size = initial_step_size    
        elif step_size_search == 'line_search':
            step_size, should_update = compute_step_size_line_search(img, obj, obj_tmp, psf_fft, grad, lam, alpha, D_matrix_inv, normalize=normalize, initial_step_size=initial_step_size, verbose=verbose)
            #step_size = step_size[None, None, :]
        else:
            raise ValueError("step_size_search must be 'rl', 'line_search' or 'fixed'.")

        #obj_new = obj + step_size * (obj_tmp - obj)
        obj_new = obj.clone()
        obj_new[..., should_update] = (obj[..., should_update] +
                    step_size[should_update] * (obj_tmp[..., should_update] - obj[..., should_update]) )   
    else:
        raise ValueError("Metodo non valido. Usare 'osl', 'sgm', 'sgp' o 'gd'.")
    
    # flux conservation
    if normalize is True:
        # Axis to sum for flux conservation
        axis_to_sum = tuple(range(img.ndim - 1))
        flux = img.sum(axis_to_sum, keepdim=True).squeeze(0)
        # Flux conservation
        obj_axis_to_sum = tuple(range(0, obj.ndim - 1))
        obj_new = obj_new * flux / obj_new.sum(obj_axis_to_sum, keepdim=True)

    return obj_new, step_size


def iterate(dset, psf, max_iter, lam, alpha, weight, tol, method, device, regularizer, initialization,
            normalize, rec_to_save, binomial, step_size_search, line_search_params, verbose=False):
    """
    Performs a single iteration update for the reconstruction process.

    Parameters:
        dset (Tensor): Input data of shape Nx x Ny x Ch x K.
        psf (Tensor): Fourier transform of the PSF of shape Nx x Ny x Ch.
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

    eps = 1e-8  # torch.finfo(torch.float32).eps
    
    conv_dims = 2  # to generalize to 3D later
    if dset.ndim != psf.ndim + 1:
        raise ValueError(f"The dataset must be one dimension larger than the psf. Got dataset with {dset.ndim} dimension and psf with {psf.ndim}.")

    # axis_not_to_conv = tuple(range(-(psf.ndim - conv_dims), 0))
    # print(f'axis_not_to_conv: {axis_not_to_conv}')
    axis_to_conv = tuple(range(conv_dims))

    # Normalize PSF
    psf_norm = psf / psf.sum()

    # Precompute FFTs
    psf_fft = rfftn(psf_norm, dim=axis_to_conv)
    psf_m_fft = torch.conj(psf_fft)

    if binomial is True:
        img = dset.sum(dim=-1)
        n_rep = dset.shape[-1]

    # Initialization
    if dset.ndim - conv_dims == 1:
        size = list(dset.shape)
    elif dset.ndim - conv_dims == 2:
        size = list(dset.shape[:-2] + dset.shape[-1:])
    else:
        raise ValueError("Dataset must have repetition and eventually channel dimensions in addition to spatial dimensions.")
        
    if rec_to_save == 'last':
        size = [1] + size
    elif rec_to_save == 'all':
        size = [max_iter + 1] + size

    obj_all = torch.empty(size).to(device)
    
    obj_all[0] = init.initialize(dset, psf_norm, method = initialization, device = None)

    #Control check parameter stepsize
    if isinstance(step_size_search, (float, int)):
        if step_size_search <= 0:
            raise ValueError("step_size_search must be a positive number.")
        initial_step_size = step_size_search
        step_size_search = 'fixed'
    else:
        initial_step_size = 1*torch.ones(obj_all.shape[-1]).to(device)

    idx = 0
    step_size = []
    # Reconstruction loop
    for m in tqdm(range(max_iter), desc=method):
        obj_set, initial_step_size = update_fft(dset, obj_all[idx], psf_fft, psf_m_fft, lam, alpha, weight, eps, m, method, regularizer, normalize,
                                                step_size_search=step_size_search, initial_step_size=initial_step_size, line_search_params=line_search_params, verbose=verbose)

        step_size.append(initial_step_size)
        
        if rec_to_save == 'all':
            idx = m + 1
        elif rec_to_save == 'last':
            idx = -1
        else:
            raise ValueError("rec_to_save must be 'last' or 'all'.")
        obj_all[idx] = obj_set

        if tol is not None:
            rel_diff = ((obj_all[m + 1] - obj_all[m])**2).sum() / ((obj_all[m])**2).sum()
            if rel_diff.item() < tol:
                obj_all = obj_all[:m + 1]
                break

        if binomial is True:
            dset = binomial_split(img, n_splits=n_rep, p=0.5, device=device)
            
    return obj_all, step_size


def variance_reconstruction(dset, psf, lambda_value, alpha_value=1, weight_map=1, max_iter=50, tol=None, method='sgm',
                            regularizer='l2', initialization='flat', normalize=False, process='gpu',
                            rec_to_save='last', binomial: bool = False, step_size_search: str = 'fixed', line_search_params=None, verbose=False):
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
        
    if verbose: 
        print('Computing reconstruction on device:', device)

    dset = torch.from_numpy(dset).float().to(device)
    psf = torch.from_numpy(psf).float().to(device)

    conv_dims = 2
    
    if dset.ndim - conv_dims == 1: #if I do not have channels
        dset = dset.unsqueeze(-2)  # Add a channel dimension #Nx x Ny x 1 x K
        psf = psf.unsqueeze(-1)  # Add a channel dimension #Mx x My x 1
    
    sz = dset.shape[:conv_dims]
    
    psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape
        
    obj_all, step_size = iterate(dset, psf_pad, max_iter, lambda_value, alpha_value, weight_map, tol, method, device,
                      regularizer, initialization, normalize, rec_to_save, binomial, step_size_search, line_search_params, verbose)
    
    if rec_to_save == 'last':
        return obj_all[-1]
    
    elif rec_to_save == 'all':
        return obj_all, step_size
    else:
        raise ValueError("rec_to_save must be 'last' or 'all'.")