from tabnanny import verbose
import torch
import numpy as np

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift
from torch import real, einsum
from tqdm import tqdm

from .loss import fidelity_loss, regularization_loss

from .utils import forward_model, centered_pad
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
    device = obj.device    
    
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
    frozen = torch.zeros(K, dtype=torch.bool).to(device)

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

def update_fft(img, obj, psf_fft, psf_m_fft, lam, alpha, weight, eps, m,
               method: str, regularizer: str, normalize: bool, step_size_search:str,
               initial_step_size: float = 1.0, line_search_params: tuple=(10,2), verbose=False):
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
    img_estimate_k = forward_model(psf_fft, obj).sum(0)
    fraction = img / (img_estimate_k + eps)

    del img_estimate_k

    update_k = alpha * forward_model(psf_m_fft, fraction).sum(-2)  # Sum over channels
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


def iterate(dset, psf, max_iter, lam, alpha, weight, stop_tol, method, device, regularizer_norm, initialization,
            normalize, stop_criteria, rec_to_save, step_size_search, line_search_params, return_pointwise_sol=False, verbose=False):
    """
    Performs a single iteration update for the reconstruction process.

    Parameters:
        dset (Tensor): Input data of shape Nx x Ny x K x Ch.
        psf (Tensor): Fourier transform of the PSF of shape Nz x Nx x Ny x Ch.
        max_iter (int): Maximum number of iterations.
        lam (float): Variance regularization parameter.
        alpha (float): Fidelity term regularization parameter.
        weight (float): Regularization weight a scalar or tensor of shape Nx x Ny.
        stop_tol (float): Convergence tolerance to stop iterations.
        stop_criteria (str): Criteria for stopping the iterations ('rel_diff' or 'step_size').
        method (str): algorithm 'osl' or 'abs' or 'sgm'.
        device (str): device to perform computation on ('cpu' or 'cuda').
        regularizer_norm (str): distance to measure the differences between o_k and o_mean: 'l2', 'kl', or 'l1'.
        initialization (str): initialization method for the object estimate ('flat' or 'data').
        normalize (bool): whether to normalize the object estimate at each iteration.

    Returns:
        Tensor: Updated object tensor of shape Nz x Nx x Ny x K.
    """

    eps = 1e-8  # torch.finfo(torch.float32).eps

    # Normalize PSF
    num_axis_to_conv = 2
    axis_to_conv = tuple(np.arange(1, num_axis_to_conv+1))
    norm_axis = tuple(np.arange(1, psf.ndim))
    psf_norm = psf / psf.sum(axis=norm_axis, keepdim=True)

    # Precompute FFTs
    psf_fft = rfftn(psf_norm, dim=axis_to_conv)
    psf_m_fft = torch.conj(psf_fft)

    # Initialization
    if return_pointwise_sol:
        size = list( psf.shape[:-1] ) # Nz x Nx x Ny
    else:
        size = list( tuple(psf.shape[:-1]) + (dset.shape[-1],) ) # Nz x Nx x Ny x K
        
    if rec_to_save == 'last':
        size = [2] + size
    elif rec_to_save == 'all':
        size = [max_iter + 1] + size
    
    # obj_all = torch.empty(size).to(device)  # max_iter + 1 x Nz x Nx x Ny x K
    # obj_set = init.initialize(dset, psf_norm, method = initialization, device = device)
    # if return_pointwise_sol:
    #     obj_all[0] = obj_set.sum(-1) #sum over K
    # else:
    #     obj_all[0] = obj_set
    
    history_device = torch.device('cpu') if rec_to_save == 'all' else device
    obj_all = torch.empty(size, device=history_device)  # keep saved iterates off-GPU when storing the full history
    obj_set = init.initialize(dset, psf_norm, method = initialization, device = device)
    if return_pointwise_sol:
        obj_all[0] = obj_set.sum(-1).detach().to(history_device) #sum over K
    else:
        obj_all[0] = obj_set.detach().to(history_device)
        
    #Control check parameter stepsize
    if isinstance(step_size_search, (float, int)):
        if step_size_search <= 0:
            raise ValueError("step_size_search must be a positive number.")
        initial_step_size = step_size_search
        step_size_search = 'fixed'
    else:
        initial_step_size = 1*torch.ones(obj_set.shape[-1]).to(device)

    idx = 0
    step_size = []
    # Reconstruction loop
    for m in tqdm(range(max_iter), desc=method):
        previous_obj_set = obj_set.clone()
        obj_set, initial_step_size = update_fft(dset, previous_obj_set, psf_fft, psf_m_fft, lam, alpha, weight, eps, m, method, regularizer_norm, normalize, 
                                                step_size_search=step_size_search, initial_step_size=initial_step_size, line_search_params=line_search_params, verbose=verbose)

        step_size.append(initial_step_size)
        
        if rec_to_save == 'all':
            idx = m + 1
            previous_idx = m
        elif rec_to_save == 'last':
            idx = 1
            previous_idx = 0
            obj_all[previous_idx] = obj_all[idx]    # copy last estimate to the first position to save memory
        else:
            raise ValueError("rec_to_save must be 'last' or 'all'.")
        
        if return_pointwise_sol:
            obj_all[idx] = obj_set.sum(-1).detach().to(history_device)
        else:
            obj_all[idx] = obj_set.detach().to(history_device)

        if stop_tol is not None:
            if stop_criteria == 'rel_diff':
                criteria = ( abs(obj_set[0] - previous_obj_set[0]) / (previous_obj_set[0] + eps) ).sum()
            elif stop_criteria == 'step_size':
                criteria = step_size[previous_idx].max()
            else:
                raise ValueError("stop_criteria must be 'rel_diff' or 'step_size'.")
            
            if criteria < stop_tol:
                obj_all = obj_all[:idx]
                break
            
    return obj_all, step_size


def variance_reconstruction(dset, psf, lambda_value, alpha_value=1, weight_map=1, max_iter=100, stop_tol=None, method='sgp',
                            regularizer_norm='l2', initialization='flat', normalize=True, process='gpu', stop_criteria='rel_diff',
                            rec_to_save='last', return_pointwise_sol=False, step_size_search: str = 'fixed', line_search_params=None, verbose=False):
    """
    Full reconstruction using variance-regularized iterative update with automatic padding, GPU check, and flexible initialization.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Nyx Ch x K.
        psf (ndarray): PSF kernel of shape Mz x Mx x My x Ch x K.
        lambda_value (float): Variance regularization parameter.
        alpha_value (float): Fidelity term regularization parameter.
        weight_map (float): Regularization weight a scalar or tensor of shape Nx x Ny.
        max_iter (int): Maximum number of iterations.
        stop_tol (float): Convergence tolerance.
        method (str): Update method ('osl', 'abs', or 'sgp').
        regularizer_norm (str): Regularization type to measure distance ('l2', 'kl', or 'l1').
        initialization (str): Initialization method for the object estimate ('flat' or 'data').
        normalize (bool): Whether to normalize the flux at each iteration.
        process (str): Processing mode ('cpu' or 'gpu').
        stop_criteria (str): Criteria for stopping the iterations ('rel_diff' or 'step_size').
        rec_to_save (str): 'last' to return last iteration, 'all' to return all iterations.
        step_size_search (str): Step size search method ('fixed' or 'adaptive').
        line_search_params (dict): Parameters for line search.
        verbose (bool): Whether to print verbose output.
    Returns:
        Tensor: Reconstructed volume of shape Nz x Nx x Ny x K.
    """

    # Ensure tensors are on the correct device
    if process == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        
    if verbose: 
        print('Computing reconstruction on device:', device)

    dset = torch.from_numpy(dset).float().to(device)
    psf = torch.from_numpy(psf).float().to(device)
    
    # Ensure correct shapes of PSF w.r.t. dataset
    if dset.ndim <= 2:
        raise Exception('The dataset should have at least 3 dimension: Nx x Ny x K, where K are noise repetitions')
    
    if psf.ndim == dset.ndim -1: # PSF has not Nz dim
        psf = psf.unsqueeze(0)  # Add a z dimension #1 x Mx x My (x Ch)
        
    if psf.ndim == 3:
        psf = psf.unsqueeze(-1)  # Add a channel dimension at the end # Mz x Mx x My x 1 
    if dset.ndim == 3:
        dset = dset.unsqueeze(-2)  # Add a channel dimension at the end # Nx x Ny x 1 x K

    if np.ndim(weight_map) == 2:
        weight_map = torch.tensor(weight_map).float().to(device)
        weight_map = weight_map.unsqueeze(0).unsqueeze(-1)  # Add z and K dimensions #1 x Nx x Ny x 1
    elif np.ndim(weight_map) == 3:
        weight_map = torch.tensor(weight_map).float().to(device)
        weight_map = weight_map.unsqueeze(0)  # Add z dimension #1 x Nx x Ny x K
    
    spatial_dims = 3
    
    # Ensure dataset has odd numbers of pixels for convolution
    oddeven_check_x = dset.shape[0] % 2
    oddeven_check_y = dset.shape[1] % 2

    data_check = dset
    if oddeven_check_x == 0:
        data_check = data_check[1:]
    if oddeven_check_y == 0:
        data_check = data_check[:, 1:]
    
    # Pad PSF to match dataset shape
    sz = dset.shape[:spatial_dims]
    psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape
        
    # Run reconstruction
    obj_all, step_size = iterate(dset, psf_pad, max_iter, lambda_value, alpha_value, weight_map, stop_tol, method, device, regularizer_norm,
                            initialization, normalize, stop_criteria, rec_to_save, step_size_search, line_search_params, return_pointwise_sol, verbose)
    
    # if rec_to_save == 'last':
    #     return obj_all[-1]
    
    # elif rec_to_save == 'all':
    #     return obj_all, step_size
    # else:
    #     raise ValueError("rec_to_save must be 'last' or 'all'.")
    return obj_all, step_size
    
    
def batch_reconstruction(dset: np.ndarray, psf: np.ndarray, batch_size: list, overlap: int,
                         lambda_value, alpha_value=1, weight_map=1, max_iter=100, stop_tol=None, method='sgp',
                            regularizer_norm='l2', initialization='flat', normalize=True, process='gpu', 
                            rec_to_save='last', step_size_search: str = 'fixed', line_search_params=None, verbose=False):
    """

    Parameters
    ----------
    dset : np.ndarray
        ISM dataset shaped as (Nx x Ny x Nch x K).
    psf : np.ndarray
        ISM PSF shaped as (Nz x Nx x Ny x Nch).
    batch_size : list
        List containing the size of the batch in the x and y dimensions.
    overlap : int
         Overlap value between the batches.
    stop: string, optional
           String describing how to stop the algorithm. The default is 'auto'. If set to 'auto' the algorithm will stop
            when the derivative of the photon counts reaches the threshold, if set to 'fixed' the algorithm will stop when
            the maximum number of iterations is reached.
    max_iter: int, optional
             maximum number of iteration. The default is 100.
    threshold: float, optional
              The default is 1e-3. This value is needed when we choose stop = 'auto'. When the derivative of the photon counts
              reaches this value the algorithm halt.
    rep_to_save: iter, optional
                  iterable containing the iteration at which one wants to save the algorithm reconstruction.The default is 'all'.
    initialization : string, optional
                      The default is 'flat'. If set to 'flat' the algorithm will initialize the first iteration with a flat object,
                     if set to 'sum' the algorithm will initialize the first iteration with the ISM dataset.
    process: string, optional
               The default is 'gpu'. If set to 'gpu' the algorithm will run on the GPU if available, if set to 'cpu' the algorithm
                will run on the CPU.

    Returns
    -------
    reconstruction : np.ndarray ( Nz x Nx x Ny x Nch x K)
                     reconstructed object.
    """

    wx, wy = batch_size
    reconstruction = np.zeros((max_iter+1,) + (psf.shape[0],) + dset.shape[:-2] + (dset.shape[-1],))    # Initialize reconstruction array with shape (Nz x Nx x Ny x x K)

    nx = dset.shape[0] // (wx - overlap) + int(dset.shape[0] % (wx - overlap) > 0)
    ny = dset.shape[1] // (wy - overlap) + int(dset.shape[1] % (wy - overlap) > 0)

    k_iter = 1
    n_batch = nx * ny

    rx = np.mod(dset.shape[0], wx - overlap)
    ry = np.mod(dset.shape[1], wy - overlap)
    thx = psf.shape[1]
    thy = psf.shape[2]

    if np.logical_and(rx > 0, rx < thx) or np.logical_and(ry > 0, ry < thy):
        raise Exception('The PSF is bigger than the last batch size.')
    
    for i in range(nx):
        for j in range(ny):
            print(f'Batch {k_iter}/{n_batch}')

            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            overlap_x = overlap if i > 0 else 0
            overlap_y = overlap if j > 0 else 0

            indx = np.s_[i * (wx - overlap_x): (i + 1) * wx - i * overlap_x]
            indy = np.s_[j * (wy - overlap_y): (j + 1) * wy - j * overlap_y]

            crop_dset = dset[indx, indy]

            recon, *_ = variance_reconstruction(crop_dset, psf, lambda_value, alpha_value, weight_map, max_iter, stop_tol, method,
                            regularizer_norm, initialization, normalize, process, 
                            rec_to_save, step_size_search, line_search_params, verbose)

            recon = recon.cpu().numpy() if isinstance(recon, torch.Tensor) else recon

            indx = np.s_[i * (wx - overlap_x) + overlap_x // 2: (i + 1) * wx - i * overlap_x]
            indy = np.s_[j * (wy - overlap_y) + overlap_y // 2: (j + 1) * wy - j * overlap_y]

            reconstruction[:, :, indx, indy] = recon[:, :, overlap_x // 2:, overlap_y // 2:]
            k_iter += 1

    return reconstruction