import torch
import string
import einops

from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift

from .utils import compute_otf_from_psf, forward_model, partial_convolution_rfft, centered_pad, kl_divergence


def fidelity_loss(dataset, obj_set, psf, eps=None, weight_map: torch.tensor = 1, full_KL=False, k_loss=False):
    """
    Compute the fidelity loss between data and the forward value from predicted solution.

    Parameters:
    dataset (Tensor): Input data tensor with dimensions Nx x Ny or Nx x Ny x K.
    obj_set (Tensor): Predicted object tensor with dimensions num_iter x Nx x Ny or num_iter x Nx x Ny x K.
    psf (Tensor): Point spread function with dimensions Mx x My or Mx x My x K or the corresponding otf.
    full_KL (bool): Whether to use full KL divergence with normalization.

    Returns:
    Tensor: Computed fidelity loss.
    """
    
    if eps is None:
        eps = 1e-8

    psf = torch.as_tensor(psf)
    num_axis_to_conv = 2 
    
    if obj_set.shape[1:num_axis_to_conv+1] == dataset.shape[:num_axis_to_conv]: #If I do not have num_iter
        obj_set = obj_set.unsqueeze(0)  # Add a iter dimension
        
    if psf.is_floating_point():         # If I have a PSF, I need to compute its FFT
        num_axis_to_conv = 2
        sz = tuple(dataset.shape[:num_axis_to_conv])
        psf_fft = compute_otf_from_psf(psf, target_shape=sz)
    elif torch.is_complex(psf):
        psf_fft = psf
    else:
        raise ValueError(f"Unsupported PSF dtype: {psf.dtype}")
            
    if psf.is_floating_point():         # If I have a PSF, I need to compute its FFT
        axis_to_conv = tuple(torch.arange(1, num_axis_to_conv+1))
        sz = dataset.shape[:num_axis_to_conv]
        psf_pad = centered_pad(psf, target_shape=sz) # Centered padding of PSF to match dataset shape
        psf_fft = rfftn(psf_pad, dim=axis_to_conv) # FFT of the PSF
    elif torch.is_complex(psf):
        psf_fft = psf
    else:
        raise ValueError(f"Unsupported PSF dtype: {psf.dtype}")

    # alphabet = string.ascii_lowercase
    
    # str_psf = alphabet[1:psf_fft.ndim+1]   # 'bcd'
    # str_obj = alphabet[:num_axis_to_conv+1] # 'abc'
    # if len(str_obj) != obj_set.ndim:
    #     str_obj += alphabet[dataset.ndim]  # 'abc e'
        
    # axis_str = alphabet[1:num_axis_to_conv+1] # 'bc'
    # str_final = alphabet[:dataset.ndim + 1]  # 'abcde'

    # if dataset.ndim == num_axis_to_conv or dataset.ndim == num_axis_to_conv + 1: # No K and no Ch dimension or no K but Ch dimension
    #     axis_to_sum = tuple(range(1, dataset.ndim + 1))  # Sum over all but the first dimension (num_iter)
    # else:
    #     axis_to_sum = tuple(range(1, dataset.ndim))  # Sum over all but the first dimension (num_iter)
    # forward = partial_convolution_rfft(psf_fft, obj_set, dim1=str_psf, dim2=str_obj, dim3=str_final, axis=axis_str, fourier=(1,0))
    
    forward_obj = None
    for it in range(obj_set.shape[0]):
        forward_it = forward_model(psf_fft, obj_set[it]).sum(0) #sum over z
        if it == 0:
            forward_obj = forward_it
        else:
            forward_obj = torch.cat((forward_obj, forward_it), dim=0)
    
    fidelity = kl_divergence(dataset, forward_obj, full_KL=full_KL, normalize_entries = False, k_loss=k_loss)
    # fidelity = forward - dataset * torch.log(forward + eps)

    # if full_KL:
    #     fidelity = fidelity + dataset * torch.log( dataset + eps ) - dataset

    # idx = ~torch.isfinite(fidelity)
    # fidelity[ idx ] = 0

    if torch.is_tensor(weight_map):
        letters = ["x", "y", "z"][: weight_map.ndim]
        lhs = " ".join(letters)
        ein_str = f"b {lhs} ... , {lhs} -> b"
        fidelity = einops.einsum(fidelity, weight_map, ein_str)
    # else:
    #     fidelity = fidelity.sum(axis=axis_to_sum) # Sum over all but the first dimension (num_iter if present) and last (K)
        
    return fidelity.squeeze()


def regularization_loss(obj_set: torch.tensor, regularizer: str = 'l2', weight_map: torch.tensor = 1, otf = None,
                        iteration_axis: bool = False, eps: float = None, fourier: bool = False):
    """
    Compute the variance regularization term for a set of objects.

    Parameters:
    obj_set (Tensor): Input tensor of shape Nx x Ny x K.
    regularizer (str): Type of regularization ('l1', 'l2', 'kl').

    Returns:
    float: Variance regularization value.
    """

    if eps is None:
        eps = 1e-8

    if fourier is True:
        ft_dim = (1, 2) if iteration_axis is True else (0, 1) # to generalize to 3D later
        obj_set = rfftn(obj_set, norm = "forward", dim=ft_dim)
        # obj_set = fftshift(obj_set, dim=ft_dim) # useful for visualization but not for computation

    if iteration_axis:
        axis_to_sum = tuple(range(1, obj_set.ndim-1))
    else:
        axis_to_sum = tuple(range(0, obj_set.ndim-1))

    centroid = compute_centroid(obj_set, regularizer)

    if regularizer.lower() == 'l2':
        residuals = torch.abs(obj_set - centroid)**2
    elif regularizer.lower() == 'l1':
        residuals = torch.abs(obj_set - centroid)
    elif regularizer.lower() == 'kl':
        residuals = centroid * torch.log( (centroid + eps) / (obj_set + eps) )
    else:
        raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")

    residuals = residuals * weight_map

    if fourier is True and otf is not None:
        mtf = torch.abs(otf)
        mtf = mtf / mtf.max()
        mtf[mtf < 1e-2] = 1e-2
        weight = 1/mtf
        #print(weight.shape, residuals.shape)
        residuals = residuals * weight

    return residuals.sum(axis=axis_to_sum)


def compute_centroid(obj_set: torch.tensor, regularizer: str = 'l2'):
    """
    Compute the centroid of a set of objects based on the specified regularization.

    Parameters:
    obj_set (Tensor): Input tensor of shape Nx x Ny x K.
    regularizer (str): Type of regularization ('l1', 'l2', 'kl').

    Returns:
    Tensor: Computed centroid.
    """

    if regularizer.lower() == 'l2':
        centroid = obj_set.mean(dim=-1, keepdim=True)
    elif regularizer.lower() == 'l1':
        centroid = obj_set.median(dim=-1, keepdim=True)
    elif regularizer.lower() == 'kl':
        eps = 1e-2
        centroid = torch.exp( torch.log(obj_set + eps).mean(dim=-1, keepdim=True) ) / torch.e # Geometric mean scaled
    else:
        raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")

    return centroid