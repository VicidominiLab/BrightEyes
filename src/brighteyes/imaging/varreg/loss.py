import torch
import string

from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift

from .utils import calculate_weight_map

from .utils import partial_convolution_rfft, centered_pad

def fidelity_loss(dataset, obj_set, psf, full_KL=False):
    """
    Compute the fidelity loss between data and the forward value from predicted solution.

    Parameters:
    dataset (Tensor): Input data tensor with dimensions Nx x Ny or Nx x Ny x K.
    obj_set (Tensor): Predicted object tensor with dimensions num_iter x Nx x Ny or num_iter x Nx x Ny x K.
    psf (Tensor): Point spread function with dimensions Mx x My or Mx x My x K.
    full_KL (bool): Whether to use full KL divergence with normalization.

    Returns:
    Tensor: Computed fidelity loss.
    """

    alphabet = string.ascii_lowercase
    n_dim = obj_set.ndim      # 3 or 4
    num_axis_to_conv = 2  
    

    if obj_set.shape[:num_axis_to_conv] == dataset.shape[:num_axis_to_conv]: #If I do not have num_iter
        obj_set = obj_set.unsqueeze(0)  # Add a batch dimension
        n_dim = n_dim + 1
        # str_psf = alphabet[:n_dim]   # 'abc' or 'abcd'
        # str_obj = alphabet[:n_dim]   # 'abc' or 'abcd'
        # axis_str = alphabet[:num_axis_to_conv] # 'ab'
        # axis_to_sum = tuple(range(obj_set.ndim))
    # else:
    #     str_psf = alphabet[1:n_dim]   # 'bc' or 'bcd'
    #     str_obj = alphabet[:n_dim]   # 'abc' or 'abcd'
    #     axis_str = alphabet[1:1+num_axis_to_conv] # 'bc'
    #     str_final = str_obj
    #     axis_to_sum = tuple(range(1, obj_set.ndim))
        
    str_psf = alphabet[1:n_dim]   # 'bc' or 'bcd'
    str_obj = alphabet[:n_dim]   # 'abc' or 'abcd'
    axis_str = alphabet[1:1+num_axis_to_conv] # 'bc'
    str_final = str_obj
    axis_to_sum = tuple(range(1, obj_set.ndim))

    sz = dataset.shape[:num_axis_to_conv]

    psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape
    
    psf_fft = rfftn(psf_pad, dim=tuple(range(num_axis_to_conv)))  # FFT of the PSF

    forward = partial_convolution_rfft(psf_fft, obj_set, dim1=str_psf, dim2=str_obj, dim3=str_final, axis=axis_str, fourier=(1,0))


    fidelity = forward - dataset * torch.log( forward )
    if full_KL:
        fidelity = fidelity + dataset * torch.log( dataset ) - dataset

    idx = ~torch.isfinite(fidelity)
    fidelity[ idx ] = 0


    return fidelity.sum(axis=axis_to_sum)  # Sum over all but the first dimension (num_iter if present)


def regularization_loss(obj_set: torch.tensor, dset: torch.tensor = None, regularizer: str = 'l2', weight: str = 'id',
                        iteration_axis: bool = False):
    """
    Compute the variance regularization term for a set of objects.

    Parameters:
    obj_set (Tensor): Input tensor of shape Nx x Ny x K.
    regularizer (str): Type of regularization ('l1', 'l2', 'kl').

    Returns:
    float: Variance regularization value.
    """

    if iteration_axis:
        axis_to_sum = tuple(range(1, obj_set.ndim))
    else:
        axis_to_sum = tuple(range(0, obj_set.ndim))

    obj_mean = obj_set.mean(dim=-1, keepdim=True)

    if regularizer.lower() == 'l2':
        residuals = (obj_set - obj_mean)**2
    elif regularizer.lower() == 'l1':
        residuals = torch.abs(obj_set - obj_mean)
    elif regularizer.lower() == 'kl':
        residuals = obj_mean * torch.log( obj_mean / obj_set )
    else:
        raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")

    if weight == "snr" and dset is not None:
        weight_map = calculate_weight_map(dset)
    elif weight == "id":
        weight_map = 1
    else:
        raise ValueError("W must be either 'snr' or 'id'")

    residuals = residuals * weight_map

    return residuals.sum(axis=axis_to_sum)