import torch
import string

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

    str_psf = alphabet[1:n_dim]   # 'bc' or 'bcd'
    str_obj = alphabet[:n_dim]   # 'abc' or 'abcd'
    axis_str = alphabet[1:1+num_axis_to_conv] # 'bc'

    sz = dataset.shape[:num_axis_to_conv]

    psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape

    forward = partial_convolution_rfft(psf_pad, obj_set, dim1=str_psf, dim2=str_obj, axis=axis_str, fourier=(0,0))
    
    dataset = dataset.unsqueeze(-1)

    fidelity = forward - dataset * torch.log( forward )
    if full_KL:
        fidelity = fidelity + dataset * torch.log( dataset ) - dataset

    idx = ~torch.isfinite(fidelity)
    fidelity[ idx ] = 0


    return fidelity.sum()


def regularization_loss(obj_set, regularizer: str = 'l2'):
    """
    Compute the variance regularization term for a set of objects.

    Parameters:
    obj_set (Tensor): Input tensor of shape Nx x Ny x K.
    regularizer (str): Type of regularization ('l1', 'l2', 'kl').

    Returns:
    float: Variance regularization value.
    """

    obj_mean = obj_set.mean(dim=-1, keepdim=True)

    if regularizer.lower() == 'l2':
        residuals = (obj_set - obj_mean)**2
    elif regularizer.lower() == 'l1':
        residuals = torch.abs(obj_set - obj_mean)
    elif regularizer.lower() == 'kl':
        residuals = obj_mean * torch.log( obj_mean / obj_set )
    else:
        raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")

    return residuals.sum()