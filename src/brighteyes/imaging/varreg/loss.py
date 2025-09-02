import torch
import string
import numpy as np

from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift

from .utils import partial_convolution_rfft, centered_pad


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


def fidelity_loss(dataset, obj_set, psf, eps=None, full_KL=False):
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
        eps = 1e-2

    if psf is isinstance(psf, np.ndarray):
        psf = torch.from_numpy(psf)

    alphabet = string.ascii_lowercase
    n_dim = obj_set.ndim      # 3 or 4
    num_axis_to_conv = 2  

    if obj_set.shape[:num_axis_to_conv] == dataset.shape[:num_axis_to_conv]: #If I do not have num_iter
        obj_set = obj_set.unsqueeze(0)  # Add a batch dimension
        n_dim = n_dim + 1
        
    str_psf = alphabet[1:n_dim]   # 'bc' or 'bcd'
    str_obj = alphabet[:n_dim]   # 'abc' or 'abcd'
    axis_str = alphabet[1:1+num_axis_to_conv] # 'bc'
    str_final = str_obj
    axis_to_sum = tuple(range(1, obj_set.ndim))

    if psf.dtype == torch.float32 or psf.dtype == torch.float64:
        sz = dataset.shape[:num_axis_to_conv]
        psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape
        psf_fft = rfftn(psf_pad, dim=tuple(range(num_axis_to_conv)))  # FFT of the PSF
    elif psf.dtype == torch.complex64 or psf.dtype == torch.complex128:
        psf_fft = psf.clone()
    else:
        raise ValueError("Unsupported PSF data type.")

    forward = partial_convolution_rfft(psf_fft, obj_set, dim1=str_psf, dim2=str_obj, dim3=str_final, axis=axis_str, fourier=(1,0))

    fidelity = forward - dataset * torch.log(forward + eps)

    if full_KL:
        fidelity = fidelity + dataset * torch.log( dataset ) - dataset

    idx = ~torch.isfinite(fidelity)
    fidelity[ idx ] = 0


    return fidelity.sum(axis=axis_to_sum)  # Sum over all but the first dimension (num_iter if present)


def regularization_loss(obj_set: torch.tensor, regularizer: str = 'l2', weight_map: torch.tensor =1,
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

    residuals = residuals * weight_map

    return residuals.sum(axis=axis_to_sum)