import torch
import string
import einops

from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift

from .utils import partial_convolution_rfft, centered_pad


def forward_model(otf, obj):
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


def fidelity_loss(dataset, obj_set, psf, eps=None, weight_map: torch.tensor = 1, full_KL=False):
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

    psf = torch.as_tensor(psf)

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

    if psf.is_floating_point():
        sz = dataset.shape[:num_axis_to_conv]
        psf_pad = centered_pad(psf, target_shape=sz) # Centered padding of PSF to match dataset shape
        psf_fft = rfftn(psf_pad, dim=tuple(range(num_axis_to_conv))) # FFT of the PSF
    elif torch.is_complex(psf):
        psf_fft = psf
    else:
        raise ValueError(f"Unsupported PSF dtype: {psf.dtype}")

    forward = partial_convolution_rfft(psf_fft, obj_set, dim1=str_psf, dim2=str_obj, dim3=str_final, axis=axis_str, fourier=(1,0))

    fidelity = forward - dataset * torch.log(forward + eps)

    if full_KL:
        fidelity = fidelity + dataset * torch.log( dataset ) - dataset

    idx = ~torch.isfinite(fidelity)
    fidelity[ idx ] = 0

    if torch.is_tensor(weight_map):
        letters = ["x", "y", "z"][: weight_map.ndim]
        lhs = " ".join(letters)
        ein_str = f"b {lhs} ... , {lhs} -> b"
        fidelity = einops.einsum(fidelity, weight_map, ein_str)
    else:
        fidelity = fidelity.sum(axis=axis_to_sum) # Sum over all but the first dimension (num_iter if present)

    return fidelity


def regularization_loss(obj_set: torch.tensor, regularizer: str = 'l2', weight_map: torch.tensor =1,
                        iteration_axis: bool = False, eps: float = None):
    """
    Compute the variance regularization term for a set of objects.

    Parameters:
    obj_set (Tensor): Input tensor of shape Nx x Ny x K.
    regularizer (str): Type of regularization ('l1', 'l2', 'kl').

    Returns:
    float: Variance regularization value.
    """

    if eps is None:
        eps = 1e-2

    if iteration_axis:
        axis_to_sum = tuple(range(1, obj_set.ndim))
    else:
        axis_to_sum = tuple(range(0, obj_set.ndim))

    centroid = compute_centroid(obj_set, regularizer)

    if regularizer.lower() == 'l2':
        residuals = (obj_set - centroid)**2
    elif regularizer.lower() == 'l1':
        residuals = torch.abs(obj_set - centroid)
    elif regularizer.lower() == 'kl':
        residuals = centroid * torch.log( (centroid + eps) / (obj_set + eps) )
    else:
        raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")

    residuals = residuals * weight_map

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