import torch
import string

from .utils import partial_convolution_rfft, centered_pad

def fidelity_loss(dataset, obj_set, psf, full_KL=False):
    """
    Compute the fidelity loss between true and predicted values.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: Fidelity loss value.
    """

    alphabet = string.ascii_lowercase
    n_dim = obj_set.ndim

    str_psf = alphabet[:n_dim]
    str_obj = alphabet[:n_dim]
    if n_dim >2:
        axis_str = alphabet[:n_dim - 1]
    else:
        axis_str = alphabet[:n_dim]

    ny, nx = dataset.shape[:2]

    psf_pad = centered_pad(psf, target_shape=(ny, nx))  # Centered padding of PSF to match dataset shape

    forward = partial_convolution_rfft(psf_pad, obj_set, dim1=str_psf, dim2=str_obj, axis=axis_str, fourier=(0,0))

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