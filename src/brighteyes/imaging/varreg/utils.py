import torch
import einops
import string
import numpy as np

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift
from torch import real, einsum

def partial_convolution_fft(volume: torch.tensor, kernel: torch.tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                        axis: str = 'jk', fourier: tuple = (False, False)):

    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))

    dims = [dim1, dim2, dim3]

    axis_list = [[d.find(c) for c in axis] for d in dims]

    if fourier[0] == False:
        volume_fft = fftn(volume, dim=axis_list[0])
    else:
        volume_fft = volume

    if fourier[1] == False:
        kernel_fft = fftn(kernel, dim=axis_list[1])
    else:
        kernel_fft = kernel

    conv = torch.einsum(f'{dim1},{dim2}->{dim3}', volume_fft, kernel_fft)

    conv = ifftn(conv, dim=axis_list[2])  # inverse FFT of the product
    conv = ifftshift(conv, dim=axis_list[2])  # Rotation of 180 degrees of the phase of the FFT
    conv = torch.real(conv)  # Clipping to zero the residual imaginary part

    return conv

def partial_convolution_rfft(kernel: torch.Tensor, volume: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                             axis: str = 'jk',  dim3: str = None, fourier: tuple = (False, False),
                             padding: list = None):

    kernel = torch.as_tensor(kernel)
    volume = torch.as_tensor(volume)

    if dim3 is None:
        dim3 = dim1 + dim2
        dim3 = ''.join(sorted(set(dim3), key=dim3.index))
    elif not isinstance(dim3, str):
        raise ValueError("dim3 must be a string")

    dims = [dim1, dim2, dim3]

    axis_list = [[d.find(c) for c in axis] for d in dims]

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]

    if fourier[0] == False:
        #kernel_shifted = ifftshift(kernel, dim=axis_list[0])  # Centra il kernel
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel

    if fourier[1] == False:
        #volume_shifted = ifftshift(volume, dim=axis_list[1])  # Centra il volume
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume

    conv = einsum(f'{dim1},{dim2}->{dim3}', kernel_fft, volume_fft)
    
    conv = irfftn(conv, dim=axis_list[2], s=padding)  # inverse FFT of the product
    conv = ifftshift(conv, dim=axis_list[2])  # Rotation of 180 degrees of the phase of the FFT
    conv = real(conv)  # Clipping to zero the residual imaginary part

    return conv


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
    axis_str = alphabet[:conv_dims]
    
    if obj.ndim == otf.ndim and obj.shape[-1] != otf.shape[-1]:  # If 
        str_obj = alphabet[:conv_dims] + alphabet[otf.ndim + 1]
    else:
        str_obj = alphabet[:obj.ndim]
    
    dset = partial_convolution_rfft(otf, obj, dim1=str_psf, dim2=str_obj, axis=axis_str, fourier=(1, 0))

    return dset


def centered_pad(arr: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Pads a tensor to match the target shape by centering the original data.

    Parameters:
        arr (Tensor): input tensor of shape (Mx, My).
        target_shape (tuple): Target shape (Nx, Ny).

    Returns:
        Tensor: Padded tensor of shape (Nx, Ny) with the input centered.
    """

    conv_dims = 2

    Mx, My = arr.shape[:conv_dims]
    Nx, Ny = target_shape[:conv_dims]

    if Nx < Mx or Ny < My:
        raise ValueError("Target shape must be larger than or equal to the input shape in both dimensions.")

    pad_x_total = Nx - Mx
    pad_y_total = Ny - My

    pad_left = pad_y_total // 2
    pad_right = pad_y_total - pad_left
    pad_top = pad_x_total // 2
    pad_bottom = pad_x_total - pad_top
    pad = (pad_left, pad_right, pad_top, pad_bottom)
    pad_extended = (0,) * (2 * (arr.ndim - 2)) + pad

    padded_arr = functional.pad(arr, pad_extended)

    return padded_arr


def calculate_snr(dset, smooth_sigma: float = 0):
    """
    Calculate pixel-wise the Signal-to-Noise Ratio (SNR) of a dataset.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        smooth_sigma (float): Standard deviation for Gaussian smoothing. Default is 0 (no smoothing).

    Returns:
        float: SNR value.
    """

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32)

    signal = dset.mean(dim = -1)
    noise = dset.std(dim=-1)

    snr = torch.where(signal == 0, 0, signal / noise)

    if smooth_sigma > 0:
        offset = 1e-1
        snr = gaussian_filter(snr.numpy(), sigma=smooth_sigma)
        snr = torch.tensor(snr, dtype=torch.float32) + offset

    return snr


def calculate_weight_map(dset, smooth_sigma: float = 0):
    """
    Calculate a weight map based on the SNR of the dataset.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        smooth_sigma (float): Standard deviation for Gaussian smoothing. Default is 0 (no smoothing).

    Returns:
        Tensor: Weight map of shape Nx x Ny.
    """

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32)

    snr = calculate_snr(dset, smooth_sigma)

    weight_map = 1 / snr

    max_weight = torch.max(weight_map[torch.isfinite(weight_map)])
    weight_map = torch.clip(weight_map, 0, max_weight)

    weight_map = einops.repeat(weight_map, 'm n -> m n k', k=dset.shape[-1])

    return weight_map

def calculate_noise_variance(obj):
    """ Compute the noise variance of an object tensor.

    Parameters:
        obj (Tensor): Input object tensor of shape N_iter x Nx x Ny x K.

    Returns:
        noise mean (Tensor): Mean of the power spectrum across K dimension shape (N_iter, Nx, Ny).
        noise variance (Tensor): Variance of the power spectrum across K dimension shape (N_iter, Nx, Ny).
    """

    obj_fft = fftn(obj, dim=(1,2))
    obj_fftshifted = fftshift(obj_fft, dim=(1,2))  # Shift the zero frequency component to the center
    mean = obj_fftshifted.mean(dim=-1, keepdim=False)  # Average over K dimension
    variance = obj_fftshifted.var(dim=-1, keepdim=False)
    noise_mean = torch.abs(mean) ** 2  # Power spectrum mean
    noise_variance = torch.abs(variance) # Power spectrum variance
    
    return noise_mean, noise_variance

def radial_profile(data, center):
    """
    Calculation of the radial profile of an image

    Parameters
    ----------
    data : np.array(Nx x Ny)
        image
    center : np.array(2)
        indices of the center of the image

    Returns
    -------
    radialprofile : np.array( np.sqrt(Nx**2 + Ny**2) )
        sum of the data over the angular coordinate
    nr : int
        number of pixels in the angular bin
    """
    
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), np.real(data).ravel())
    #tbin += 1j*np.bincount(r.ravel(), np.imag(data).ravel())
    
    nr = np.bincount(r.ravel())
    radialprofile = tbin

    return radialprofile, nr

def compute_radial_noise_variance(obj):
    """ Compute the radial noise variance of an object tensor.

    Parameters:
        obj (Tensor): Input object tensor of shape N_iter x Nx x Ny x K.

    Returns:
        Tensor: Radial noise variance of shape (N_iter, Nx, Ny).
    """
    n_iter, Nx, Ny, _ = obj.shape
    cx = int((Nx + np.mod(Nx, 2)) / 2)
    cy = int((Ny + np.mod(Ny, 2)) / 2)
    center = [cx, cy]

    # Compute the noise variance
    noise_mean, noise_variance = calculate_noise_variance(obj)
    radial_variance = []
    for n in range(n_iter):
        dato, _ = radial_profile(noise_variance[n], center=center)
        radial_variance.append(dato)
    return torch.tensor(radial_variance)


def peak_signal_to_noise_ratio(img_ref, img, normalize_entries = True):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        img (Tensor): Input image of shape Nx x Ny.
        img_ref (Tensor): Reference image of shape Nx x Ny.

    Returns:
        psnr (Tensor): PSNR value in decibels (dB).
    """

    img = torch.tensor(img, dtype=torch.float32) if not isinstance(img, torch.Tensor) else img.float()
    img_ref = torch.tensor(img_ref, dtype=torch.float32) if not isinstance(img_ref, torch.Tensor) else img_ref.float()

    diff_dim = img.ndim - img_ref.ndim

    if diff_dim > 0:
        img_ref = img_ref.unsqueeze(0)

    axis_to_sum = tuple(range(diff_dim, img.ndim))

    if normalize_entries is True:
        img = img / torch.sum(img, dim=axis_to_sum, keepdim=True)
        img_ref = img_ref / torch.sum(img_ref, dim=axis_to_sum, keepdim=True)

    mse = torch.mean((img - img_ref) ** 2, dim=axis_to_sum)

    psnr = 20 * torch.log10( img_ref.max() / torch.sqrt(mse) )

    return torch.squeeze(psnr).numpy()

def kl_divergence(img_ref, img, normalize_entries=True, eps=None, full_KL=False):
    """
    Compute the Kullback-Leibler divergence (KL divergence) between two images.

    Parameters:
        img (Tensor): Input image of shape Nx x Ny.
        img_ref (Tensor): Reference image of shape Nx x Ny.

    Returns:
        kl_div (Array): KL divergence value.
    """

    if eps is None:
        eps = torch.finfo(torch.float32).eps
    img = torch.tensor(img, dtype=torch.float32) if not isinstance(img, torch.Tensor) else img.float()
    img_ref = torch.tensor(img_ref, dtype=torch.float32) if not isinstance(img_ref, torch.Tensor) else img_ref.float()

    diff_dim = img.ndim - img_ref.ndim

    if diff_dim > 0:
        img_ref = img_ref.unsqueeze(0)

    axis_to_sum = tuple(range(diff_dim, img.ndim))

    if normalize_entries is True:
        img = img / torch.sum(img, dim=axis_to_sum, keepdim=True)
        img_ref = img_ref / torch.sum(img_ref, dim=axis_to_sum, keepdim=True)
    
    arg = img_ref / (img + eps)
    fidelity = img_ref * torch.log( arg + eps )
    
    if full_KL:
        fidelity = fidelity + img_ref - img

    kl_div = torch.sum(fidelity, dim=axis_to_sum)

    return torch.squeeze(kl_div).numpy()

def forward_fill_nan(tensor):
    
    tensor = torch.tensor(tensor)
    
    # mask True dove ci sono NaN
    mask = torch.isnan(tensor)
    
    # Sostituisco i NaN con 0 temporaneamente
    tensor = torch.where(mask, torch.zeros_like(tensor), tensor)

    # Creiamo un indicatore cumulativo dei valori non-NaN
    # Questo ci permette di propagare l'ultimo valore valido
    # step 1: indice booleano per i valori validi
    valid = (~mask).float()

    # step 2: cumulative sum per contare quante volte abbiamo valori validi
    cumsum_valid = valid.cumsum(dim=0)

    # step 3: somma cumulativa dei valori
    cumsum_tensor = tensor.cumsum(dim=0)

    # step 4: calcolo della media cumulativa * numero di valori validi → ultimo valore valido
    filled = torch.where(cumsum_valid > 0, cumsum_tensor / cumsum_valid, torch.zeros_like(tensor))

    return filled
