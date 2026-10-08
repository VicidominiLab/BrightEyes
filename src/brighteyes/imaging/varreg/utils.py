# Array-dimension conventions used throughout this module:
# Nx, Ny   : lateral image dimensions
# Mx, My   : unpadded kernel or PSF dimensions
# Nz       : axial object/PSF planes
# Nch      : detector or measurement channels
# K        : independent noise realizations
# Niter    : saved reconstruction iterations
# L        : repeated experiments or reconstruction runs

import torch
import einops
import string
import numpy as np

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift, fftn, ifftn, fftshift
from torch import real, einsum


def _axis_indices(dimensions, axis):
    """Return the positions of named FFT axes after validating all labels."""
    missing = [(label, dims) for dims in dimensions for label in axis if label not in dims]
    if missing:
        details = ', '.join(f"'{label}' in '{dims}'" for label, dims in missing)
        raise ValueError(f"FFT axis labels are missing from dimension strings: {details}.")
    return [[dims.index(label) for label in axis] for dims in dimensions]

def partial_convolution_fft(volume: torch.Tensor, kernel: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                        axis: str = 'jk', fourier: tuple = (False, False)):

    """Convolve two tensors along named axes using full complex FFTs.

    Each character in ``dim1`` and ``dim2`` labels one tensor axis and is used
    directly in the Einstein summation. The characters listed in ``axis`` are
    the spatial axes transformed by the FFT. Output axes follow the first
    occurrence of each label in ``dim1 + dim2``.

    Parameters
    ----------
    volume : torch.Tensor
        First input tensor, with one axis for each character in ``dim1``.
    kernel : torch.Tensor
        Second input tensor, with one axis for each character in ``dim2``.
    dim1, dim2 : str
        Axis labels for ``volume`` and ``kernel``. Shared labels identify axes
        that must have compatible lengths.
    axis : str, default='jk'
        Labels of the axes on which to apply the forward and inverse FFTs.
    fourier : tuple[bool, bool], default=(False, False)
        Whether ``volume`` and ``kernel``, respectively, are already in the
        Fourier domain.

    Returns
    -------
    torch.Tensor
        Real-valued convolution result. Its axis order is printed and follows
        the ordered union of ``dim1`` and ``dim2``.

    Notes
    -----
    No explicit zero-padding is applied. The transformed axes must therefore
    already have compatible sampled sizes, and the operation follows the
    same-size circular-convolution convention.
    """
    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))
    print(f'Final dim: {dim3}')

    dims = [dim1, dim2, dim3]

    axis_list = _axis_indices(dims, axis)
    # axis_list = [[d.find(c) for c in axis] for d in dims]

    if fourier[0] == False:
        volume_fft = fftn(volume, dim=axis_list[0])
    else:
        volume_fft = volume

    if fourier[1] == False:
        kernel_fft = fftn(kernel, dim=axis_list[1])
    else:
        kernel_fft = kernel

    conv = torch.einsum(f'{dim1},{dim2}->{dim3}', volume_fft, kernel_fft)

    conv = ifftn(conv, dim=axis_list[2])  # Transform the Fourier-domain product back to real space
    conv = ifftshift(conv, dim=axis_list[2])  # Move the spatial origin to the centered-array convention
    conv = torch.real(conv)  # Discard residual imaginary values from numerical round-off

    return conv

def partial_convolution_rfft(kernel: torch.Tensor, volume: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                             axis: str = 'jk',  dim3: str = None, fourier: tuple = (False, False),
                             padding: list = None):

    """Convolve real tensors along named axes using real-valued FFTs.

    The axis-label strings are passed to ``torch.einsum`` after transforming
    the labels listed in ``axis``. This supports partial convolutions while
    preserving non-convolved axes such as axial planes, detector channels, or
    independent noise realizations.

    Parameters
    ----------
    kernel : torch.Tensor or array-like
        First input, with one axis for each character in ``dim1``.
    volume : torch.Tensor or array-like
        Second input, with one axis for each character in ``dim2``.
    dim1, dim2 : str
        Axis labels for ``kernel`` and ``volume``.
    axis : str, default='jk'
        Labels of the real-space axes to transform and convolve.
    dim3 : str or None, default=None
        Output axis labels. When omitted, the ordered union of ``dim1`` and
        ``dim2`` is used.
    fourier : tuple[bool, bool], default=(False, False)
        Whether ``kernel`` and ``volume``, respectively, are already in the
        Fourier domain.
    padding : list or tuple or None, default=None
        Real-space lengths passed as ``s`` to ``rfftn`` and ``irfftn``. By
        default, lengths are read from the transformed axes of ``volume``.

    Returns
    -------
    torch.Tensor
        Real-valued convolution result with axes ordered according to ``dim3``.

    Notes
    -----
    With the default padding taken from ``volume``, the result follows a
    same-size circular-convolution convention.
    """
    kernel = torch.as_tensor(kernel)
    volume = torch.as_tensor(volume)

    if dim3 is None:
        dim3 = dim1 + dim2
        dim3 = ''.join(sorted(set(dim3), key=dim3.index))
    elif not isinstance(dim3, str):
        raise ValueError("dim3 must be a string")

    dims = [dim1, dim2, dim3]

    axis_list = _axis_indices(dims, axis)

    if padding is None:
        padding = [volume.size(d) for d in axis_list[1]]

    if fourier[0] == False:
        # kernel_shifted = ifftshift(kernel, dim=axis_list[0])  # Optional kernel centering
        kernel_fft = rfftn(kernel, dim=axis_list[0], s=padding)
    else:
        kernel_fft = kernel

    if fourier[1] == False:
        # volume_shifted = ifftshift(volume, dim=axis_list[1])  # Optional volume centering
        volume_fft = rfftn(volume, dim=axis_list[1], s=padding)
    else:
        volume_fft = volume

    conv = einsum(f'{dim1},{dim2}->{dim3}', kernel_fft, volume_fft)
    
    conv = irfftn(conv, dim=axis_list[2], s=padding)  # Transform the Fourier-domain product back to real space
    conv = ifftshift(conv, dim=axis_list[2])  # Move the spatial origin to the centered-array convention
    conv = real(conv)  # Discard residual imaginary values from numerical round-off

    return conv


def forward_model(otf, obj):
    """Apply the repository's 2D forward or back-projection convolution.

    The first OTF axis is the object/PSF axial axis, axes 1 and 2 are the
    lateral Fourier axes, and the optional final OTF axis is the detector
    channel axis. The last object/data axis is the realization axis ``K``.

    Typical forward-projection shapes are::

        otf : (Nz, Nx, Ny_rfft, Nch)
        obj : (Nz, Nx, Ny, K)
        out : (Nz, Nx, Ny, Nch, K)

    Typical back-projection shapes are::

        otf : (Nz, Nx, Ny_rfft, Nch)
        obj : (Nx, Ny, Nch, K)
        out : (Nz, Nx, Ny, Nch, K)

    Callers subsequently sum over ``Nz`` for forward projection or over
    ``Nch`` for back-projection. Channel-free OTFs follow the analogous shape
    convention without ``Nch``.

    Parameters
    ----------
    otf : torch.Tensor
        Fourier-domain PSF, usually generated with ``compute_otf_from_psf``.
    obj : torch.Tensor
        Object estimates for forward projection or data-space ratios for
        back-projection.

    Returns
    -------
    torch.Tensor
        Partially convolved tensor before summation over axial planes or
        detector channels.

    Notes
    -----
    Back-projection is identified first from the data-space layout. This
    resolves the otherwise ambiguous case ``Nz == Nx == Ny`` without changing
    the standard forward/back-projection conventions used by the package.
    """
    conv_dims = 2
    alphabet = string.ascii_lowercase

    str_otf = alphabet[:otf.ndim]       # 'abcd' for (Nz, Nx, Ny_rfft, Nch)
    axis_str = alphabet[1:conv_dims+1]  # 'bc': always convolve lateral axes 1 and 2

    is_forward = obj.ndim >= 3 and obj.shape[0] == otf.shape[0] and obj.shape[1] == otf.shape[1]
    is_backprojection = obj.ndim == otf.ndim and obj.shape[0] == otf.shape[1]
    if otf.ndim >= 4:
        is_backprojection = is_backprojection and obj.shape[-2] == otf.shape[-1]

    if is_backprojection:
        str_obj = alphabet[1:obj.ndim+1]  # 'bcde' for (Nx, Ny, Nch, K)
    elif is_forward:
        str_obj = alphabet[:conv_dims+1] + alphabet[otf.ndim]  # 'abce' for (Nz, Nx, Ny, K)
    else:
        raise ValueError(f"Incompatible OTF and object shapes: {tuple(otf.shape)} and {tuple(obj.shape)}.")

    dset = partial_convolution_rfft(otf, obj, dim1=str_otf, dim2=str_obj, axis=axis_str, fourier=(1, 0))
    return dset


def centered_pad(arr: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Zero-pad the two square spatial axes around the input center.

    The supported layouts are ``(Mx, My)``, ``(Mx, My, Nch)``,
    ``(Nz, Mx, My)``, and ``(Nz, Mx, My, Nch)``. For tensors with at least
    three axes, equal dimensions at positions 1 and 2 are interpreted as the
    lateral PSF axes. This gives the 3D PSF convention priority when
    ``Nz == Mx == My``. Otherwise, equal dimensions at positions 0 and 1 are
    interpreted as a 2D spatial pair.

    Parameters
    ----------
    arr : torch.Tensor
        Tensor containing square spatial dimensions ``(Mx, My)`` in axes
        ``(0, 1)`` or ``(1, 2)``.
    target_shape : tuple
        Target lateral size; only ``target_shape[:2] = (Nx, Ny)`` is used.

    Returns
    -------
    torch.Tensor
        Zero-padded tensor with unchanged non-spatial axes and lateral size
        ``(Nx, Ny)``. For odd padding differences, the extra pixel is placed
        on the right or bottom side.
    """
    conv_dims = 2
    if arr.ndim >= 3 and arr.shape[1] == arr.shape[2]:
        Mx, My = arr.shape[1:conv_dims+1]
        axis_to_move = (1, 2)
    elif arr.ndim >= 2 and arr.shape[0] == arr.shape[1]:
        Mx, My = arr.shape[:conv_dims]
        axis_to_move = (0, 1)
    else:
        raise ValueError("Input tensor must contain a square spatial pair in axes (0, 1) or (1, 2).")

    Nx, Ny = target_shape[:conv_dims]
    if Nx < Mx or Ny < My:
        raise ValueError("Target shape must be larger than or equal to the input shape in both dimensions.")

    pad_x_total = Nx - Mx
    pad_y_total = Ny - My
    pad_left = pad_y_total // 2
    pad_right = pad_y_total - pad_left
    pad_top = pad_x_total // 2
    pad_bottom = pad_x_total - pad_top

    # Move the inferred spatial axes to the end, pad them, and restore the original order.
    arr_moved = torch.movedim(arr, axis_to_move, (-2, -1))
    arr_moved = functional.pad(arr_moved, (pad_left, pad_right, pad_top, pad_bottom))
    return torch.movedim(arr_moved, (-2, -1), axis_to_move)


def compute_otf_from_psf(psf, target_shape):
    """Pad, normalize, and Fourier-transform a PSF over its lateral axes.

    The repository convention is ``psf.shape = (Nz, Mx, My, Nch)``. A common
    confocal case is ``(1, Mx, My, 1)``. The PSF is padded to ``(Nx, Ny)``,
    normalized independently for every first-axis entry over all remaining
    axes, and transformed over axes ``(1, 2)``.

    Parameters
    ----------
    psf : torch.Tensor
        Real-space PSF, typically shaped ``(Nz, Mx, My, Nch)``.
    target_shape : tuple
        Target lateral image size. The first two values are interpreted as
        ``(Nx, Ny)``.

    Returns
    -------
    torch.Tensor
        Complex OTF with typical shape ``(Nz, Nx, Ny // 2 + 1, Nch)``.
    """
    num_axis_to_conv = 2
    axis_to_conv = tuple(torch.arange(1, num_axis_to_conv+1))
    psf_pad = centered_pad(psf, target_shape=target_shape)  # (Nz, Nx, Ny, Nch)
    # Normalize each first-axis PSF entry over all remaining axes
    norm_axis = tuple(np.arange(1, psf.ndim))
    psf_norm = psf_pad / psf_pad.sum(axis=norm_axis, keepdim=True)
    psf_fft = rfftn(psf_norm, dim=axis_to_conv)  # (Nz, Nx, Ny // 2 + 1, Nch)
    
    return psf_fft

def calculate_snr(dset, smooth_sigma: float = 0, eps = None):
    """Compute a pixel-wise signal-to-noise ratio across realizations.

    The last input axis is interpreted as ``K``, the number of independent
    noise realizations. Signal and noise are the sample mean and PyTorch sample
    standard deviation across that axis.

    Parameters
    ----------
    dset : torch.Tensor or np.ndarray
        Dataset shaped ``(..., K)``; in current usage this is ``(Nx, Ny, K)``.
    smooth_sigma : float, default=0
        Standard deviation of an optional SciPy Gaussian filter. Smoothing is
        CPU-only and adds the legacy offset ``0.1`` after filtering.
    eps : float or None, default=None
        Positive denominator offset. The float32 machine epsilon is used when
        omitted.

    Returns
    -------
    torch.Tensor
        SNR map with shape ``dset.shape[:-1]``.
    """
    
    if eps is None:
        eps = torch.finfo(torch.float32).eps

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32)

    signal = dset.mean(dim = -1)
    noise = dset.std(dim=-1)

    snr = signal / (noise + eps)

    if smooth_sigma > 0:
        offset = 1e-1
        snr = gaussian_filter(snr.numpy(), sigma=smooth_sigma)
        snr = torch.tensor(snr, dtype=torch.float32) + offset

    return snr

def normalization(x, vmin, vmax):
    """Linearly map the global input range to ``[vmin, vmax]``.

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        Input array of arbitrary shape.
    vmin, vmax : scalar
        Requested output values corresponding to the global minimum and
        maximum of ``x``.

    Returns
    -------
    torch.Tensor or np.ndarray
        Rescaled array with the same shape and backend as ``x``. Constant
        inputs evaluate to ``vmin`` because of the fixed denominator offset.
    """
    x_min = x.min()
    x_max = x.max()

    x_norm = vmin + (x - x_min) * (vmax - vmin) / (x_max - x_min + 1e-18)
    return x_norm

def calculate_weight_map(dset, smooth_sigma: float = 0, normalize: bool = True):
    """Build an inverse-SNR weight map and repeat it over realizations.

    Parameters
    ----------
    dset : torch.Tensor or np.ndarray
        CPU dataset shaped ``(Nx, Ny, K)``. The last axis contains independent
        noise realizations.
    smooth_sigma : float, default=0
        Standard deviation of the optional spatial Gaussian smoothing.
    normalize : bool, default=True
        If true, linearly map the weight range from its current minimum to 1.

    Returns
    -------
    np.ndarray
        Weight array shaped ``(Nx, Ny, K)``. The same 2D inverse-SNR map is
        repeated along ``K``.

    Notes
    -----
    The SNR is clipped at its 99.99th percentile before inversion. This helper
    converts through NumPy and is therefore intended for CPU tensors.
    """

    if not isinstance(dset, torch.Tensor):
        dset = torch.tensor(dset, dtype=torch.float32)

    snr = calculate_snr(dset)
    snr = torch.clamp(snr, max=torch.quantile(snr, 0.9999)) 
    
    weight_map = 1 / snr

    max_weight = torch.max(weight_map[torch.isfinite(weight_map)])
    weight_map = torch.clamp(weight_map, min=0, max=max_weight)
    
    if smooth_sigma > 0:
        weight_map = torch.tensor(gaussian_filter(weight_map.numpy(), sigma=smooth_sigma))

    weight_map = einops.repeat(weight_map, 'm n -> m n k', k=dset.shape[-1])
    
    if normalize:
        weight_map = normalization(weight_map, vmin=torch.min(weight_map), vmax=1)

    return weight_map.numpy()

def calculate_noise_variance(obj):
    """Compute Fourier-domain statistics across repeated reconstructions.

    Parameters
    ----------
    obj : torch.Tensor
        Reconstruction collection shaped ``(Niter, Nx, Ny, L)``, where ``L``
        indexes repeated experiments or reconstruction realizations.

    Returns
    -------
    noise_mean : torch.Tensor
        Squared magnitude of the mean complex Fourier spectrum across ``L``,
        shaped ``(Niter, Nx, Ny)``.
    noise_variance : torch.Tensor
        Magnitude of the variance of the complex Fourier coefficients across
        ``L``, shaped ``(Niter, Nx, Ny)``.

    Notes
    -----
    The FFT is unnormalized and ``torch.var`` uses its default correction.
    """

    obj_fft = fftn(obj, dim=(1,2))
    obj_fftshifted = fftshift(obj_fft, dim=(1,2))  # Place the zero-frequency component at the array center
    mean = obj_fftshifted.mean(dim=-1, keepdim=False)  # Average complex Fourier coefficients over repeated runs L
    variance = obj_fftshifted.var(dim=-1, keepdim=False)
    noise_mean = torch.abs(mean) ** 2  # Squared magnitude of the mean complex spectrum
    noise_variance = torch.abs(variance) # Magnitude of the Fourier-coefficient variance
    
    return noise_mean, noise_variance

def radial_profile(data, center):
    """Sum a 2D image into integer-radius annular bins.

    Parameters
    ----------
    data : array-like
        Two-dimensional real or complex image shaped ``(Nx, Ny)``. Only the
        real part contributes to the profile.
    center : sequence of length 2
        Radial origin in ``(x, y)`` index order.

    Returns
    -------
    radialprofile : np.ndarray
        Sum of image values in each integer-radius bin, shaped ``(Nr,)``.
    nr : np.ndarray
        Number of pixels contributing to each radius, also shaped ``(Nr,)``.

    Notes
    -----
    This function returns annular sums, not annular means.
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
    """Compute radial profiles of Fourier variance for each iteration.

    Parameters
    ----------
    obj : torch.Tensor
        CPU tensor shaped ``(Niter, Nx, Ny, L)``. ``L`` indexes repeated
        reconstructions used to estimate the Fourier variance.

    Returns
    -------
    torch.Tensor
        Radial variance profiles shaped ``(Niter, Nr)``.
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
    return torch.from_numpy(np.stack(radial_variance))

def root_mean_square_error(img_ref, img):
    """Compute RMSE between a reference image and one or more estimates.

    Parameters
    ----------
    img_ref : torch.Tensor or np.ndarray
        Reference image shaped ``(Nx, Ny)`` or with the same trailing shape as
        ``img``.
    img : torch.Tensor or np.ndarray
        Estimate shaped ``(Nx, Ny)`` or ``(..., Nx, Ny)``. Leading axes are
        retained in the output.

    Returns
    -------
    np.ndarray
        Scalar RMSE or an array over the leading axes of ``img`` after
        singleton dimensions are squeezed.
    """

    img = torch.tensor(img, dtype=torch.float32) if not isinstance(img, torch.Tensor) else img.float()
    img_ref = torch.tensor(img_ref, dtype=torch.float32) if not isinstance(img_ref, torch.Tensor) else img_ref.float()

    diff_dim = img.ndim - img_ref.ndim

    if diff_dim > 0:
        img_ref = img_ref.unsqueeze(0)

    axis_to_sum = tuple(range(diff_dim, img.ndim))

    mse = torch.mean((img - img_ref) ** 2, dim=axis_to_sum)
    rmse = torch.sqrt(mse)

    return torch.squeeze(rmse).numpy()

def peak_signal_to_noise_ratio(img_ref, img, normalize_entries = True):
    """Compute peak signal-to-noise ratio for one or more estimates.

    Parameters
    ----------
    img_ref : torch.Tensor or np.ndarray
        Reference image shaped ``(Nx, Ny)`` or matching the trailing image
        dimensions of ``img``.
    img : torch.Tensor or np.ndarray
        Estimate shaped ``(Nx, Ny)`` or ``(..., Nx, Ny)``.
    normalize_entries : bool, default=True
        Normalize each image by its sum over the compared image axes before
        computing the mean-squared error.

    Returns
    -------
    np.ndarray
        PSNR in decibels, with leading estimate axes retained and singleton
        dimensions squeezed.
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

def kl_divergence(img_ref, img, normalize_entries=True, eps=None, full_KL=False, k_loss=False):
    """Compute a reduced Poisson Kullback--Leibler data fidelity.

    Parameters
    ----------
    img_ref : torch.Tensor or np.ndarray
        Reference photon-count data ``a`` with shape ``(Nx, Ny, ...)``.
    img : torch.Tensor or np.ndarray
        Predicted data ``b`` with the same shape or one additional leading
        iteration axis, commonly ``(Niter, Nx, Ny, Nch, K)``.
    normalize_entries : bool, default=True
        Normalize both inputs by their sums over the reduction axes.
    eps : float or None, default=None
        Positive offset used inside divisions and logarithms.
    full_KL : bool, default=False
        Include the linear term ``b - a`` and compute
        ``b - a + a * log(a / b)``. When false, only the logarithmic term is
        returned; this is equivalent after summation when flux is conserved.
    k_loss : bool, default=False
        Preserve the last axis ``K`` instead of reducing it.

    Returns
    -------
    torch.Tensor
        Reduced fidelity values on the same device as the tensor input, with
        singleton dimensions squeezed.
    """
    device = img.device if isinstance(img, torch.Tensor) else img_ref.device if isinstance(img_ref, torch.Tensor) else None
    img = torch.as_tensor(img, dtype=torch.float32, device=device)
    img_ref = torch.as_tensor(img_ref, dtype=torch.float32, device=device)

    if eps is None:
        eps = torch.finfo(img.dtype).eps

    diff_dim = img.ndim - img_ref.ndim
    if diff_dim < 0:
        raise ValueError("img cannot have fewer dimensions than img_ref.")
    if diff_dim > 0:
        img_ref = img_ref.reshape((1,) * diff_dim + tuple(img_ref.shape))

    if k_loss is True:
        axis_to_sum = tuple(range(diff_dim, img.ndim-1))  # Preserve Niter and the final realization axis K
    else:
        axis_to_sum = tuple(range(diff_dim, img.ndim))  # Preserve only leading iteration axes

    if normalize_entries is True:
        img = img / torch.sum(img, dim=axis_to_sum, keepdim=True)
        img_ref = img_ref / torch.sum(img_ref, dim=axis_to_sum, keepdim=True)

    fidelity = img_ref * torch.log((img_ref + eps) / (img + eps))
    if full_KL:
        fidelity = fidelity + img - img_ref

    return torch.squeeze(torch.sum(fidelity, dim=axis_to_sum))


def autocorrelation2d(image, normalize=True):
    """Compute a centered circular autocorrelation of a 2D image.

    Parameters
    ----------
    image : array-like
        Two-dimensional image shaped ``(Nx, Ny)``. It is converted to a float
        NumPy array and mean-centered before the FFT.
    normalize : bool, default=True
        Divide the autocorrelation by its global maximum.

    Returns
    -------
    np.ndarray
        Real, FFT-shifted circular autocorrelation with the same shape as the
        input image.
    """
    image = np.asarray(image, dtype=float)
    image = image - image.mean()

    f = np.fft.fftn(image)
    ac = np.fft.ifftn(np.abs(f)**2).real
    ac = np.fft.fftshift(ac)

    if normalize:
        ac /= ac.max()

    return ac