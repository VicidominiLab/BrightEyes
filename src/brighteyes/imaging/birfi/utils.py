import torch
from torch.fft import fftn, ifftn, ifftshift
import numpy as np
import matplotlib.pyplot as plt


def pad_tensor(x: torch.Tensor, pad_left: int, pad_right: int, dim: int, mode: str = "reflect"):
    """
    Pad a tensor along one dimension.

    Args:
        x (torch.Tensor): Input tensor
        pad_left (int): Padding size before data
        pad_right (int): Padding size after data
        dim (int): Dimension along which to pad
        mode (str, optional): Padding mode. One of {"reflect", "replicate", "constant"}.
                              Default = "reflect"

    Returns:
        torch.Tensor: Padded tensor
    """
    if pad_left == 0 and pad_right == 0:
        return x

    L = x.shape[dim]
    indices = torch.arange(L, device=x.device)

    if mode == "reflect":
        left_idx = torch.arange(pad_left, 0, -1, device=x.device)
        right_idx = torch.arange(L - 2, L - pad_right - 2, -1, device=x.device)
    elif mode == "replicate":
        left_idx = torch.zeros(pad_left, dtype=torch.long, device=x.device)
        right_idx = torch.full((pad_right,), L - 1, dtype=torch.long, device=x.device)
    elif mode == "constant":
        pad_shape = list(x.shape)
        pad_shape[dim] = pad_left + pad_right
        constant_pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([constant_pad.narrow(dim, 0, pad_left),
                          x,
                          constant_pad.narrow(dim, pad_left, pad_right)], dim=dim)
    else:
        raise ValueError(f"Unsupported padding mode: {mode}")

    # Select slices
    pad_left_tensor = x.index_select(dim, left_idx)
    pad_right_tensor = x.index_select(dim, right_idx)

    return torch.cat([pad_left_tensor, x, pad_right_tensor], dim=dim)


def median_filter(x: torch.Tensor, window_size=3, dims=None, mode="reflect"):
    """
    Apply an N-dimensional median filter over user-specified dimensions.

    Args:
        x (torch.Tensor): Input tensor (any shape).
        window_size (int or list/tuple of ints, optional): Window size(s).
            - If int, same size is used for all selected dims.
            - If list/tuple, must have length equal to len(dims).
            Default: 3
        dims (list/tuple of ints, optional): Dimensions to filter along.
            If None, all dimensions are filtered.
        mode (str, optional): Padding mode. One of {"reflect", "replicate", "constant"}.

    Returns:
        torch.Tensor: Median-filtered tensor (same shape as x).
    """
    if dims is None:
        dims = list(range(x.ndim))

    if isinstance(window_size, int):
        window_size = [window_size] * len(dims)
    elif len(window_size) != len(dims):
        raise ValueError("window_size must be scalar or match len(dims)")

    # check for odd values
    for w in window_size:
        if w % 2 == 0:
            raise ValueError(f"All window sizes must be odd, got {w}")

    out = x
    for d, w in zip(dims, window_size):
        pad_left = (w - 1) // 2
        pad_right = w // 2

        # Pad along dimension
        out = pad_tensor(out, pad_left, pad_right, d, mode=mode)

        # Unfold and compute median
        out = out.unfold(d, w, 1).median(dim=-1).values

    return out


def generate_truncated_exponential(t, params):
    """
    Generate truncated exponential curve from fit parameters.

    Model: y = A * exp(-(t - t0) * k) + C, for t >= t0.
           y = C, for t < t0.

    Args:
        t (array-like): 1D array of time points.
        params (dict): Dictionary with keys {"A", "k", "C", "t0"}.

    Returns:
        np.ndarray: Model values for each x.
    """
    A, k, C, t0 = params["A"], params["k"], params["C"], params["t0"]

    t = torch.as_tensor(t)

    y = torch.where(
        t >= t0,
        A * torch.exp(-(t - t0) * k) + C,
        torch.ones_like(t) * C
    )

    return y


def plot_dataset(x, y, color = 'C0', linestyle = 'solid', marker = None, sharex = True, sharey = True, figsize = None,
                 xlabel = 'Time (ns)', ylabel = 'Intensity', fig = None, ax = None):
    """
    Plot all channels of a 2D dataset in a single figure.

    Args:
        x (np.ndarray): 1D array of shape (T,) where T is number of samples.
        y (np.ndarray): 2D array of shape (T, C) where T is number of samples and C is number of channels.
        color (str, optional): Line color. Default is 'C0'.
        linestyle (str, optional): Line style. Default is 'solid'.
        marker (str, optional): Marker style. Default is None (no markers).
        sharex (bool, optional): Whether to share x-axis among subplots. Default is True.
        sharey (bool, optional): Whether to share y-axis among subplots. Default is True.
        figsize (tuple, optional): Figure size as (width, height). Default is None, which lets matplotlib choose.
        xlabel (str, optional): Label for x-axis. Default is 'Time (ns)'.
        ylabel (str, optional): Label for y-axis. Default is 'Intensity'.
        fig (optional): Matplotlib figure to plot on. If None, a new figure is created.
        ax (optional): Matplotlib axes array to plot on. If None, a new figure is created.

    Returns:
        fig, ax: Matplotlib figure and axes array.
    """

    if np.ndim(y) != 2:
        raise ValueError("y must be a 2D array")

    T, C = y.shape
    nrows = int(np.ceil(np.sqrt(C)))
    ncols = int(np.ceil(C / nrows))

    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey, figsize=figsize)

    ax = np.array(ax).reshape(-1)

    for c in range(C):
        ax[c].plot(x, y[:, c], color=color, linestyle=linestyle, marker=marker)
        ax[c].set_title(f"Channel {c}")
        if c % ncols == 0:
            ax[c].set_ylabel(ylabel)
        if c // ncols == nrows - 1:
            ax[c].set_xlabel(xlabel)

    for c in range(C, len(ax)):
        ax[c].axis('off')

    fig.tight_layout()

    return fig, ax


def partial_convolution(volume: torch.tensor, kernel: torch.tensor, dim1: str = 'ijk', dim2: str = 'jkl',
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


def estimate_lifetime(x: torch.Tensor, y: torch.Tensor, t0: int, t1: int) -> float:
    """
    Estimate initial decay rate k from centroid of baseline-subtracted signal.

    Args:
        x (torch.Tensor): 1D tensor of time points (same length as y).
        y (torch.Tensor): 1D tensor of signal values.

    Returns:
        float: Initial estimate for k (positive).
    """

    x = torch.as_tensor(x[t0:t1+1], dtype= torch.float32)
    x = x - x.min()  # shift to start at 0

    # Baseline subtraction and clamping
    y = torch.as_tensor(y[t0:t1+1], dtype= torch.float32)
    y_clamped = torch.clamp(y - y.min(), min=0.0) # subtract baseline and clamp to zero

    # Centroid-based lifetime estimation
    tau = torch.sum(x * y_clamped) / torch.sum(y_clamped)

    return tau