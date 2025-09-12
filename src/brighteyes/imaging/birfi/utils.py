import torch
import numpy as np


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