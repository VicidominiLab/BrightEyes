import torch
import torch.nn.functional as F
import numpy as np

def median_filter(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Apply a median filter along the last dimension of a tensor.

    Args:
        x (torch.Tensor): Input tensor of any shape.
        kernel_size (int): Size of the median window (must be odd).

    Returns:
        torch.Tensor: Median-filtered tensor (same shape as input).
    """
    if kernel_size <= 1:
        return x.clone()

    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    # Pad reflectively along the last dimension
    pad = kernel_size // 2
    x_padded = F.pad(x.unsqueeze(0).unsqueeze(0), (pad, pad), mode="reflect")[0, 0]

    # Use unfold to create sliding windows along last dim
    unfolded = x_padded.unfold(-1, kernel_size, 1)  # [..., L, kernel_size]

    # Take median across window dimension
    return unfolded.median(dim=-1).values


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
    y = torch.where(t >= t0, A * np.exp(-(t - t0) * k ) + C, C)

    return y