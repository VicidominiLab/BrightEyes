"""Centroids and split-gradient terms for the supported RbN regularizers.

Array convention
----------------
The final axis is always ``K``, the number of independent reconstructions.
All preceding axes are object axes, typically ``(Nz, Nx, Ny)``. Functions in
this module preserve the input shape so their outputs broadcast directly in
the multiplicative or scaled-gradient updates.
"""

import torch


def _positive_eps(obj_set: torch.Tensor, eps: float = None) -> float:
    """Return a positive numerical offset compatible with ``obj_set``."""
    if eps is None:
        dtype = obj_set.real.dtype if torch.is_complex(obj_set) else obj_set.dtype
        eps = torch.finfo(dtype).eps
    if eps <= 0:
        raise ValueError("eps must be positive.")
    return eps


def compute_centroid(obj_set: torch.Tensor, regularizer: str = 'l2', eps: float = None) -> torch.Tensor:
    """Compute the point-wise centroid across the realization axis ``K``.

    Parameters
    ----------
    obj_set : torch.Tensor
        Reconstruction ensemble shaped ``(..., K)``, normally
        ``(Nz, Nx, Ny, K)`` or ``(Niter, Nz, Nx, Ny, K)``.
    regularizer : {'l2', 'l1', 'kl'}, default='l2'
        Distance used to define the centroid. ``l2`` uses the arithmetic mean,
        ``l1`` the median, and ``kl`` the geometric mean associated with the
        generalized KL divergence.
    eps : float or None, default=None
        Positive floor used only by the KL centroid.

    Returns
    -------
    torch.Tensor
        Point-wise centroid with shape ``obj_set.shape[:-1] + (1,)``.
    """
    obj_set = torch.as_tensor(obj_set)
    regularizer = regularizer.lower()

    if regularizer == 'l2':
        return obj_set.mean(dim=-1, keepdim=True)
    if regularizer == 'l1':
        if torch.is_complex(obj_set):
            raise ValueError("The L1 centroid is not defined for complex-valued tensors.")
        return obj_set.median(dim=-1, keepdim=True).values
    if regularizer == 'kl':
        if torch.is_complex(obj_set):
            raise ValueError("The KL centroid requires a real non-negative tensor.")
        if torch.any(obj_set < 0):
            raise ValueError("The KL centroid requires non-negative values.")
        eps = _positive_eps(obj_set, eps)
        return torch.exp(torch.log(obj_set.clamp_min(eps)).mean(dim=-1, keepdim=True))

    raise ValueError("Regularizer must be 'l1', 'l2', or 'kl'.")


def grad_l2(obj_set: torch.Tensor):
    """Split the L2-gradient as ``v - u = 2 * (obj_set - mean)``.

    Parameters
    ----------
    obj_set : torch.Tensor
        Reconstruction ensemble shaped ``(..., K)``.

    Returns
    -------
    u, v : tuple[torch.Tensor, torch.Tensor]
        Non-negative terms with the same broadcastable shape as ``obj_set``.
    """
    obj_mean = compute_centroid(obj_set, 'l2')
    return 2 * obj_mean, 2 * obj_set


def grad_l1(obj_set: torch.Tensor):
    """Return a smooth split-gradient approximation of the L1 penalty.

    The difference ``v - u`` equals
    ``(obj_set - median) / sqrt((obj_set - median)**2 + eps**2)``.

    Parameters
    ----------
    obj_set : torch.Tensor
        Real reconstruction ensemble shaped ``(..., K)``.

    Returns
    -------
    u, v : tuple[torch.Tensor, torch.Tensor]
        Positive and negative gradient components, each shaped as ``obj_set``.
    """
    obj_set = torch.as_tensor(obj_set)
    eps = _positive_eps(obj_set)
    obj_median = compute_centroid(obj_set, 'l1')
    denominator = torch.sqrt((obj_set - obj_median)**2 + eps**2)
    return obj_median / denominator, obj_set / denominator


def grad_kl(obj_set: torch.Tensor):
    """Split the generalized KL gradient as ``v - u = 1 - centroid / obj``.

    Parameters
    ----------
    obj_set : torch.Tensor
        Non-negative reconstruction ensemble shaped ``(..., K)``.

    Returns
    -------
    u, v : tuple[torch.Tensor, torch.Tensor]
        Positive and negative gradient components, each shaped as ``obj_set``.
    """
    obj_set = torch.as_tensor(obj_set)
    eps = _positive_eps(obj_set)
    obj_centroid = compute_centroid(obj_set, 'kl', eps=eps)
    u = obj_centroid / obj_set.clamp_min(eps)
    v = torch.ones_like(obj_set)
    return u, v