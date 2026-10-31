"""Preprocessing utilities for temporal rebinning and photon-count splitting.

Array-axis convention used throughout this module:
    Nx, Ny : lateral image dimensions
    Nch    : detector or measurement channels
    Nbin   : temporal acquisition bins
    K      : independent noise realizations
    L      : repeated experiments or independent groups

Unless stated otherwise, split functions append the realization axis as the last axis.
"""

import math
from typing import Optional, Sequence, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]
DeviceLike = Optional[Union[str, torch.device]]


def divisors(n: int) -> list[int]:
    """Return all positive divisors of ``n`` in ascending order.

    Parameters
    ----------
    n : int
        Positive integer.

    Returns
    -------
    list[int]
        Sorted list containing every positive divisor of ``n``.
    """
    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n // i])
    divs.extend([n])
    return list(sorted(set(divs)))


def rebin(dset: np.ndarray, nbin: int, length: int, channel: int = -1) -> np.ndarray:
    """Sum contiguous samples along one axis into ``nbin`` output bins.

    The selected axis is moved to the last position, truncated to a complete
    ``(nbin, length)`` block, reshaped, and summed over ``length``. The selected
    axis is therefore returned as the last axis.

    Parameters
    ----------
    dset : np.ndarray
        Input array with arbitrary leading dimensions and a rebinning axis of
        length ``Nrep``.
    nbin : int
        Number of output bins, denoted ``K`` in the reconstruction workflow.
    length : int
        Number of consecutive input samples summed into each output bin.
    channel : int, default=-1
        Axis to rebin. The name is retained for backward compatibility; this is
        an axis index rather than a detector-channel value.

    Returns
    -------
    np.ndarray
        Rebinned array with shape ``(*leading_shape, nbin)``.
    """
    dset = np.moveaxis(dset, channel, -1)
    n_rep = dset.shape[-1]
    integer_part = (n_rep // (nbin * length)) * (nbin * length)
    remainder = n_rep - integer_part

    if integer_part == 0:
        raise ValueError("The number of bins is too large for the dataset.")
    dataset = dset[..., :n_rep - remainder]

    new_size = dataset.shape[:-1] + (nbin, length)
    dataset = dataset.reshape(new_size)
    dataset = dataset.sum(axis=-1)
    return dataset


def binomial_split(x: ArrayLike, n_splits: int = 2, p: float = 0.5, device: DeviceLike = None) -> torch.Tensor:
    """Generate ``n_splits`` recursively binomial-thinned count images.

    ``n_splits`` must be a power of two. Each output follows one path through a
    binary thinning tree and is stored along a new final axis. For ``p=0.5``,
    each output has marginal thinning probability ``1 / n_splits``.

    Notes
    -----
    This routine does **not** partition the observed counts: the outputs are
    sampled independently from replicated count tensors and their sum is not
    constrained to equal the input. Use :func:`multinomial_split` when exact
    photon-count conservation is required.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Non-negative count array with arbitrary shape ``(*shape,)``.
    n_splits : int, default=2
        Number of outputs. It must be a positive power of two.
    p : float, default=0.5
        Left-branch probability applied at every binary split stage.
    device : str or torch.device, optional
        Output device. If omitted, an existing non-CPU tensor device is kept;
        otherwise CUDA is selected when available, falling back to CPU.

    Returns
    -------
    torch.Tensor
        Integer tensor with shape ``(*x.shape, n_splits)`` and dtype
        ``torch.int32``.
    """
    if n_splits < 1 or (n_splits & (n_splits - 1)) != 0:
        raise ValueError("n_splits must be a positive power of 2.")
    if not 0 <= p <= 1:
        raise ValueError("p must lie in the interval [0, 1].")

    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if device is None:
        device = x.device if x.device.type != "cpu" else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    x = x.to(device=device, dtype=torch.float32)
    if not torch.all(x >= 0):
        raise ValueError("x must be non-negative.")

    shape = x.shape
    splits = x.unsqueeze(-1).expand(*shape, n_splits).clone()  # (*shape, n_splits)
    depth = n_splits.bit_length() - 1
    mask = torch.arange(n_splits, device=device).unsqueeze(0)  # (1, n_splits)

    for d in range(depth):
        left_mask = ((mask >> (depth - 1 - d)) & 1) == 0
        probs = torch.where(left_mask, p, 1 - p)
        splits = torch.binomial(splits, probs)

    return splits.to(torch.int32)


def multinomial_split(
    x: ArrayLike,
    n_splits: Optional[int] = None,
    probabilities: Optional[Sequence[float]] = None,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.int64,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Partition integer photon counts among mutually exclusive realizations.

    The multinomial draw is implemented as a sequence of conditional binomial
    draws. This avoids expanding every individual photon while preserving the
    exact count at each input element.

    Parameters
    ----------
    x : np.ndarray or torch.Tensor
        Non-negative integer counts with arbitrary shape ``(*shape,)``.
    n_splits : int, optional
        Number of output realizations ``K``. Defaults to 2 when
        ``probabilities`` is omitted. If probabilities are provided, it must
        match their length.
    probabilities : sequence of float, optional
        Category probabilities. They are normalized to sum to one. Uniform
        probabilities are used when omitted.
    device : str or torch.device, optional
        Output device. By default, the input device is preserved.
    dtype : torch.dtype, default=torch.int64
        Output dtype.
    generator : torch.Generator, optional
        Torch random-number generator used by the binomial draws.

    Returns
    -------
    torch.Tensor
        Count-conserving split with shape ``(*x.shape, n_splits)``. For every
        input element, ``splits.sum(-1) == x`` exactly after integer conversion.
    """
    if isinstance(x, np.ndarray):
        if x.dtype == np.uint64:
            if np.any(x > np.iinfo(np.int64).max):
                raise OverflowError("x contains uint64 values that cannot be represented as int64.")
            x = x.astype(np.int64, copy=False)
        x = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    if x.dtype == torch.uint64:
        x_cpu = x.cpu() if x.device.type != "cpu" else x
        if np.any(x_cpu.numpy() > np.iinfo(np.int64).max):
            raise OverflowError("x contains uint64 values that cannot be represented as int64.")
        x = x.to(torch.int64)

    device = x.device if device is None else torch.device(device)
    x = x.to(device)

    if x.is_complex():
        raise TypeError("x must contain real-valued counts.")
    if not torch.all(torch.isfinite(x)):
        raise ValueError("x contains NaN or infinite values.")
    if torch.any(x < 0):
        raise ValueError("x must contain non-negative values.")
    if x.is_floating_point() and not torch.all(x == torch.round(x)):
        raise ValueError("x must contain integer counts.")

    x_integer = torch.round(x).to(torch.int64)

    if probabilities is None:
        if n_splits is None:
            n_splits = 2
        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits must be a positive integer.")
        probs = torch.full((n_splits,), 1.0 / n_splits, dtype=torch.float64, device=device)
    else:
        probs = torch.as_tensor(probabilities, dtype=torch.float64, device=device)
        if probs.ndim != 1:
            raise ValueError("probabilities must be a one-dimensional vector.")
        if probs.numel() < 1:
            raise ValueError("probabilities cannot be empty.")
        if n_splits is not None and n_splits != probs.numel():
            raise ValueError(f"n_splits does not match len(probabilities): {n_splits} != {probs.numel()}.")

        n_splits = probs.numel()
        if not torch.all(torch.isfinite(probs)):
            raise ValueError("probabilities contains NaN or infinite values.")
        if torch.any(probs < 0):
            raise ValueError("probabilities must be non-negative.")

        prob_sum = probs.sum()
        if prob_sum <= 0:
            raise ValueError("The sum of probabilities must be positive.")
        probs = probs / prob_sum

    if n_splits == 1:
        return x_integer.unsqueeze(-1).to(dtype)

    splits = torch.zeros((*x_integer.shape, n_splits), dtype=torch.int64, device=device)
    remaining_counts = x_integer.clone()
    remaining_probability = torch.tensor(1.0, dtype=torch.float64, device=device)

    for idx in range(n_splits - 1):
        current_probability = probs[idx]
        if current_probability <= 0:
            sampled = torch.zeros_like(remaining_counts)
        else:
            conditional_probability = (current_probability / remaining_probability).clamp(0.0, 1.0)
            sampled = torch.binomial(
                remaining_counts.to(torch.float64),
                torch.full_like(remaining_counts, conditional_probability.item(), dtype=torch.float64),
                generator=generator,
            ).to(torch.int64)

        splits[..., idx] = sampled
        remaining_counts = remaining_counts - sampled
        remaining_probability = remaining_probability - current_probability

    splits[..., -1] = remaining_counts
    return splits.to(dtype)


def make_noisy_dataset(
    data: np.ndarray,
    bins_to_sum: int,
    K: int,
    L: int,
    seed: Optional[int] = None,
    replace: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build repeated noise-realization datasets from temporal acquisition bins.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape ``(Nx, Ny, Nch, Nbin)``.
    bins_to_sum : int
        Number of temporal bins summed into each realization.
    K : int
        Number of noise realizations per repeated experiment.
    L : int
        Number of independently sampled repeated experiments.
    seed : int, optional
        Seed for the local NumPy random-number generator.
    replace : bool, default=False
        When False, all ``K * bins_to_sum`` indices within one repetition are
        distinct. When True, indices are drawn independently with replacement.

    Returns
    -------
    out : np.ndarray
        Dataset with shape ``(Nx, Ny, Nch, K, L)``.
    bins_used : np.ndarray
        Selected temporal-bin indices with shape ``(L, K, bins_to_sum)``.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected data with shape (Nx, Ny, Nch, Nbin), got {data.shape}.")
    if not all(isinstance(value, int) and value > 0 for value in (bins_to_sum, K, L)):
        raise ValueError("bins_to_sum, K, and L must be positive integers.")

    Nx, Ny, Nch, Nbin = data.shape
    if not replace and K * bins_to_sum > Nbin:
        raise ValueError(
            f"Need K*bins_to_sum <= Nbin when sampling without replacement, "
            f"but got K={K}, bins_to_sum={bins_to_sum}, Nbin={Nbin}."
        )

    rng = np.random.default_rng(seed)
    out = np.empty((Nx, Ny, Nch, K, L), dtype=data.dtype)
    bins_used = np.empty((L, K, bins_to_sum), dtype=int)

    for l in range(L):
        if replace:
            for k in range(K):
                idx = rng.integers(0, Nbin, size=bins_to_sum)
                bins_used[l, k] = idx
                out[..., k, l] = data[..., idx].sum(axis=-1)
        else:
            selected = rng.permutation(Nbin)[:K * bins_to_sum].reshape(K, bins_to_sum)
            bins_used[l] = selected
            for k in range(K):
                out[..., k, l] = data[..., selected[k]].sum(axis=-1)

    return out, bins_used