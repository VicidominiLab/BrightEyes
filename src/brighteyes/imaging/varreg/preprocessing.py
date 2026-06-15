import numpy as np
import torch
import math

def divisors(n):
    """
    Generate all divisors of a given integer n.
    Parameters:
        n (int): positive integer number.

    Returns:
        List of divisors.
    """

    divs = [1]
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divs.extend([i, n // i])
    divs.extend([n])
    return list(sorted(set(divs)))


def rebin(dset, nbin, length, channel: int = -1):
    """
    Rebin a dataset by averaging over specified bins.

    Parameters:
        dset (ndarray): Input dataset of shape Nx x Ny x K.
        nbin (int): Number of bins to average over.
        length (int): Length of the rebinned dimension.
        channel (int): Channel index to rebin. Default is -1 (last channel).

    Returns:
        ndarray: Rebinned dataset of shape Nx x Ny x length.
    """

    dset = np.moveaxis(dset, channel, -1)
    n_rep = dset.shape[-1]
    integer_part = ( n_rep // (nbin * length) ) * (nbin * length)
    remainder = n_rep - integer_part

    if integer_part == 0:
        raise ValueError("The number of bins is too large for the dataset.")
    else:
        dataset = dset[..., :n_rep - remainder]

    new_size = dataset.shape[:-1] + (nbin, length)
    dataset = dataset.reshape( new_size )
    dataset = dataset.sum(axis=-1)

    return dataset


def binomial_split(x: torch.tensor, n_splits: int = 2, p: float = 0.5, device: str = None):
    """
    Binomial split into `n_splits` parts (must be a power of 2).

    Args:
        x (torch.Tensor or np.ndarray): Input non-negative tensor/array.
        n_splits (int): Number of splits (must be a power of 2).
        p (float): Split probability at each stage.
        device: Optional device (defaults to x.device or 'cuda' if available).

    Returns:
        torch.Tensor: Tensor of shape (*x.shape, n_splits) with integer splits.
    """
    # ---- Check input validity ----
    # n_splits must be a power of 2
    if n_splits < 1 or (n_splits & (n_splits - 1)) != 0:
        raise ValueError("n_splits must be a positive power of 2")

    # Convert NumPy input to torch tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # Select device: use given one, else GPU if available, else CPU
    if device is None:
        device = x.device if x.device.type != 'cpu' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    # Convert to float for torch.binomial
    x = x.to(torch.float32)

    # Ensure input is non-negative
    if not torch.all(x >= 0):
        raise ValueError("x must be non-negative")

    # Original shape
    shape = x.shape

    # Start with tensor expanded along last dimension to hold n_splits parts
    # Example: if x.shape = (m, n) and n_splits=8 → shape = (m, n, 8)
    splits = x.unsqueeze(-1).expand(*shape, n_splits).clone()

    # Number of binary split stages = log2(n_splits)
    depth = n_splits.bit_length() - 1

    # Create an index mask [0, 1, ..., n_splits-1]
    # Each index encodes the path in the binary split tree (in binary form)
    mask = torch.arange(n_splits, device=device).unsqueeze(0)  # shape (1, n_splits)

    # ---- Perform vectorized splits ----
    for d in range(depth):
        # Extract bit at position (depth - 1 - d) for each split index
        # 0 = "go left", 1 = "go right"
        left_mask = ((mask >> (depth - 1 - d)) & 1) == 0

        # Probability of going left vs right at this depth
        probs = torch.where(left_mask, p, 1 - p)

        # Perform binomial split for all elements at this stage
        # Each split tensor entry gets randomly reduced according to probs
        splits = torch.binomial(splits, probs)

    # Convert back to integer type
    return splits.to(torch.int32)

import numpy as np

def make_noisy_dataset(data, bins_to_sum, K, L, seed=None, replace=False):
    """
    Build a dataset with shape (Nx, Ny, Ch, K, L) from input data of shape
    (Nx, Ny, Ch, Nbin), by summing bins_to_sum temporal bins for each of K
    noise realizations, repeated L times.

    Parameters
    ----------
    data : np.ndarray
        Input array with shape (Nx, Ny, Ch, Nbin).
    bins_to_sum : int
        Number of temporal bins to sum for each realization.
    K : int
        Number of noise realizations per repetition.
    L : int
        Number of repetitions of the experiment.
    seed : int or None
        Random seed for reproducibility.
    replace : bool
        If False, each realization samples bins without replacement.
        If True, bins may repeat within a realization.

    Returns
    -------
    out : np.ndarray
        Array with shape (Nx, Ny, Ch, K, L).
    bins_used : np.ndarray
        Integer array with shape (L, K, bins_to_sum) containing the sampled bins.
    """
    if data.ndim != 4:
        raise ValueError(f"Expected data with shape (Nx, Ny, Ch, Nbin), got {data.shape}")

    Nx, Ny, Ch, Nbin = data.shape
    if not replace and K * bins_to_sum > Nbin:
        raise ValueError(
            f"Need K*bins_to_sum <= Nbin when sampling without replacement, "
            f"but got K={K}, bins_to_sum={bins_to_sum}, Nbin={Nbin}"
        )

    rng = np.random.default_rng(seed)
    out = np.empty((Nx, Ny, Ch, K, L), dtype=data.dtype)
    bins_used = np.empty((L, K, bins_to_sum), dtype=int)

    for l in range(L):
        if replace:
            for k in range(K):
                idx = rng.integers(0, Nbin, size=bins_to_sum)
                bins_used[l, k] = idx
                out[..., k, l] = data[..., idx].sum(axis=-1)
        else:
            perm = rng.permutation(Nbin)
            selected = perm[:K * bins_to_sum].reshape(K, bins_to_sum)
            bins_used[l] = selected
            for k in range(K):
                out[..., k, l] = data[..., selected[k]].sum(axis=-1)

    return out, bins_used

