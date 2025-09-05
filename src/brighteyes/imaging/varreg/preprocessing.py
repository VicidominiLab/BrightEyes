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

    n_rep = dset.shape[channel]
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

import torch
import numpy as np

def binomial_split(x, n_splits=2, p=0.5, device=None):
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
    # n_splits must be a power of 2 (e.g. 2, 4, 8, 16...)
    if n_splits < 1 or (n_splits & (n_splits - 1)) != 0:
        raise ValueError("n_splits must be a positive power of 2")

    # Convert NumPy input to torch tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    # Select device: use given one, else GPU if available, else CPU
    if device is None:
        device = x.device if x.device.type != 'cpu' else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)

    # Ensure input is non-negative
    if not torch.all(x >= 0):
        raise ValueError("x must be non-negative")

    # Convert to float for torch.binomial
    x = x.to(torch.float32)

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



