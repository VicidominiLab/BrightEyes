import numpy as np
import torch
import math
from typing import Optional, Sequence, Union


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


def multinomial_split(
    x,
    n_splits=None,
    probabilities=None,
    device=None,
    dtype=torch.int64,
    generator=None,
):

    if isinstance(x, np.ndarray):
        if x.dtype == np.uint64:
            if np.any(x > np.iinfo(np.int64).max):
                raise OverflowError(
                    "x contiene valori UInt64 non rappresentabili come int64."
                )

            x = x.astype(np.int64, copy=False)

        x = torch.from_numpy(x)

    elif not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)

    # Gestione anche di input torch.uint64
    if x.dtype == torch.uint64:
        if x.device.type != "cpu":
            x_cpu = x.cpu()
        else:
            x_cpu = x

        if np.any(x_cpu.numpy() > np.iinfo(np.int64).max):
            raise OverflowError(
                "x contiene valori UInt64 non rappresentabili come int64."
            )

        x = x.to(torch.int64)

    if device is None:
        device = x.device
    else:
        device = torch.device(device)

    x = x.to(device)

    if x.is_complex():
        raise TypeError("x deve contenere conteggi reali.")

    if not torch.all(torch.isfinite(x)):
        raise ValueError("x contiene valori NaN o infiniti.")

    if torch.any(x < 0):
        raise ValueError("x deve contenere valori non negativi.")

    if x.is_floating_point():
        if not torch.all(x == torch.round(x)):
            raise ValueError("x deve contenere conteggi interi.")

    x_integer = torch.round(x).to(torch.int64)


    # ------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------
    if probabilities is None:
        if n_splits is None:
            n_splits = 2

        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits deve essere un intero positivo.")

        probs = torch.full(
            (n_splits,),
            fill_value=1.0 / n_splits,
            dtype=torch.float64,
            device=device,
        )

    else:
        probs = torch.as_tensor(
            probabilities,
            dtype=torch.float64,
            device=device,
        )

        if probs.ndim != 1:
            raise ValueError("probabilities deve essere un vettore 1D.")

        if probs.numel() < 1:
            raise ValueError("probabilities non può essere vuoto.")

        if n_splits is not None and n_splits != probs.numel():
            raise ValueError(
                "n_splits non coincide con len(probabilities): "
                f"{n_splits} != {probs.numel()}."
            )

        n_splits = probs.numel()

        if not torch.all(torch.isfinite(probs)):
            raise ValueError(
                "probabilities contiene valori NaN o infiniti."
            )

        if torch.any(probs < 0):
            raise ValueError(
                "Le probabilità devono essere non negative."
            )

        prob_sum = probs.sum()

        if prob_sum <= 0:
            raise ValueError(
                "La somma delle probabilità deve essere positiva."
            )

        probs = probs / prob_sum

    # Caso banale
    if n_splits == 1:
        return x_integer.unsqueeze(-1).to(dtype)

    # ------------------------------------------------------------
    # Multinomial Splitting
    # ------------------------------------------------------------
    output_shape = (*x_integer.shape, n_splits)

    splits = torch.zeros(
        output_shape,
        dtype=torch.int64,
        device=device,
    )

    remaining_counts = x_integer.clone()
    remaining_probability = torch.tensor(
        1.0,
        dtype=torch.float64,
        device=device,
    )

    for idx in range(n_splits - 1):
        current_probability = probs[idx]

        if current_probability <= 0:
            sampled = torch.zeros_like(remaining_counts)

        else:
            # Probabilità condizionata fra le categorie ancora disponibili
            conditional_probability = (
                current_probability / remaining_probability
            ).clamp(0.0, 1.0)

            # torch.binomial richiede un input floating point.
            # float64 evita la perdita di precisione di float32 per
            # conteggi interi relativamente grandi.
            sampled = torch.binomial(
                remaining_counts.to(torch.float64),
                torch.full_like(
                    remaining_counts,
                    fill_value=conditional_probability.item(),
                    dtype=torch.float64,
                ),
                generator=generator,
            ).to(torch.int64)

        splits[..., idx] = sampled
        remaining_counts = remaining_counts - sampled
        remaining_probability = remaining_probability - current_probability

    # L'ultima categoria riceve esattamente i conteggi residui
    splits[..., -1] = remaining_counts

    return splits.to(dtype)


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

    np.random.seed(seed)
    torch.manual_seed(seed)
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

