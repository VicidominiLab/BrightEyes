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


