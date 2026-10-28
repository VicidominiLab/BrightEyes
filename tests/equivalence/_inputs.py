import numpy as np

CHANNELS = 25
SIDE = 32


def ism_dataset() -> np.ndarray:
    """Return a non-negative `(32, 32, 25)` float64 ISM dataset."""
    return np.random.default_rng(0).random((SIDE, SIDE, CHANNELS))


def ism_psf() -> np.ndarray:
    """Return a `(32, 32, 25)` float64 PSF stack, each channel summing to 1."""
    psf = np.random.default_rng(0).random((SIDE, SIDE, CHANNELS))
    return psf / psf.sum(axis=(0, 1))


def photon_arrival_times() -> dict[str, np.ndarray]:
    """Return arrival times as `{"det0": ..., "det24": ...}`, one `(400, 2)` int64 array per channel.

    The columns are the sorted macrotime and the microtime, the layout of the
    `detN` attributes of the FFS arrival-time data object.
    """
    rng = np.random.default_rng(0)
    times = {}
    for channel in range(CHANNELS):
        macro = np.sort(rng.integers(0, 10**6, 400))
        micro = rng.integers(0, 256, 400)
        times[f"det{channel}"] = np.column_stack((macro, micro)).astype(np.int64)
    return times


def time_trace() -> np.ndarray:
    """Return a `(10000, 25)` uint32 array of photon counts per time bin and channel."""
    return np.random.default_rng(0).poisson(2.0, (10000, CHANNELS)).astype(np.uint32)


def decay_histogram() -> np.ndarray:
    """Return a `(25, 256)` int64 array of photon counts per channel and time bin."""
    return np.random.default_rng(0).poisson(10.0, (CHANNELS, 256)).astype(np.int64)


def correlation_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return `(tau, G)`: lag times `(60,)` and correlations `(60, 25)`, both float64."""
    tau = np.logspace(-6, 0, 60)
    amplitude = np.random.default_rng(0).random(CHANNELS) + 0.5
    return tau, amplitude / (1.0 + tau[:, None] / 1e-3)
