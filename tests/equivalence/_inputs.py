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


FLIM_BINS = 64
FLIM_PERIOD_NS = 25.0


def flim_time_axis() -> np.ndarray:
    """Return the `(64,)` float64 time axis in ns, spanning one 25 ns excitation period."""
    return np.arange(FLIM_BINS) * FLIM_PERIOD_NS / FLIM_BINS


def flim_decay(tau_ns: float = 3.0, shift_bins: float = 5.0) -> np.ndarray:
    """Return a `(64,)` float64 periodic mono-exponential decay with its onset at `shift_bins`."""
    t = (flim_time_axis() - shift_bins * FLIM_PERIOD_NS / FLIM_BINS) % FLIM_PERIOD_NS
    decay = np.exp(-t / tau_ns)
    return 1000.0 * decay / decay.sum()


def flim_decay_histogram() -> np.ndarray:
    """Return a `(64, 25)` float64 Poisson decay histogram, channel `c` delayed by `c % 5` bins."""
    rng = np.random.default_rng(0)
    return np.stack(
        [
            rng.poisson(flim_decay(3.0, 5.0 + c % 5) * 3).astype(float)
            for c in range(CHANNELS)
        ],
        axis=1,
    )


def flim_image(side: int = 6, tau_ns: float = 3.0) -> np.ndarray:
    """Return a `(side, side, 64, 25)` float64 image of Poisson decays.

    The lifetime grows by 0.1 ns per pixel column and channel `c` is delayed by `c % 5` bins.
    """
    rng = np.random.default_rng(1)
    image = np.empty((side, side, FLIM_BINS, CHANNELS))
    for y in range(side):
        for x in range(side):
            for c in range(CHANNELS):
                mean = (
                    flim_decay(tau_ns + 0.1 * x, 5.0 + c % 5 + 0.3 * c / CHANNELS) * 0.2
                )
                image[y, x, :, c] = rng.poisson(mean)
    return image
