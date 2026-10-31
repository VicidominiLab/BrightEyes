import numpy as np

SIZE = 32
CHANNELS = 25


def ism_dataset() -> np.ndarray:
    """Return a small `(x, y, channels)` array of photon counts."""
    return np.random.default_rng(0).poisson(10.0, size=(SIZE, SIZE, CHANNELS)).astype(np.float64)


def ism_psf() -> np.ndarray:
    """Return a `(x, y, channels)` array holding one shifted Gaussian per channel."""
    axis = np.arange(SIZE) - SIZE / 2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    shifts = np.random.default_rng(0).uniform(-3, 3, size=(CHANNELS, 2))
    psf = np.stack([np.exp(-((x - dx) ** 2 + (y - dy) ** 2) / (2 * 2.0**2)) for dx, dy in shifts], axis=-1)
    return psf / psf.sum(axis=(0, 1), keepdims=True)


def photon_arrival_times() -> np.ndarray:
    """Return about 10^4 sorted integer arrival times."""
    return np.sort(np.random.default_rng(0).integers(0, 10**7, size=10_000)).astype(np.int64)


def time_trace() -> np.ndarray:
    """Return a `(time, channels)` array of photon counts."""
    return np.random.default_rng(0).poisson(5.0, size=(10_000, CHANNELS)).astype(np.float64)


def decay_histogram() -> np.ndarray:
    """Return a noisy single-exponential photon arrival histogram."""
    time = np.arange(256)
    return np.random.default_rng(0).poisson(1000.0 * np.exp(-time / 40.0)).astype(np.float64)


def correlation_curve() -> tuple[np.ndarray, np.ndarray]:
    """Return lag times and a noisy diffusion correlation curve."""
    tau = np.logspace(-6, 0, 50)
    noise = np.random.default_rng(0).normal(0, 1e-3, size=tau.shape)
    return tau, 1.0 / (1.0 + tau / 1e-3) + noise


def ism_series() -> np.ndarray:
    """Return a `(x, y, time, channels)` array of photon counts of a smooth sample."""
    rng = np.random.default_rng(0)
    axis = np.arange(SIZE)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    sample = sum(
        np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0**2)) for cx, cy in rng.uniform(6, SIZE - 6, size=(4, 2))
    )
    detector = np.arange(CHANNELS)
    radius2 = (detector // 5 - 2) ** 2 + (detector % 5 - 2) ** 2
    weights = np.exp(-radius2 / (2 * 1.2**2)) * rng.uniform(0.9, 1.1, size=CHANNELS)
    mean = 400.0 * sample[:, :, None, None] * weights[None, None, None, :] + 2.0
    return rng.poisson(np.broadcast_to(mean, (SIZE, SIZE, 4, CHANNELS))).astype(np.float64)


def _flim_decays(seed: int, tau: float, lag: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bins = 64
    time = np.arange(bins)[:, None]
    lags = lag * (np.arange(CHANNELS) % 5)[None, :]
    decay = np.exp(-np.mod(time - lags, bins) / tau)
    amplitude = rng.uniform(5.0, 30.0, size=(8, 8))
    return rng.poisson(amplitude[:, :, None, None] * decay[None, None]).astype(np.float64)


def flim_stack() -> np.ndarray:
    """Return an `(x, y, time, channels)` array of photon counts with a decay per pixel."""
    return _flim_decays(0, tau=8.0, lag=1.0)


def flim_reference_stack() -> np.ndarray:
    """Return a `flim_stack`-shaped array of a reference sample with another lifetime and delays."""
    return _flim_decays(0, tau=5.0, lag=2.0)


def photon_streams() -> dict[int, np.ndarray]:
    """Return a `(photons, 2)` array of macro and micro times in ps for each of 25 channels."""
    rng = np.random.default_rng(0)
    return {
        channel: np.column_stack(
            (
                np.sort(rng.integers(0, 200_000, size=400)).astype(np.float64),
                rng.integers(0, 25_000, size=400).astype(np.float64),
            )
        )
        for channel in range(CHANNELS)
    }


def time_channel_array() -> np.ndarray:
    """Return a `(time, 1 + channels)` array of sorted unique times and counts of 5 channels."""
    rng = np.random.default_rng(0)
    times = np.sort(rng.choice(20_000, size=1500, replace=False)).astype(np.float64)
    return np.column_stack((times, rng.poisson(0.4, size=(1500, 5)).astype(np.float64)))
