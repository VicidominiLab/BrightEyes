from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SIZE = 32
CHANNELS = 25


def ism_dataset() -> np.ndarray:
    """Return a small `(x, y, channels)` array of photon counts."""
    return (
        np.random.default_rng(0)
        .poisson(10.0, size=(SIZE, SIZE, CHANNELS))
        .astype(np.float64)
    )


def ism_psf() -> np.ndarray:
    """Return a `(x, y, channels)` array holding one shifted Gaussian per channel."""
    axis = np.arange(SIZE) - SIZE / 2
    x, y = np.meshgrid(axis, axis, indexing="ij")
    shifts = np.random.default_rng(0).uniform(-3, 3, size=(CHANNELS, 2))
    psf = np.stack(
        [np.exp(-((x - dx) ** 2 + (y - dy) ** 2) / (2 * 2.0**2)) for dx, dy in shifts],
        axis=-1,
    )
    return psf / psf.sum(axis=(0, 1), keepdims=True)


def photon_arrival_times() -> np.ndarray:
    """Return about 10^4 sorted integer arrival times."""
    return np.sort(np.random.default_rng(0).integers(0, 10**7, size=10_000)).astype(
        np.int64
    )


def time_trace() -> np.ndarray:
    """Return a `(time, channels)` array of photon counts."""
    return (
        np.random.default_rng(0)
        .poisson(5.0, size=(10_000, CHANNELS))
        .astype(np.float64)
    )


def decay_histogram() -> np.ndarray:
    """Return a noisy single-exponential photon arrival histogram."""
    time = np.arange(256)
    return (
        np.random.default_rng(0)
        .poisson(1000.0 * np.exp(-time / 40.0))
        .astype(np.float64)
    )


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
        np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 3.0**2))
        for cx, cy in rng.uniform(6, SIZE - 6, size=(4, 2))
    )
    detector = np.arange(CHANNELS)
    radius2 = (detector // 5 - 2) ** 2 + (detector % 5 - 2) ** 2
    weights = np.exp(-radius2 / (2 * 1.2**2)) * rng.uniform(0.9, 1.1, size=CHANNELS)
    mean = 400.0 * sample[:, :, None, None] * weights[None, None, None, :] + 2.0
    return rng.poisson(np.broadcast_to(mean, (SIZE, SIZE, 4, CHANNELS))).astype(
        np.float64
    )


def _flim_decays(seed: int, tau: float, lag: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bins = 64
    time = np.arange(bins)[:, None]
    lags = lag * (np.arange(CHANNELS) % 5)[None, :]
    decay = np.exp(-np.mod(time - lags, bins) / tau)
    amplitude = rng.uniform(5.0, 30.0, size=(8, 8))
    return rng.poisson(amplitude[:, :, None, None] * decay[None, None]).astype(
        np.float64
    )


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


def pch_psf() -> np.ndarray:
    """Return a cubic Gaussian focal volume of 12 voxels per side with its maximum at 1."""
    axis = np.arange(12) - 6
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.exp(-2 * (x**2 + y**2) / 3.0**2 - 2 * z**2 / 5.0**2)


def bead_image() -> np.ndarray:
    """Return a `(x, y, channels)` array holding five Gaussian beads on a dark background."""
    axis = np.arange(64)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    beads = np.array([[16, 16], [16, 44], [32, 30], [46, 14], [48, 48]])
    spot = sum(
        np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5**2)) for cx, cy in beads
    )
    weights = np.random.default_rng(0).uniform(0.5, 1.0, size=4)
    return np.round(200 * spot[:, :, None] * weights[None, None, :])


def mcs_h5_file(path: Path) -> Path:
    """Write a small current-schema MCS file with `/raw/spad` and `/raw/aux`, and return its path."""
    rng = np.random.default_rng(0)
    shape = (1, 1, 4, 5, 8)
    spad = rng.poisson(3.0, size=(*shape, 5)).astype(np.uint16)
    aux = rng.poisson(2.0, size=(*shape, 2)).astype(np.uint16)
    time_bin_ns = 0.5
    with h5py.File(path, "w") as handle:
        handle.attrs.update(
            data_format_version="0.0.6",
            schema_name="brighteyes_mcs_file",
            default="/raw/spad",
            data_path="/raw",
            metadata_path="/raw/metadata",
            axes_path="/raw/axes",
            legacy_path="/raw/legacy",
            calibration_path="/calibration",
            output_path="/output",
            contains_output=False,
        )
        raw = handle.create_group("raw")
        for name, data, channel_axis in (
            ("spad", spad, "detector_channel"),
            ("aux", aux, "aux_channel"),
        ):
            dataset = raw.create_dataset(name, data=data)
            dataset.attrs.update(
                axis_order=f"repetition,z,y,x,time_bin,{channel_axis}",
                time_axis_path="/raw/axes/digital_time_ns",
                metadata_path="/raw/metadata",
                units="counts",
            )
        metadata = raw.create_group("metadata")
        metadata.attrs.update(
            time_bin_ns=time_bin_ns,
            digital_time_bin_ns=time_bin_ns,
            laser_frequency_mhz=80.0,
            nrep=shape[0],
            nz=shape[1],
            ny=shape[2],
            nx=shape[3],
            time_bins=shape[4],
        )
        axes = raw.create_group("axes")
        axes.create_dataset("digital_time_ns", data=np.arange(shape[4]) * time_bin_ns)
        raw.create_group("legacy")
        handle.create_group("calibration")
    return path


def ttm_new_protocol_bytes() -> bytes:
    """Return a uint16 stream of 200 events with channel, dummy, laser, pixel, scan and line words and `0x7FFF` fillers."""
    rng = np.random.default_rng(0)
    words = []
    for event in range(200):
        for channel in rng.choice([0, 3, 7, 24, 50], size=rng.integers(0, 4)):
            valid = int(rng.random() < 0.7)
            words.append(valid << 15 | int(channel) << 8 | int(rng.integers(0, 256)))
        for word_id, enable_probability in (
            (123, 0.3),
            (124, 0.0),
            (125, 0.2),
            (126, 0.05 if event else 1.0),
            (127, 0.1),
        ):
            valid = int(rng.random() < 0.5)
            enable = int(rng.random() < enable_probability) << 7
            words.append(
                valid << 15 | word_id << 8 | enable | int(rng.integers(0, 127))
            )
        words += [0x7FFF] * int(rng.integers(0, 3))
    return np.array(words, dtype=np.uint16).tobytes()


def ttm_raw_frame() -> pd.DataFrame:
    """Return 3000 parsed time-tagger rows with photons on channels 0 and 3, laser flags, and scan, line and pixel edges."""
    rng = np.random.default_rng(0)
    rows = 3000
    cumulative_step = np.cumsum(rng.integers(1, 200, size=rows)).astype(np.int64)
    row = np.arange(rows)
    frame = pd.DataFrame(
        {
            "t_L": rng.integers(0, 256, size=rows),
            "t_0": rng.integers(0, 256, size=rows),
            "t_3": rng.integers(0, 256, size=rows),
            "valid_tdc_0": rng.random(rows) < 0.3,
            "valid_tdc_3": rng.random(rows) < 0.3,
            "valid_tdc_L": rng.random(rows) < 0.4,
            "scan_enable": row % 1000 < 10,
            "line_enable": (row % 100 >= 20) & (row % 100 < 25),
            "pixel_enable": row % 10 == 3,
        }
    ).astype(np.uint8)
    frame["step"] = (cumulative_step % 2**16).astype(np.uint16)
    frame["cumulativeStep"] = cumulative_step
    return frame


def flus2ism_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Return the central 21 by 21 pixels of the example PSF `(z, x, y, channels)` and phantom `(structures, x, y)`."""
    folder = Path(__file__).parents[2] / "examples" / "imaging" / "flus2ism"
    psf = np.load(folder / "PSFs.npy")[:, 40:61, 40:61]
    phantom = np.load(folder / "simulated_tubulin.npz")["phantom"][:, 40:61, 40:61]
    return psf, phantom
