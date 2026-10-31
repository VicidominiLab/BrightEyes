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
    with np.load(folder / "simulated_tubulin.npz") as data:
        phantom = data["phantom"][:, 40:61, 40:61]
    return psf, phantom


def mcs_output_h5_file(path: Path) -> Path:
    """Write the `mcs_h5_file` layout plus an `/output` group with a sum run, IRF and reference traces, virtual channels and a calibration lifetime, and return its path."""
    mcs_h5_file(path)
    rng = np.random.default_rng(1)
    with h5py.File(path, "a") as handle:
        handle.attrs["contains_output"] = True
        output = handle.create_group("output")
        output.attrs.update(
            default="/output/sum_channels_001/products/spad",
            default_run="/output/sum_channels_001",
            default_irf_trace_id="sum_irf_001",
            default_ref_trace_id="sum_ref_001",
        )
        run = output.create_group("sum_channels_001")
        run.attrs.update(output_type="image_tool", output_id="sum_channels_001")
        product = run.create_group("products").create_dataset(
            "spad", data=rng.poisson(8.0, size=(1, 1, 4, 5, 8)).astype(np.uint32)
        )
        product.attrs.update(
            axis_order="repetition,z,y,x,time_bin",
            time_axis_path="/output/sum_channels_001/axes/time_ns",
            metadata_path="/raw/metadata",
            product_name="spad",
        )
        run.create_group("axes").create_dataset("time_ns", data=np.arange(8) * 0.5)
        for run_id, kind, shape in (
            ("sum_irf_001", "sum_irf_trace", (8,)),
            ("sum_ref_001", "sum_reference_trace", (8, 2)),
        ):
            trace_run = output.create_group(run_id)
            trace = trace_run.create_group("products").create_dataset(
                "trace", data=rng.poisson(20.0, size=shape).astype(np.float64)
            )
            trace.attrs.update(output_type="trace", trace_kind=kind)
            trace_run.attrs["output_type"] = "trace"
        virtual = output.create_group("virtual_channels")
        for kind, channels in (("spad", 5), ("aux", 2)):
            group = virtual.create_group(kind)
            for channel in range(channels):
                dataset = group.create_dataset(
                    f"channel_{channel}",
                    data=rng.poisson(3.0, size=(1, 1, 4, 5, 8)).astype(np.uint16),
                )
                dataset.attrs.update(
                    virtual_channel_type=kind, axis_order="repetition,z,y,x,time_bin"
                )
        fit = handle["calibration"].create_group("results/spad/fit")
        fit.create_dataset("tau_reference_ns", data=np.array([np.nan, 3.5, 4.0]))
    return path


def mcs_flim_h5_file(
    path: Path,
    tau_ns: float = 2.5,
    seed: int = 0,
    silent_channel: int | None = None,
    circular: bool = False,
) -> Path:
    """Write a legacy-layout MCS file of 64 time bins at 40 MHz holding a decay of `tau_ns`, shifted differently on each of 6 SPAD channels and 2 auxiliary channels, and return its path.

    The SPAD channel `silent_channel` holds no counts. With `circular`, the pixels carry circular repetition and point axes (8 dimensions instead of 6).
    """
    rng = np.random.default_rng(seed)
    bins = 64
    period_ns = 1e3 / 40.0
    time = np.arange(bins) * period_ns / bins

    prefix = (1, 1, 2, 2, 2, 3) if circular else (1, 1, 3, 3)

    def counts(delays: list[float]) -> np.ndarray:
        shape = np.stack(
            [
                np.exp(-np.mod(time - delay * period_ns / bins, period_ns) / tau_ns)
                for delay in delays
            ],
            axis=-1,
        )
        rate = 300.0 * shape / shape.sum(axis=0)
        return rng.poisson(np.broadcast_to(rate, (*prefix, bins, len(delays))))

    with h5py.File(path, "w") as handle:
        handle.attrs.update(data_format_version="0.0.4", comment="flim file")
        spad = counts([0.0, 1.0, 2.0, -1.0, 0.5, 1.5]).astype(np.uint16)
        if silent_channel is not None:
            spad[..., silent_channel] = 0
        handle.create_dataset("data", data=spad)
        handle.create_dataset(
            "data_channels_extra", data=counts([0.0, 3.0]).astype(np.uint16)
        )
        gui = handle.create_group("configurationGUI")
        gui.attrs.update(
            range_x=6.0,
            range_y=6.0,
            range_z=1.0,
            timebin_per_pixel=bins,
            time_resolution=0.01,
            nx=3,
            ny=3,
            nframe=1,
            nrep=1,
            bitFile="C:/bits/DFD_40M64.bit",
        )
        handle.create_group("configurationFPGA").attrs.update(DFD_Activate=True)
    return path


def mcs_legacy_h5_file(path: Path) -> Path:
    """Write a legacy-layout MCS file with root `/data` and `/data_channels_extra` and `configurationGUI` and `configurationFPGA` groups, and return its path."""
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as handle:
        handle.attrs.update(data_format_version="0.0.4", comment="legacy file")
        handle.create_dataset(
            "data", data=rng.poisson(3.0, size=(1, 2, 4, 5, 8, 5)).astype(np.uint16)
        )
        handle.create_dataset(
            "data_channels_extra",
            data=rng.poisson(2.0, size=(1, 2, 4, 5, 8, 2)).astype(np.uint16),
        )
        gui = handle.create_group("configurationGUI")
        gui.attrs.update(
            calib_x=2.0,
            calib_y=3.0,
            calib_z=4.0,
            range_x=10.0,
            range_y=12.0,
            range_z=6.0,
            offset_x=0.5,
            offset_y=0.25,
            offset_z=0.125,
            timebin_per_pixel=8,
            time_resolution=1.0,
            nx=5,
            ny=4,
            nframe=2,
            nrep=1,
            bitFile="C:/bits/DFD_40M91.bit",
        )
        fpga = handle.create_group("configurationFPGA")
        fpga.attrs.update(
            DFD_Activate=True, DFD_Trig_Selector=2, DFD_LaserSyncDebug=False
        )
    return path


def periodic_decay(
    tau_ns: float = 2.5,
    delay_ns: float = 5.0,
    bins: int = 64,
    period_ns: float = 25.0,
    counts: float = 3000.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the time axis in ns and a Poisson histogram of a decay repeating every `period_ns`, starting at `delay_ns`."""
    time = np.arange(bins) * period_ns / bins
    shape = np.exp(-np.mod(time - delay_ns, period_ns) / tau_ns)
    rate = counts * shape / shape.sum()
    return time, np.random.default_rng(seed).poisson(rate).astype(np.float64)


def _ttm_channel_columns(rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Return random times and valid flags `(rows, 21)` of the 21 channels beyond `ttm_raw_frame`."""
    rng = np.random.default_rng(1)
    return (
        rng.integers(0, 256, size=(rows, 21)).astype(np.uint8),
        (rng.random((rows, 21)) < 0.3).astype(np.uint8),
    )


def ttm_raw_bytes(no_data_rows: tuple[int, ...] = ()) -> bytes:
    """Return the packed RAW stream of the rows of `ttm_raw_frame`: 32 bytes a row, 21 channels, and the rows in `no_data_rows` flagged as no data."""
    frame = ttm_raw_frame()
    rows = len(frame)
    times, valid = _ttm_channel_columns(rows)
    names = [("step", "<u2"), ("t_L", "u1"), ("WORD_HEAD", "u1")]
    for word in range(7):
        names += [(f"t_{3 * word + k}", "u1") for k in (2, 1, 0)]
        names.append((f"WORD_{word}", "u1"))
    stream = np.zeros(rows, dtype=names)
    head = (
        frame["scan_enable"].to_numpy()
        | frame["line_enable"].to_numpy() << 1
        | frame["pixel_enable"].to_numpy() << 2
        | frame["valid_tdc_L"].to_numpy() << 3
    )
    head[list(no_data_rows)] |= 15 << 4
    stream["step"] = frame["step"]
    stream["t_L"] = frame["t_L"]
    stream["WORD_HEAD"] = head
    for word in range(7):
        for k in range(3):
            stream[f"t_{3 * word + k}"] = times[:, 3 * word + k]
        stream[f"WORD_{word}"] = (
            valid[:, 3 * word] << 1
            | valid[:, 3 * word + 1] << 2
            | valid[:, 3 * word + 2] << 3
        )
    return stream.tobytes()


def ttm_parsed_bytes() -> bytes:
    """Return the "parsed" stream of the rows of `ttm_raw_frame`: 96 bytes a row, a 12-byte header and seven 12-byte payloads."""
    frame = ttm_raw_frame()
    rows = len(frame)
    times, valid = _ttm_channel_columns(rows)
    names = [
        ("event", "<u4"),
        ("step", "<u2"),
        ("t_L", "u1"),
        ("scan_enable", "u1"),
        ("line_enable", "u1"),
        ("pixel_enable", "u1"),
        ("valid_tdc_L", "u1"),
        ("id_0", "u1"),
    ]
    for word in range(7):
        names.append((f"event_{3 * word}", "<u4"))
        names += [(f"t_{3 * word + k}", "u1") for k in (2, 1, 0)]
        names.append((f"useless_{word}", "u1"))
        names += [(f"valid_tdc_{3 * word + k}", "u1") for k in (2, 1, 0)]
        names.append((f"id_{word + 1}", "u1"))
    stream = np.zeros(rows, dtype=names)
    stream["event"] = np.arange(rows)
    for name in (
        "step",
        "t_L",
        "scan_enable",
        "line_enable",
        "pixel_enable",
        "valid_tdc_L",
    ):
        stream[name] = frame[name]
    for channel in range(21):
        stream[f"t_{channel}"] = times[:, channel]
        stream[f"valid_tdc_{channel}"] = valid[:, channel]
    return stream.tobytes()


def ttm_calibration_h5_file(path: Path) -> Path:
    """Write a pandas HDF5 file with `main`, `ch_0`, `ch_3` (no `ch_1`) and `metadata` tables, as the converters lay them out, and return `path`."""
    rng = np.random.default_rng(0)
    rows = 2000
    main = pd.DataFrame(
        {
            "cumulative_step": np.cumsum(rng.integers(1, 200, size=rows)),
            "t_L": rng.integers(0, 150, size=rows).astype(np.uint8),
            "total_photon": rng.integers(0, 3, size=rows).astype(np.uint8),
        }
    )
    main.to_hdf(path, mode="w", key="main", format="table")
    for channel, spread in ((0, 120), (3, 90)):
        index = np.sort(rng.choice(rows, size=600, replace=False))
        pd.DataFrame(
            {
                f"t_{channel}": rng.integers(0, spread, size=600).astype(np.int16),
                f"dS_{channel}": rng.integers(0, 4, size=600).astype(np.uint16),
            },
            index=pd.Index(index, name=f"index_{channel}"),
        ).to_hdf(path, key=f"ch_{channel}", format="table")
    pd.DataFrame(
        {
            "sysclk_MHz": np.asarray(240.0),
            "laser_MHz": np.asarray(40.0),
            "sysclk_ps": np.asarray(1e6 / 240.0),
            "laser_ps": np.asarray(1e6 / 40.0),
        },
        index=[0],
    ).to_hdf(path, key="metadata")
    return path
