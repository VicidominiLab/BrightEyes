"""Equivalence cases for the small modules of `brighteyes_mcs_dataprep`."""

import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from _capture import error as _error
from _h5_dump import h5_contents
from _inputs import mcs_h5_file, periodic_decay
from _registry import Case
from matplotlib import pyplot as plt

WRITERS = "brighteyes_mcs_dataprep.h5_output_writers"
HASH = "brighteyes_mcs_dataprep.h5_file_hash"
PHASOR = "brighteyes_mcs_dataprep.tools_phasor"
SKEW = "brighteyes_mcs_dataprep.channel_skew_estimator"
GRAPH = "brighteyes_mcs_dataprep.graph"

CREATED_AT = frozenset({"created_utc"})


def _drawn(call: Callable[[], Any]) -> Any:
    plt.close("all")
    result = call()
    content = [
        (
            [line.get_xydata() for line in axes.lines],
            [
                (
                    collection.get_array(),
                    collection.get_offsets(),
                    [path.vertices for path in collection.get_paths()],
                )
                for collection in axes.collections
            ],
            [
                (type(patch).__name__, patch.get_path().vertices)
                for patch in axes.patches
            ],
            [text.get_text() for text in axes.texts],
            [text.get_text() for text in (axes.get_legend() or _NoLegend).get_texts()],
            [axes.get_title(), axes.get_xlabel(), axes.get_ylabel()],
            axes.get_xscale(),
            axes.get_yscale(),
        )
        for number in plt.get_fignums()
        for axes in plt.figure(number).axes
    ]
    plt.close("all")
    return result, content


class _NoLegend:
    @staticmethod
    def get_texts() -> list:
        return []


def _in_folder(use: Callable[[Any, Path], Any]) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            return use(f, Path(name))

    return call


def _products(module: Any) -> list[Any]:
    return [
        module.H5OutputProduct(
            "spad",
            np.arange(24, dtype=np.float64).reshape(2, 3, 4),
            {
                "axis_order": "y,x,time_bin",
                "run": "{run_id} at {run_path}",
                "shape": (2, 3, 4),
                "units": None,
            },
        ),
        module.H5OutputProduct(
            "fit_maps/tau", np.linspace(1, 2, 6).reshape(2, 3), compression=None
        ),
        module.H5OutputProduct(
            "scalar", np.float64(3.5), {"where": Path("a") / "b"}, chunks=None
        ),
    ]


def _written(
    products: Callable[[Any], Any] | None = _products,
    output_name: str | None = None,
    times: int = 1,
    **kwargs: Any,
) -> Callable[[Any], Any]:
    def use(f: Callable[..., Any], folder: Path) -> Any:
        source = mcs_h5_file(folder / "source.h5")
        module = import_module(f.__module__)
        options = dict(kwargs)
        if output_name is not None:
            options["output_path"] = folder / output_name
        written = [
            f(
                source,
                "tool",
                products(module) if products is not None else {"x": np.ones(3)},
                **options,
            )
            for _ in range(times)
        ]
        contents = {
            path.name: h5_contents(path, folder, CREATED_AT)
            for path in sorted(folder.glob("*.h5"))
        }
        return [(Path(p).name, run) for p, run in written], contents

    return _in_folder(use)


def _write_error(**kwargs: Any) -> Callable[[Any], Any]:
    def use(f: Callable[..., Any], folder: Path) -> Any:
        source = mcs_h5_file(folder / "source.h5")
        options = dict(kwargs)
        products = options.pop("products", {"x": np.ones(3)})
        return _error(lambda: f(source, "tool", products, **options))

    return _in_folder(use)


def _extra_products(module: Any) -> list[Any]:
    return [
        module.H5OutputProduct(
            "image", np.arange(12.0).reshape(3, 4), {"data_role": "image"}
        )
    ]


def _ensure_output_group(f: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "file.h5"
        with h5py.File(path, "w") as handle:
            first = dict(f(handle).attrs)
            handle["output"].attrs["default"] = "kept"
            second = dict(f(handle).attrs)
            custom = f(handle, "/results/").name
            nested = dict(f(handle, "a").attrs)
            keys = sorted(handle)
        return (
            first,
            second,
            custom,
            nested,
            keys,
            _error(lambda: _open_and(path, f, "/")),
        )


def _open_and(path: Path, f: Callable[..., Any], key: str) -> None:
    with h5py.File(path, "a") as handle:
        f(handle, key)


def _resolve_run_id(f: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        with h5py.File(Path(folder) / "file.h5", "w") as handle:
            group = handle.create_group("output")
            for name in ("run", "run_001", "run_007", "plain", "x_1"):
                group.create_group(name)
            ids = [
                f(group, "new"),
                f(group, "run"),
                f(group, "run_001"),
                f(group, "run_007"),
                f(group, "plain"),
                f(group, "x_1"),
                f(group, "/new/"),
            ]
            overwritten = f(group, "run_001", output_key_overwrite=True)
            remaining = sorted(group)
            errors = [
                _error(lambda: f(group, "")),
                _error(lambda: f(group, "a/b")),
            ]
        return ids, overwritten, remaining, errors


def _write_attrs(f: Callable[..., Any]) -> Any:
    with (
        tempfile.TemporaryDirectory() as folder,
        h5py.File(Path(folder) / "file.h5", "w") as handle,
    ):
        f(handle, None)
        f(handle, {})
        f(
            handle,
            {
                "text": "abc",
                "number": 3,
                "real": 1.5,
                "nothing": None,
                "path": Path("a") / "b",
                "mapping": {"a": np.arange(3), "b": np.float64(2.0)},
                "sequence": [1, 2, np.int64(3)],
                "pair": (1.0, "x"),
                "array": np.arange(4),
                "scalar": np.float32(2.5),
                7: "integer key",
            },
        )
        return dict(handle.attrs)


def _fingerprint(f: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = mcs_h5_file(Path(folder) / "file.h5")
        with h5py.File(path, "r") as handle:
            return [f(handle["raw/spad"]), f(handle["raw/aux"])]


def _file_hash(f: Callable[..., Any], **kwargs: Any) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = mcs_h5_file(Path(folder) / "file.h5")
        with h5py.File(path, "r") as handle:
            return f(handle, **kwargs)


def _hash_of(**kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: _file_hash(f, **kwargs)


def _legacy_hash(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with (
            tempfile.TemporaryDirectory() as folder,
            h5py.File(Path(folder) / "file.h5", "w") as handle,
        ):
            handle.create_dataset("data", data=np.arange(24).reshape(2, 3, 4))
            handle.create_dataset("analog", data=np.arange(6).reshape(3, 2))
            return f(handle, **kwargs)

    return call


def _skew_data(channels: int = 6, bins: int = 64, noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    axis = np.arange(bins)
    shifts = np.array([0.0, 1.5, 3.0, -2.0, 0.7, 2.2, -1.1, 4.0])[:channels]
    data = np.stack(
        [np.exp(-((axis - 30 - shift) ** 2) / (2 * 2.0**2)) for shift in shifts],
        axis=1,
    )
    return data + noise * rng.uniform(size=data.shape)


def _skew(**kwargs: Any) -> Callable[[Any], Any]:
    reference = kwargs.pop("reference", False)
    noise = kwargs.pop("noise", 0.0)

    def call(f: Callable[..., Any]) -> Any:
        data = _skew_data(noise=noise)
        args = (
            (data, _skew_data(channels=4, noise=noise)[:, ::-1])
            if reference
            else (data,)
        )
        return f(*args, **kwargs)

    return call


def _skew_error(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: _error(lambda: f(*args, **kwargs))


def _decay(**kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    return periodic_decay(**kwargs)


def _birfi(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        time, histogram = _decay(bins=128, period_ns=25.0, counts=20000.0)
        return f(time, histogram, **kwargs)

    return call


def _birfi_stack(axis: int, **kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        time = _decay(bins=128)[0]
        stack = np.stack(
            [
                _decay(bins=128, tau_ns=tau, counts=20000.0, seed=index)[1]
                for index, tau in enumerate((1.5, 2.5, 3.5, 1.0, 2.0, 4.0))
            ],
            axis=1,
        ).reshape(128, 2, 3)
        return f(time, np.moveaxis(stack, 0, axis), axis=axis, **kwargs)

    return call


def _birfi_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        time, histogram = _decay(bins=128)
        x = options.pop("x", time)
        y = options.pop("y", histogram)
        return _error(lambda: f(x, y, **options))

    return call


def _log(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        time, histogram = _decay(counts=20000.0)
        values = {"dt_ns": 25.0 / 64, "nbin": 64, "period_ns": 25.0, **kwargs}
        histogram = values.pop("histogram", histogram)
        return f(histogram, time, **values)

    return call


def _circmean(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        counts = _decay(counts=20000.0)[1]
        values = {"t0_ns": 5.0, **kwargs}
        counts = values.pop("counts", counts)
        return f(counts, **values)

    return call


def _lifetimes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(2.5, 0.3, size=400), rng.uniform(0.1, 5.0, size=400)


def _histogram(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        lifetime, weights = _lifetimes()
        if options.pop("weighted", False):
            options["weights"] = weights
        return _drawn(lambda: f(lifetime, **options))

    return call


def _summary() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    channels = np.arange(8)
    tau = 2.5 + 0.2 * rng.normal(size=8)
    return pd.DataFrame(
        {
            "channel": channels,
            "tau_ns": tau,
            "tau_err_ns": 0.05 + 0.01 * rng.uniform(size=8),
            "tau_reference_ns": np.full(8, 1.0),
            "residual_error": 0.01 * rng.uniform(size=8),
            "channel_skew": rng.normal(0, 1.0, size=8),
            "channel_skew_err": 0.1 * rng.uniform(size=8),
            "delay_correction_ns": rng.normal(0, 0.2, size=8),
            "fitted_delay_ns": rng.normal(1, 0.2, size=8),
            "fitted_delay_err_ns": 0.02 * rng.uniform(size=8),
        }
    )


def _lifetime_summary(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        table = _summary()
        if options.pop("without_reference", False):
            table["tau_reference_ns"] = np.nan
        if options.pop("mapping", False):
            table = {key: table[key].to_numpy() for key in table}
        return _drawn(lambda: f(table, **options))

    return call


def _shift_summary(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        table = _summary()
        tables = table
        if options.pop("two", False):
            second = table.copy()
            second["channel_skew"] = second["channel_skew"] + 0.5
            second["fitted_delay_ns"] = second["fitted_delay_ns"] * 1.2
            tables = [table, second]
        if options.pop("existing_figure", False):
            options["fig"] = plt.figure()
        if options.pop("blank_axes", False):
            figure = plt.figure()
            figure.add_subplot(121)
            options["fig"] = figure
        return _drawn(lambda: f(tables, **options))

    return call


def _fit_traces(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        time, measured = _decay(counts=5000.0)
        traces = {
            "measured_trace": measured,
            "reference_trace": _decay(tau_ns=1.0, seed=1)[1],
            "aligned_irf_trace": _decay(tau_ns=0.3, seed=2)[1],
            "fitted_trace": np.exp(-time / 2.5),
            "custom": np.zeros(64),
            "missing": None,
        }
        if options.pop("existing_axes", False):
            options["ax"] = plt.subplots()[1]
        return _drawn(lambda: f(time, traces, **options))

    return call


def _error_of(*args: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: _error(lambda: f(*args, **kwargs))


CASES: list[Case] = [
    Case(
        f"{WRITERS}:H5OutputProduct",
        lambda cls: (cls("a", np.ones(2)), cls("b", [1, 2], {"k": 1}, None, 5)),
    ),
    Case(f"{WRITERS}:ensure_output_group", _ensure_output_group),
    Case(f"{WRITERS}:resolve_output_run_id", _resolve_run_id),
    Case(f"{WRITERS}:write_attrs", _write_attrs),
    Case(f"{WRITERS}:write_h5_output_run", _written(), "append"),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(products=None, times=2),
        "append twice versions the run",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(products=_extra_products, times=3, output_key_overwrite=True),
        "append overwrite",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(mode="copy"),
        "copy to default path",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(mode="copy", output_name="copy.h5", set_default=True),
        "copy with default",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(mode="outputs_only", output_name="fragment.h5"),
        "outputs only",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(mode="outputs_only"),
        "outputs only default path",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(
            set_default=True,
            default_product="fit_maps/tau",
            tool_name="Tool",
            algorithm_name="algorithm",
            output_type="fit_tool",
            parameters={"order": 2, "levels": [1, 2], "label": "{run_id}"},
            metadata={"unit": "ns", "path": "{run_path}"},
            inputs={"source": Path("data") / "x"},
            attrs={"note": "for {run_id}", "output_type": "overridden"},
            axes={
                "time_ns": np.arange(4) * 0.5,
                "scaled": np.ones((2, 2)),
            },
            intermediates={"mask": np.ones((2, 2), dtype=bool)},
        ),
        "all sections",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _written(set_default=True, default_product="/output/x/products/scalar"),
        "default product by path",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _write_error(mode="other"),
        "invalid mode",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _write_error(products=[]),
        "no products",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _write_error(products=[np.ones(2)]),
        "invalid product",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _write_error(set_default=True, default_product="missing"),
        "unknown default product",
    ),
    Case(
        f"{WRITERS}:write_h5_output_run",
        _write_error(output_path="other.h5"),
        "append with output path",
    ),
    Case(f"{HASH}:channel_fingerprint", _fingerprint),
    Case(f"{HASH}:channel_fingerprint_file_hash", _hash_of(), "sha256"),
    Case(
        f"{HASH}:channel_fingerprint_file_hash",
        _hash_of(algorithm="md5"),
        "md5",
    ),
    Case(
        f"{HASH}:channel_fingerprint_file_hash",
        _legacy_hash(),
        "legacy keys",
    ),
    Case(f"{HASH}:channel_fingerprint_file_hash_attrs", _hash_of(), "default prefix"),
    Case(
        f"{HASH}:channel_fingerprint_file_hash_attrs",
        _hash_of(prefix="source_file"),
        "prefix",
    ),
    Case(
        f"{HASH}:channel_fingerprint_file_hash_attrs",
        _legacy_hash(prefix="source_file"),
        "legacy keys",
    ),
    Case(f"{SKEW}:estimate_channel_skew", _skew(), "last channel"),
    Case(f"{SKEW}:estimate_channel_skew", _skew(reference_channel=2), "channel 2"),
    Case(f"{SKEW}:estimate_channel_skew", _skew(apodize=True, noise=0.01), "apodize"),
    Case(
        f"{SKEW}:estimate_channel_skew",
        _skew(reference=True),
        "reference histogram",
    ),
    Case(
        f"{SKEW}:estimate_channel_skew",
        lambda f: f(
            _skew_data(),
            _skew_data(channels=4)[:, ::-1],
            reference_channel=1,
            upsampling=4,
        ),
        "reference channel",
    ),
    Case(
        f"{SKEW}:estimate_channel_skew",
        _skew_error(np.zeros(5)),
        "not two-dimensional",
    ),
    Case(
        f"{SKEW}:estimate_channel_skew",
        _skew_error(np.zeros((8, 0))),
        "no channels",
    ),
    Case(
        f"{SKEW}:estimate_channel_skew",
        _skew_error(np.zeros((8, 3)), reference_channel=3),
        "reference out of range",
    ),
    Case(
        f"{SKEW}:estimate_channel_skew",
        _skew_error(np.zeros((8, 3)), np.zeros((6, 3))),
        "bin count mismatch",
    ),
    Case(
        f"{SKEW}:estimate_channel_skew",
        _skew_error(np.zeros((8, 3)), np.zeros((8, 3)), reference_channel=5),
        "reference histogram channel out of range",
    ),
    Case(f"{PHASOR}:estimate_lifetime_from_birfi", _birfi(), "1d"),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi(return_bounds=True, window_length=7, polyorder=2),
        "bounds",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi(persistence=3, threshold=0.2),
        "persistence and threshold",
    ),
    Case(f"{PHASOR}:estimate_lifetime_from_birfi", _birfi_stack(0), "stack axis 0"),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_stack(2, return_bounds=True),
        "stack last axis",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(x=np.zeros((4, 2))),
        "time axis not 1d",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(y=np.float64(1.0)),
        "scalar data",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(y=np.ones(10)),
        "length mismatch",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(window_length=10),
        "even window",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(window_length=500),
        "window too long",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(polyorder=11),
        "polyorder too large",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(persistence=0),
        "persistence below one",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        _birfi_error(x=np.arange(2.0), y=np.ones(2)),
        "too few samples",
    ),
    Case(f"{PHASOR}:estimate_lifetime_from_log", _log(), "default"),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        _log(start_level=0.8, end_level=0.1),
        "levels",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        _log(histogram=np.zeros(64)),
        "zero histogram",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        _log(histogram=np.ones(64)),
        "flat histogram",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        _log(histogram=np.ones(3)),
        "too few bins",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        _log(start_level=0.2, end_level=0.5),
        "inverted levels",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        _log(dt_ns=0.0),
        "zero bin width",
    ),
    Case(f"{PHASOR}:estimate_lifetime_from_circmean", _circmean(), "default"),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        _circmean(repetition_rate_MHz=80.0, background_per_bin=20.0),
        "rate and background",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        _circmean(truncate_ns=12.0),
        "truncated",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        _circmean(counts=np.zeros(64)),
        "zero counts",
    ),
    Case(
        f"{GRAPH}:normalize_histogram",
        lambda f: [
            f(np.arange(5)),
            f(np.zeros(4)),
            f([1, np.nan, 2]),
            f(-np.ones(3)),
        ],
    ),
    Case(
        f"{GRAPH}:weighted_lifetime_stats",
        lambda f: [
            f(_lifetimes()[0]),
            f(_lifetimes()[0], _lifetimes()[1]),
            f([np.nan, np.nan]),
            f([1.0, 2.0, np.nan], [1.0, 0.0, 2.0]),
            f([1.0, 2.0], [0.0, -1.0]),
            _error(lambda: f([1.0, 2.0], [1.0])),
        ],
    ),
    Case(f"{GRAPH}:plot_lifetime_histogram", _histogram(bins=40), "x axis"),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        _histogram(bins=40, lifetime_axis="y", weighted=True, label="channel"),
        "y axis weighted",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        _histogram(
            bins=30,
            lifetime_bounds=(1.0, 4.0),
            gaussian=False,
            color="red",
            edgecolor="blue",
            count_label="Counts",
        ),
        "bounds without gaussian",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        _error_of([1.0, 2.0], lifetime_axis="z"),
        "invalid axis",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        _error_of([1.0, 2.0], weights=[1.0]),
        "weights mismatch",
    ),
    Case(f"{GRAPH}:plot_calibration_lifetime_summary", _lifetime_summary(), "y axis"),
    Case(
        f"{GRAPH}:plot_calibration_lifetime_summary",
        _lifetime_summary(histogram_lifetime_axis="x", without_reference=True),
        "x axis without reference",
    ),
    Case(
        f"{GRAPH}:plot_calibration_lifetime_summary",
        _lifetime_summary(mapping=True),
        "mapping table",
    ),
    Case(
        f"{GRAPH}:plot_calibration_lifetime_summary",
        _error_of(5),
        "table without columns",
    ),
    Case(f"{GRAPH}:plot_calibration_shift_summary", _shift_summary(), "one table"),
    Case(
        f"{GRAPH}:plot_calibration_shift_summary",
        _shift_summary(two=True, labels=["data", "aux"], reference_channel=3),
        "two tables",
    ),
    Case(
        f"{GRAPH}:plot_calibration_shift_summary",
        _shift_summary(existing_figure=True),
        "figure without axes",
    ),
    Case(
        f"{GRAPH}:plot_calibration_shift_summary",
        _shift_summary(blank_axes=True),
        "figure with one axes",
    ),
    Case(f"{GRAPH}:plot_calibration_fit_traces", _fit_traces(), "log scale"),
    Case(
        f"{GRAPH}:plot_calibration_fit_traces",
        _fit_traces(title="channel 3", log_scale=False, existing_axes=True),
        "linear scale on given axes",
    ),
]

SKIPPED: dict[str, str] = {}
