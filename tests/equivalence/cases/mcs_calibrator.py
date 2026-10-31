"""Equivalence cases for `brighteyes_mcs_dataprep.h5_data_calibrator`."""

import tempfile
import warnings
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from _capture import error, printed, warned
from _h5_dump import h5_contents
from _inputs import mcs_flim_h5_file, mcs_h5_file, mcs_output_h5_file
from _registry import Case

CAL = "brighteyes_mcs_dataprep.h5_data_calibrator"
CREATED_AT = frozenset({"created_utc", "created_at_iso8601"})


def _snapshot(folder: Path) -> dict[str, Any]:
    return {
        path.name: h5_contents(path, folder, CREATED_AT)
        for path in sorted(folder.glob("*.h5"))
    }


def _plain(instance: Any, folder: Path) -> dict[str, Any]:
    return {
        name: str(value.relative_to(folder)) if isinstance(value, Path) else value
        for name, value in vars(instance).items()
    }


def _custom_model(
    t: np.ndarray, irf: np.ndarray, period: float, C: float, dT: float, tau: float
) -> np.ndarray:
    decay = np.exp(-np.mod(t, period) / tau)
    decay = decay / decay.sum()
    phase = np.exp(-2j * np.pi * np.fft.fftfreq(len(t)) * dT)
    shifted = np.fft.ifft(np.fft.fft(irf / irf.sum()) * phase).real
    return C * np.fft.ifftshift(
        np.fft.ifft(np.fft.fft(decay) * np.fft.fft(shifted)).real
    )


def _inputs(
    folder: Path,
    silent_data: int | None = None,
    silent_reference: int | None = None,
    circular: bool = False,
) -> tuple[Path, Path]:
    data = mcs_flim_h5_file(
        folder / "data.h5", 2.5, 0, silent_channel=silent_data, circular=circular
    )
    reference = mcs_flim_h5_file(
        folder / "reference.h5",
        1.0,
        1,
        silent_channel=silent_reference,
        circular=circular,
    )
    return data, reference


def _options(kwargs: dict[str, Any], folder: Path) -> dict[str, Any]:
    options = {
        "irf_iterations": 20,
        "tau_ref": 1.0,
        "output_path": folder / "out.h5",
        **kwargs,
    }
    if isinstance(options["output_path"], str):
        options["output_path"] = folder / options["output_path"]
    return options


def _calibrator(
    module: Any, folder: Path, kwargs: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    options = _options(kwargs, folder)
    data, reference = _inputs(
        folder,
        options.pop("silent_data", None),
        options.pop("silent_reference", None),
        options.pop("circular", False),
    )
    return module.H5DataCalibrator(data, reference, **options), options


def _calibrate(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            instance, _ = _calibrator(import_module(f.__module__), folder, kwargs)
            returned, caught = warned(lambda: f(instance))
            return Path(returned).name, caught, _snapshot(folder)

    return call


def _calibrate_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            instance, _ = _calibrator(import_module(f.__module__), folder, kwargs)
            return error(lambda: f(instance))

    return call


def _constructed(**kwargs: Any) -> Callable[[Any], Any]:
    def call(cls: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            options = _options(kwargs, folder)
            data, reference = _inputs(folder)
            instance = cls(data, reference, **options)
            returned, caught = warned(instance.calibrate)
            return (
                _plain(instance, folder),
                Path(returned).name,
                caught,
                _snapshot(folder),
            )

    return call


def _construction_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(cls: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            data, reference = _inputs(folder)
            return error(lambda: cls(data, reference, **_options(kwargs, folder)))

    return call


def _wrapper(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            options = _options(kwargs, folder)
            data, reference = _inputs(folder)
            returned, caught = warned(lambda: f(data, reference, **options))
            return Path(returned).name, caught, _snapshot(folder)

    return call


def _wrapper_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            options = _options(kwargs, folder)
            data, reference = _inputs(folder)
            return error(lambda: f(data, reference, **options))

    return call


def _calibrated_file(module: Any, folder: Path, circular: bool = False) -> Path:
    data, reference = _inputs(folder, circular=circular)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Path(
            module.calibrate_h5_file(
                data,
                reference,
                output_path=folder / "calibrated.h5",
                create_output=False,
                irf_iterations=20,
                tau_ref=1.0,
            )
        )


def _with_analog(path: Path) -> Path:
    with h5py.File(path, "a") as handle:
        spad = handle["raw/spad"]
        analog = handle["raw"].create_dataset(
            "analog",
            data=np.arange(np.prod(spad.shape[:5]) * 2, dtype=np.uint16).reshape(
                *spad.shape[:5], 2
            ),
        )
        analog.attrs.update(axis_order="repetition,z,y,x,time_bin,analog_channel")
    return path


def _built(
    source: str = "calibrated", output_name: str | None = None, **kwargs: Any
) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            module = import_module(f.__module__)
            path = _source(module, folder, source)
            options = dict(kwargs)
            if output_name is not None:
                options["output_path"] = folder / output_name
            builder = module.H5OutputBuilder(path, **options)
            returned, caught = warned(lambda: f(builder))
            return Path(returned).name, caught, _snapshot(folder)

    return call


def _source(module: Any, folder: Path, source: str) -> Path:
    match source:
        case "calibrated":
            return _calibrated_file(module, folder)
        case "calibrated with analog":
            return _with_analog(_calibrated_file(module, folder))
        case "calibrated circular":
            return _calibrated_file(module, folder, circular=True)
        case "uncalibrated":
            return _uncalibrated(folder)
        case "legacy":
            return mcs_flim_h5_file(folder / "legacy.h5")
    raise ValueError(source)


def _uncalibrated(folder: Path) -> Path:
    return mcs_h5_file(folder / "plain.h5")


def _build_error(source: str = "calibrated", **kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            module = import_module(f.__module__)
            path = _source(module, folder, source)
            return error(lambda: f(module.H5OutputBuilder(path, **kwargs)))

    return call


def _builder_instance(cls: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        path = folder / "file.h5"
        default = cls(path)
        configured = cls(
            path,
            folder / "other.h5",
            in_place=True,
            overwrite=False,
            output_key="/results/",
            primary_data_key="data",
            extra_data_key="data_channels_extra",
            create_virtual_channels=False,
            create_sum_channels=False,
            sum_channels_run_id="sum_a",
            sum_channels_with_skew_correction_run_id="sum_b",
            channels=[1, 2],
            extra_channels=[0],
            include_extra_sum=False,
            include_extra_shifted_sum=False,
            require_shifted_sum=True,
            shifted_sum_backend="cpu",
            shifted_sum_chunk_size=8,
            shifted_sum_reverse_shifts=False,
            shifted_sum_show_progress=True,
            compression=None,
            max_block_bytes=1024,
        )
        aliased, caught = warned(
            lambda: cls(
                path,
                spad_data_key="a",
                aux_data_key="b",
                aux_channels=[2],
                include_aux_sum=False,
                include_aux_shifted_sum=False,
                create_sum=False,
                create_sum_using_shift=False,
                sum_run_id="x",
                sum_using_shift_run_id="y",
            )
        )
        return (
            _plain(default, folder),
            _plain(configured, folder),
            _plain(aliased, folder),
            caught,
            error(lambda: cls(path, max_block_bytes=0)),
        )


def _alias_properties(prop: property) -> Any:
    cls = import_module(prop.fget.__module__).H5OutputBuilder
    with tempfile.TemporaryDirectory() as name:
        builder = cls(Path(name) / "file.h5")

        def exercise() -> list[Any]:
            values = []
            for alias, value in (
                ("create_sum", False),
                ("create_sum_using_shift", False),
                ("create_sum_shifted", True),
                ("sum_run_id", "run_a"),
                ("sum_using_shift_run_id", "run_b"),
            ):
                values.append(getattr(builder, alias))
                setattr(builder, alias, value)
                values.append(getattr(builder, alias))
            return values

        values, caught = warned(exercise)
        return values, caught, vars(builder)["create_sum_channels"]


def _functions(f: Callable[..., Any], output_name: str | None = None, **kwargs: Any):
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name)
        module = import_module(f.__module__)
        path = _calibrated_file(module, folder)
        output = folder / output_name if output_name is not None else None
        returned, caught = warned(lambda: f(path, output, **kwargs))
        return Path(returned).name, caught, _snapshot(folder)


def _structure(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as name:
            path = mcs_output_h5_file(Path(name) / "file.h5")
            return printed(lambda: f(path, **kwargs))

    return call


def _structure_html(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        f.__globals__["_WIDGET_COUNTER"] = 0
        with tempfile.TemporaryDirectory() as name:
            path = mcs_output_h5_file(Path(name) / "file.h5")
            return printed(lambda: f(path, **kwargs))

    return call


def _time_axis(*arguments: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: f(*arguments, **kwargs)


def _time_axis_error(*arguments: Any, **kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: error(lambda: f(*arguments, **kwargs))


class _Metadata:
    def __init__(self, **values: Any) -> None:
        self.__dict__.update(values)


CASES: list[Case] = [
    Case(f"{CAL}:H5DataCalibrator", _constructed(), "default"),
    Case(
        f"{CAL}:H5DataCalibrator",
        _constructed(
            reference_key={
                "data": "data",
                "data_channels_extra": "data_channels_extra",
            },
            reference_type="reference",
            fit_mode="irf_shift",
            fit_type="poisson",
            C_ref=1.5,
            overwrite=True,
            channels=[0, 1],
            calibration_key="calib",
            period_ns=25.0,
            param_names=["C", "dT", "tau"],
            model_kwargs={"scale": 2},
            eps=1e-6,
            regularization=3,
            irf_corrections_type="single",
            channel_skew_fit_reference_channel=1,
            channel_skew_fit_upsampling=4,
            channel_skew_fit_apodize=True,
            output_options={
                "output_path": "ignored.h5",
                "create_virtual_channels": False,
            },
        ),
        "all options",
    ),
    Case(f"{CAL}:H5DataCalibrator", _construction_error(data_key=[]), "empty keys"),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(data_key=["data", "data"]),
        "duplicate keys",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(data_key=5),
        "keys not iterable",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(reference_key={"other": "data"}),
        "reference key missing for a data key",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(reference_key=["a", "b", "c"]),
        "reference key count mismatch",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(reference_type="x"),
        "invalid reference type",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(irf_corrections_type="x"),
        "invalid correction type",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(channel_skew_type="x"),
        "unsupported skew type",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(channel_skew_source="x"),
        "invalid skew source",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(channel_skew_source=np.ones((2, 2))),
        "skew source not 1d",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(channel_skew_fit_upsampling=0),
        "invalid upsampling",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(parameter_names=["a"], param_names=["b"]),
        "parameter names disagree",
    ),
    Case(
        f"{CAL}:H5DataCalibrator",
        _construction_error(
            model_fn=_custom_model, p0=[1.0, 0.0, 2.0], delay_param="x"
        ),
        "custom model without delay parameter",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis({"time_bins": 64, "laser_frequency_mhz": 40.0}),
        "laser frequency",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis({"nbin": 32, "dfd_freq": 80.0}, nbin=16),
        "explicit bin count",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis(
            _Metadata(digital_time_bins=50, pixel_dwell_time_us=2.0),
        ),
        "pixel dwell time",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis({"dfd_nbins": 20, "time_resolution_us": 0.25}),
        "time resolution",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis({"time_bins": 10}, period_ns=30.0),
        "explicit period",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis_error({}),
        "no bin count",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis_error({"time_bins": 0}),
        "non-positive bin count",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis_error({"time_bins": 10}),
        "no timing information",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.build_time_axis",
        _time_axis_error({"time_bins": 10}, period_ns=-1.0),
        "negative period",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(),
        "estimated reference lifetime",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(tau_ref=None, fit_mode="irf_shift"),
        "irf shift",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(tau_ref="circmean", fit_type="curve_fit_circular"),
        "circular mean lifetime and circular curve fit",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(tau_ref="birfi", fit_type="curve_fit"),
        "birfi lifetime and curve fit",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(reference_type="irf", clean_irf=True, channel_skew_source="irf"),
        "reference is an irf, cleaned",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(irf_corrections_type="single_ch", channel_skew_source="irf"),
        "single channel corrections and irf skew source",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(channel_skew_source="data", channel_skew_fit_apodize=True),
        "data skew source with apodization",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(
            data_key="data",
            channel_skew_source=np.array([0.0, 1.0, -1.0, 0.5, 2.0, -0.5]),
        ),
        "forced skew vector",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(
            data_key="data",
            channels=[4, 0, 2],
            channel_skew_fit_reference_channel=2,
            channel_skew_fit_upsampling=2,
        ),
        "channel subset",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(data_key="data_channels_extra", calibration_key="calib2"),
        "auxiliary data only",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(
            period_ns=25.0,
            force_C_normalized=True,
            initial_tau=2.0,
            initial_dT=1.0,
            initial_C=1.0,
            C_ref=2.0,
            eps=1e-6,
            regularization=3,
        ),
        "explicit period and fit settings",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(
            reference_type="irf",
            channel_skew_source="irf",
            model_fn=_custom_model,
            p0=[1.0, 0.0, 2.0],
            parameter_names=["C", "dT", "tau"],
            bounds=([0.0, -10.0, 0.1], [2.0, 10.0, 20.0]),
        ),
        "custom model",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(create_output=False),
        "without output group",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(
            output_options={
                "create_virtual_channels": False,
                "sum_channels_run_id": "plain_sum",
                "output_path": "ignored.h5",
            }
        ),
        "output options",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(circular=True),
        "circular eight-dimensional layout",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(silent_data=2),
        "silent data channel",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(silent_reference=1),
        "silent reference channel",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate_error(overwrite=False),
        "existing output group without overwrite",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate(overwrite=False, create_output=False),
        "without overwrite and without output group",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate_error(channels=[9]),
        "channel out of range",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate_error(channels=[1, 1]),
        "duplicate channels",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate_error(channel_skew_source="metadata"),
        "metadata skew source",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate_error(data_key="missing"),
        "missing data key",
    ),
    Case(
        f"{CAL}:H5DataCalibrator.calibrate",
        _calibrate_error(output_path="data.h5"),
        "output is the input",
    ),
    Case(
        f"{CAL}:H5OutputBuilder",
        _builder_instance,
        "attributes",
    ),
    Case(f"{CAL}:H5OutputBuilder.build", _built(), "in place"),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(output_name="copy.h5"),
        "copy",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(create_virtual_channels=False, create_sum_channels=False),
        "skew correction only",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(create_sum_channels_with_skew_correction=False),
        "sum only",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(
            output_key="/results/",
            sum_channels_run_id="plain",
            sum_channels_with_skew_correction_run_id="skew",
        ),
        "named runs",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(
            channels=[1, 3, 4],
            aux_channels=[1],
            include_aux_sum=False,
            include_aux_shifted_sum=True,
        ),
        "channel subsets",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(include_extra_sum=False, include_extra_shifted_sum=False),
        "without auxiliary sums",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(
            shifted_sum_backend="cpu",
            shifted_sum_chunk_size=3,
            shifted_sum_reverse_shifts=False,
            require_shifted_sum=True,
            compression=None,
            max_block_bytes=64,
        ),
        "forward shifts and small blocks",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(compression="lzf", max_block_bytes=200, shifted_sum_backend="numpy"),
        "lzf compression",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(source="calibrated with analog"),
        "with analog data",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(source="calibrated circular"),
        "circular eight-dimensional layout",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(source="uncalibrated"),
        "file without calibration",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(
            source="legacy",
            output_name="legacy_out.h5",
            spad_data_key="data",
            aux_data_key="data_channels_extra",
            create_sum_channels_with_skew_correction=False,
        ),
        "legacy keys",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _built(
            create_sum=False,
            create_sum_using_shift=True,
            sum_using_shift_run_id="aliased",
            primary_data_key="raw/spad",
        ),
        "deprecated aliases",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _build_error(source="uncalibrated", require_shifted_sum=True),
        "shifted sum required without calibration",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _build_error(source="legacy"),
        "legacy file with default keys",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _build_error(overwrite=False),
        "existing output without overwrite",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.build",
        _build_error(channels=[20]),
        "channel out of range",
    ),
    Case(
        f"{CAL}:H5OutputBuilder.create_sum", _alias_properties, "deprecated properties"
    ),
    Case(
        f"{CAL}:build_h5_output",
        lambda f: _functions(f, output_name="built.h5"),
        "copy",
    ),
    Case(
        f"{CAL}:build_h5_output",
        lambda f: _functions(f, create_virtual_channels=False, overwrite=True),
        "in place",
    ),
    Case(
        f"{CAL}:add_output_to_h5_file",
        lambda f: _functions(f, output_name="added.h5", create_sum_channels=False),
        "copy",
    ),
    Case(f"{CAL}:calibrate_h5_file", _wrapper(), "default"),
    Case(
        f"{CAL}:calibrate_h5_file",
        _wrapper(
            data_key="data",
            reference_key="data",
            reference_type="irf",
            fit_mode="irf_shift",
            fit_type="curve_fit_circular",
            C_ref=1.0,
            regularization=2,
            clean_irf=True,
            irf_corrections_type="each",
            channel_skew_source="irf",
            channel_skew_fit_reference_channel=1,
            channel_skew_fit_upsampling=5,
            channel_skew_fit_apodize=True,
            overwrite=True,
            create_output=False,
            channels=[0, 1, 2],
            calibration_key="cal",
        ),
        "options",
    ),
    Case(
        f"{CAL}:calibrate_h5_file",
        _wrapper(
            param_names=["C", "dT", "tau"],
            model_fn=_custom_model,
            p0=[1.0, 0.0, 2.0],
            output_options={"include_extra_sum": False},
        ),
        "custom model and output options",
    ),
    Case(
        f"{CAL}:calibrate_h5_file",
        _wrapper_error(parameter_names=["a", "b", "c"], param_names=["C", "dT", "tau"]),
        "parameter names disagree",
    ),
    Case(
        f"{CAL}:calibrate_h5_file",
        _wrapper_error(unknown_option=1),
        "unknown option",
    ),
    Case(f"{CAL}:show_h5_structure", _structure(), "default"),
    Case(
        f"{CAL}:show_h5_structure",
        _structure(attrs_inline=True, max_attr_len=30),
        "inline attributes",
    ),
    Case(
        f"{CAL}:show_h5_structure",
        _structure(include_attrs=False, tree_chars=False),
        "without attributes",
    ),
    Case(
        f"{CAL}:show_h5_structure",
        _structure(max_depth=1, tree_chars=False, attrs_inline=True),
        "depth limit",
    ),
    Case(
        f"{CAL}:show_h5_structure",
        _structure(max_depth=2, max_attr_len=12),
        "depth two with short attributes",
    ),
    Case(f"{CAL}:show_h5_structure_html", _structure_html(), "default"),
    Case(
        f"{CAL}:show_h5_structure_html",
        _structure_html(attrs_inline=False, show_full_path=False, max_attr_len=20),
        "attribute lists",
    ),
    Case(
        f"{CAL}:show_h5_structure_html",
        _structure_html(include_attrs=False, display_output=False),
        "without attributes",
    ),
]

SKIPPED: dict[str, str] = {}
