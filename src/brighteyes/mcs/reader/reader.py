# Origin: VicidominiLab/BrightEyes-MCS-Reader, src/brighteyes_mcs_reader/reader.py at 8b81d62.
"""Read-only BrightEyes-MCS HDF5 reader."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

ChannelSelection = int | Literal["sum"] | None

SCHEMA_NAME = "brighteyes_mcs_file"
SCHEMA_VERSION = "0.0.6"

BRIGHTEYES_H5_DATA_PATH = "/raw"
BRIGHTEYES_H5_OUTPUT_PATH = "/output"
SUM_IRF_TRACE_OUTPUT_KIND = "sum_irf_trace"
SUM_REFERENCE_TRACE_OUTPUT_KIND = "sum_reference_trace"

VALID_RAW_KINDS = {"spad", "aux", "analog"}
VALID_VIRTUAL_CHANNEL_KINDS = VALID_RAW_KINDS

_RAW_DATASETS = ("raw/spad", "raw/aux", "raw/analog")
_TIME_AXES = {"repetition", "rep", "r"}
_DEPTH_AXES = {"z", "depth"}
_Y_AXES = {"y"}
_X_AXES = {"x"}
_HISTOGRAM_AXES = {"time_bin", "time", "histogram", "histogram_bin", "h"}
_CHANNEL_AXES = {
    "channel",
    "detector_channel",
    "spad_channel",
    "aux_channel",
    "analog_channel",
    "c",
}
_CIRCULAR_REPETITION_AXES = {"circular_repetition", "circular_rep"}
_CIRCULAR_POINT_AXES = {"circular_point"}

_AXIS_FALLBACKS = {
    3: ("y", "x", "time_bin"),
    5: ("repetition", "z", "y", "x", "time_bin"),
    6: ("repetition", "z", "y", "x", "time_bin", "channel"),
    8: (
        "repetition",
        "z",
        "y",
        "x",
        "circular_repetition",
        "circular_point",
        "time_bin",
        "channel",
    ),
}


@dataclass(frozen=True)
class Signal:
    """Histogram signal returned by :func:`read_signal`."""

    data: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetInfo:
    """Information about a readable BrightEyes-MCS HDF5 dataset."""

    path: str
    label: str
    shape: tuple[int, ...]
    axes: tuple[str, ...] = ()
    kind: Literal["data", "reference", "irf"] = "data"
    is_default: bool = False


def is_brightemcs_h5(path: str | PathLike[Any]) -> bool:
    """Return whether *path* is a BrightEyes-MCS HDF5 file."""
    with h5py.File(path, "r") as handle:
        return _is_brighteyes_handle(handle)


def is_brighteyes_mcs_h5(path: str | PathLike[Any]) -> bool:
    """Return whether *path* is a BrightEyes-MCS HDF5 file."""
    return is_brightemcs_h5(path)


def list_datasets(path: str | PathLike[Any]) -> list[DatasetInfo]:
    """Return readable datasets in a BrightEyes-MCS HDF5 file."""
    with h5py.File(path, "r") as handle:
        _require_brightemcs(handle)
        infos: list[DatasetInfo] = []
        seen: set[tuple[str, str]] = set()

        default_data = _default_data_path(handle, required=False)
        default_ref = _default_trace_path(
            handle, "reference", required=False
        )
        default_irf = _default_trace_path(handle, "irf", required=False)

        if default_data is not None:
            _append_dataset_info(
                infos,
                seen,
                handle,
                default_data,
                kind="data",
                is_default=True,
            )

        _append_output_product_infos(infos, seen, handle, default_data)
        _append_virtual_channel_infos(infos, seen, handle)

        for raw_path in _RAW_DATASETS:
            if raw_path in handle and isinstance(
                handle[raw_path], h5py.Dataset
            ):
                _append_dataset_info(
                    infos,
                    seen,
                    handle,
                    raw_path,
                    kind="data",
                    is_default=_same_h5_path(default_data, raw_path),
                )

        if default_ref is not None:
            _append_dataset_info(
                infos,
                seen,
                handle,
                default_ref,
                kind="reference",
                is_default=True,
            )
        if default_irf is not None:
            _append_dataset_info(
                infos,
                seen,
                handle,
                default_irf,
                kind="irf",
                is_default=True,
            )

        return infos


def read_signal(
    path: str | PathLike[Any],
    *,
    dataset: str | None = None,
    time: int = 0,
    depth: int = 0,
    channel: ChannelSelection = 0,
) -> Signal:
    """Return a BrightEyes-MCS histogram signal.

    Parameters
    ----------
    path:
        BrightEyes-MCS HDF5 file.
    dataset:
        Dataset to read. ``None`` selects the default image dataset.
        ``"reference"`` and ``"irf"`` select the default calibration traces.
        Other strings are interpreted as literal HDF5 paths.
    time:
        Index along the HDF5 repetition axis.
    depth:
        Index along the HDF5 z/depth axis.
    channel:
        Channel index, ``"sum"`` to sum a channel axis, or ``None`` for
        unchannelled data.
    """
    with h5py.File(path, "r") as handle:
        _require_brightemcs(handle)
        data, kind = _resolve_dataset(handle, dataset)

        if kind in {"reference", "irf"} or data.ndim in (1, 2):
            return _read_trace(handle, data, kind, channel)

        return _read_image(handle, data, time, depth, channel)


def load_raw(source, kind="spad", selection=None):
    """
    Load ``/raw/<kind>`` from a current-schema file.

    ``kind`` must be one of ``"spad"``, ``"aux"``, or ``"analog"``.
    ``selection`` is forwarded to the HDF5 dataset before loading.
    """
    kind = _validate_loader_kind(kind, "raw kind")
    with _open_loader_source(source) as handle:
        dataset = _loader_dataset_at(
            handle, f"{BRIGHTEYES_H5_DATA_PATH}/{kind}"
        )
        return _read_loader_dataset(dataset, selection)


def load_default_output_spad(source, selection=None):
    """
    Load the dataset selected by ``/output.attrs['default']``.

    Current-schema files store a complete product dataset path such as
    ``/output/sum_channels_001/products/spad``. Older files that stored only
    the output run id are still accepted and resolved to ``products/spad``.

    ``source`` may be a file path or an open ``h5py.File``. ``selection`` is
    forwarded to the HDF5 dataset before loading, so callers can avoid reading
    a full image stack when only one slice is needed.
    """
    with _open_loader_source(source) as handle:
        output = _loader_output_group(handle)
        dataset = _loader_output_product_dataset_from_default(
            handle,
            output,
            output.attrs.get("default", ""),
            output.attrs.get("default_run", ""),
            "spad",
        )
        run_group = _loader_output_run_group_for_product(
            handle, output, dataset
        )
        output_type = _loader_attr_to_string(
            run_group.attrs.get("output_type", "")
        ).strip()
        if output_type != "image_tool":
            raise ValueError(
                f"{run_group.name}.attrs['output_type'] must be 'image_tool'"
            )
        return _read_loader_dataset(dataset, selection)


def load_virtual_channel(source, kind, channel_index, selection=None):
    """
    Load ``/output/virtual_channels/<kind>/channel_<channel_index>``.

    ``kind`` must be one of ``"spad"``, ``"aux"``, or ``"analog"``. The
    loader is strict on the grouped current-schema layout and does not fall
    back to old flat virtual channel names.
    """
    kind = _validate_loader_kind(kind, "virtual channel kind")
    try:
        channel_index = int(channel_index)
    except (TypeError, ValueError) as exc:
        raise TypeError("channel_index must be an integer") from exc
    if channel_index < 0:
        raise ValueError("channel_index must be non-negative")

    path = (
        f"{BRIGHTEYES_H5_OUTPUT_PATH}/virtual_channels/"
        f"{kind}/channel_{channel_index}"
    )
    with _open_loader_source(source) as handle:
        dataset = _loader_dataset_at(handle, path)
        virtual_type = _loader_attr_to_string(
            dataset.attrs.get("virtual_channel_type", "")
        ).strip()
        if virtual_type != kind:
            raise ValueError(
                f"{dataset.name}.attrs['virtual_channel_type'] must be "
                f"{kind!r}, got {virtual_type!r}"
            )
        return _read_loader_dataset(dataset, selection)


def load_default_irf(source, selection=None):
    """Load the default summed IRF trace selected by ``/output`` attrs."""
    return _load_default_trace(
        source, SUM_IRF_TRACE_OUTPUT_KIND, selection=selection
    )


def load_default_ref(source, selection=None):
    """Load the default summed reference trace selected by ``/output`` attrs."""
    return _load_default_trace(
        source,
        SUM_REFERENCE_TRACE_OUTPUT_KIND,
        selection=selection,
    )


def _require_brightemcs(handle: h5py.File) -> None:
    if not _is_brighteyes_handle(handle):
        raise ValueError(
            f"{handle.filename!r} is not a BrightEyes-MCS HDF5 "
            f"{SCHEMA_VERSION} file"
        )


def _is_brighteyes_handle(handle: h5py.File) -> bool:
    attrs = _read_attrs(handle)
    if attrs.get("schema_name") == SCHEMA_NAME:
        return attrs.get("data_format_version") == SCHEMA_VERSION
    return (
        attrs.get("data_format_version") == SCHEMA_VERSION
        and "raw" in handle
        and "raw/metadata" in handle
        and "raw/axes" in handle
    )


def _resolve_dataset(
    handle: h5py.File, dataset: str | None
) -> tuple[h5py.Dataset, Literal["data", "reference", "irf"]]:
    if dataset is None:
        path = _default_data_path(handle, required=True)
        kind: Literal["data", "reference", "irf"] = "data"
    elif dataset == "reference":
        path = _default_trace_path(handle, "reference", required=True)
        kind = "reference"
    elif dataset == "irf":
        path = _default_trace_path(handle, "irf", required=True)
        kind = "irf"
    else:
        path = _normalize_path(dataset)
        kind = "data"

    node = _dataset_at(handle, path)
    return node, kind


def _default_data_path(
    handle: h5py.File, *, required: bool
) -> str | None:
    candidates = []
    if "output" in handle:
        output = handle["output"]
        if isinstance(output, h5py.Group):
            candidates.append(_as_text(output.attrs.get("default", "")))
            default_run = _as_text(output.attrs.get("default_run", ""))
            if default_run:
                candidates.append(f"{default_run}/products/spad")
    candidates.extend(
        [
            _as_text(handle.attrs.get("default", "")),
            "raw/spad",
        ]
    )
    return _first_dataset_path(handle, candidates, "default data", required)


def _default_trace_path(
    handle: h5py.File,
    trace: Literal["reference", "irf"],
    *,
    required: bool,
) -> str | None:
    if trace == "reference":
        attr = "default_ref_trace_id"
        candidates = [
            "output/sum_ref_001/products/trace",
            "calibration/results/spad/aligned/reference_trace",
        ]
        label = "default reference trace"
    else:
        attr = "default_irf_trace_id"
        candidates = [
            "output/sum_irf_001/products/trace",
            "calibration/results/spad/aligned/irf_trace",
        ]
        label = "default IRF trace"

    output_candidates = []
    if "output" in handle and isinstance(handle["output"], h5py.Group):
        output = handle["output"]
        selected = _as_text(output.attrs.get(attr, ""))
        if selected:
            output_candidates.append(selected)
            if "/" not in selected.strip("/"):
                output_candidates.append(f"output/{selected}/products/trace")
    return _first_dataset_path(
        handle, [*output_candidates, *candidates], label, required
    )


def _first_dataset_path(
    handle: h5py.File,
    paths: list[str],
    label: str,
    required: bool,
) -> str | None:
    for path in paths:
        if not path:
            continue
        key = _normalize_path(path)
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return key
    if required:
        raise ValueError(f"BrightEyes-MCS file has no readable {label}")
    return None


def _dataset_at(handle: h5py.File, path: str) -> h5py.Dataset:
    key = _normalize_path(path)
    if key not in handle:
        raise ValueError(f"HDF5 dataset {path!r} not found")
    node = handle[key]
    if not isinstance(node, h5py.Dataset):
        raise ValueError(f"HDF5 object {path!r} is not a dataset")
    return node


@contextmanager
def _open_loader_source(source):
    if isinstance(source, h5py.File):
        yield source
        return
    if isinstance(source, (str, bytes, Path)):
        with h5py.File(source, "r") as handle:
            yield handle
        return
    raise TypeError("source must be a file path or an open h5py.File")


def _loader_attr_to_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _loader_attr_to_string(value.item())
    return str(value)


def _loader_required_attr(attrs, name, owner_path):
    if name not in attrs:
        raise KeyError(
            f"{owner_path}.attrs[{name!r}] is required by the current scheme"
        )
    value = _loader_attr_to_string(attrs[name]).strip()
    if not value:
        raise KeyError(f"{owner_path}.attrs[{name!r}] is empty")
    return value


def _normalize_loader_path(path):
    path = _loader_attr_to_string(path).strip()
    if not path:
        raise KeyError("HDF5 path is empty")
    return "/" + path.strip("/")


def _loader_output_group(handle):
    key = BRIGHTEYES_H5_OUTPUT_PATH.strip("/")
    if key not in handle or not isinstance(handle[key], h5py.Group):
        raise KeyError(
            f"{BRIGHTEYES_H5_OUTPUT_PATH} group is required by the current "
            "scheme"
        )
    return handle[key]


def _loader_node_at(handle, path):
    path = _normalize_loader_path(path)
    key = path.strip("/")
    if key not in handle:
        raise KeyError(f"{path} is required by the current scheme")
    return handle[key]


def _loader_dataset_at(handle, path):
    path = _normalize_loader_path(path)
    node = _loader_node_at(handle, path)
    if not isinstance(node, h5py.Dataset):
        raise KeyError(f"{path} dataset is required by the current scheme")
    return node


def _read_loader_dataset(dataset, selection):
    if selection is None:
        return np.asarray(dataset[...])
    return np.asarray(dataset[selection])


def _validate_loader_kind(kind, label):
    kind = _loader_attr_to_string(kind).strip()
    if kind not in VALID_RAW_KINDS:
        allowed = ", ".join(sorted(VALID_RAW_KINDS))
        raise ValueError(f"{label} must be one of {allowed}; got {kind!r}")
    return kind


def _loader_output_product_dataset_from_default(
    handle,
    output_group,
    default,
    default_run,
    product_name,
):
    default = _loader_attr_to_string(default).strip()
    default_run = _loader_attr_to_string(default_run).strip()
    if default:
        if "/" not in default.strip("/"):
            default = f"{output_group.name}/{default}/products/{product_name}"
    elif default_run:
        default = f"{_normalize_loader_path(default_run)}/products/{product_name}"
    else:
        default = _loader_required_attr(
            {"default": default}, "default", output_group.name
        )
    dataset = _loader_dataset_at(handle, default)
    return dataset


def _loader_output_run_group_for_product(handle, output_group, dataset):
    output_key = output_group.name.strip("/")
    parts = dataset.name.strip("/").split("/")
    if len(parts) < 4 or parts[0] != output_key or parts[2] != "products":
        raise KeyError(
            f"{dataset.name} must be under "
            f"{output_group.name}/<run_id>/products"
        )
    run_path = "/" + "/".join(parts[:2])
    node = _loader_node_at(handle, run_path)
    if not isinstance(node, h5py.Group):
        raise KeyError(f"{run_path} group selected by /output default is missing")
    return node


def _load_default_trace(source, trace_kind, selection=None):
    id_attrs = {
        SUM_IRF_TRACE_OUTPUT_KIND: "default_irf_trace_id",
        SUM_REFERENCE_TRACE_OUTPUT_KIND: "default_ref_trace_id",
    }
    id_attr = id_attrs[trace_kind]
    with _open_loader_source(source) as handle:
        output = _loader_output_group(handle)
        trace_id = _loader_required_attr(output.attrs, id_attr, output.name)
        if "/" in trace_id.strip("/"):
            trace_path = trace_id
        else:
            trace_path = f"{output.name}/{trace_id}/products/trace"
        dataset = _loader_dataset_at(handle, trace_path)
        output_type = _loader_attr_to_string(
            dataset.attrs.get("output_type", "")
        ).strip()
        actual_trace_kind = _loader_attr_to_string(
            dataset.attrs.get("trace_kind", "")
        ).strip()
        if output_type != "trace":
            raise ValueError(
                f"{dataset.name}.attrs['output_type'] must be 'trace'"
            )
        if actual_trace_kind != trace_kind:
            raise ValueError(
                f"{dataset.name}.attrs['trace_kind'] must be "
                f"{trace_kind!r}, got {actual_trace_kind!r}"
            )
        return _read_loader_dataset(dataset, selection)


def _read_image(
    handle: h5py.File,
    dataset: h5py.Dataset,
    time: int,
    depth: int,
    channel: ChannelSelection,
) -> Signal:
    axes = _axis_order(dataset)
    if not axes:
        raise ValueError(
            f"HDF5 dataset {dataset.name!r} has no supported axis metadata"
        )

    y_axis = _required_axis_index(axes, _Y_AXES, "y axis")
    x_axis = _required_axis_index(axes, _X_AXES, "x axis")
    h_axis = _required_axis_index(axes, _HISTOGRAM_AXES, "time-bin axis")
    time_axis = _axis_index(axes, _TIME_AXES)
    depth_axis = _axis_index(axes, _DEPTH_AXES)
    channel_axis = _axis_index(axes, _CHANNEL_AXES)
    q_axis = _axis_index(axes, _CIRCULAR_REPETITION_AXES)
    p_axis = _axis_index(axes, _CIRCULAR_POINT_AXES)

    key: list[Any] = [slice(None)] * dataset.ndim
    selection: dict[str, Any] = {}

    if time_axis is None:
        if time not in (0, None):
            raise IndexError(
                "time is out of bounds for data without time axis"
            )
    else:
        time = _as_index(time, dataset.shape[time_axis], "time")
        key[time_axis] = time
        selection["time"] = time

    if depth_axis is None:
        if depth not in (0, None):
            raise IndexError(
                "depth is out of bounds for data without depth axis"
            )
    else:
        depth = _as_index(depth, dataset.shape[depth_axis], "depth")
        key[depth_axis] = depth
        selection["depth"] = depth

    output_axes = [y_axis, x_axis]
    dims = ["Y", "X"]
    if q_axis is not None:
        output_axes.append(q_axis)
        dims.append("Q")
    if p_axis is not None:
        output_axes.append(p_axis)
        dims.append("P")

    if channel_axis is None:
        if channel not in (0, "sum", None):
            raise IndexError(
                "channel is out of bounds for data without channel axis"
            )
        channel_selection: int | str | None = None
    elif channel == "sum":
        channel_selection = "sum"
    elif channel is None:
        raise ValueError("channel must be an integer or 'sum'")
    else:
        channel = _as_index(channel, dataset.shape[channel_axis], "channel")
        key[channel_axis] = channel
        channel_selection = channel
    selection["channel"] = channel_selection

    output_axes.append(h_axis)
    dims.append("H")

    data = np.asarray(dataset[tuple(key)])
    kept_axes = [
        axis
        for axis in range(dataset.ndim)
        if isinstance(key[axis], slice)
    ]
    if channel_axis is not None and channel_selection == "sum":
        current_axis = kept_axes.index(channel_axis)
        data = data.sum(axis=current_axis)
        kept_axes.pop(current_axis)

    transpose_axes = [kept_axes.index(axis) for axis in output_axes]
    if transpose_axes != list(range(data.ndim)):
        data = np.transpose(data, transpose_axes)

    attrs = _signal_attrs(
        handle,
        dataset,
        axes,
        selection,
        samples=dataset.shape[h_axis],
        kind="data",
    )
    coords = {
        "H": _histogram_coordinates(handle, dataset, dataset.shape[h_axis])
    }
    return Signal(data, tuple(dims), coords, attrs)


def _read_trace(
    handle: h5py.File,
    dataset: h5py.Dataset,
    kind: Literal["data", "reference", "irf"],
    channel: ChannelSelection,
) -> Signal:
    if dataset.ndim not in (1, 2):
        raise ValueError(
            f"HDF5 trace dataset {dataset.name!r} has shape "
            f"{dataset.shape!r}; expected (time,) or (time, channel)"
        )

    shape = tuple(int(size) for size in dataset.shape)
    if dataset.ndim == 1:
        if channel not in (0, "sum", None):
            raise IndexError(
                "channel is out of bounds for unchannelled trace data"
            )
        channel_selection: int | str | None = None
        data = np.asarray(dataset[:]).reshape(1, 1, shape[0])
        axes = ("time_bin",)
    elif channel == "sum":
        channel_selection = "sum"
        data = np.asarray(dataset[:, :]).sum(axis=1).reshape(1, 1, shape[0])
        axes = ("time_bin", "channel")
    elif channel is None:
        raise ValueError("channel must be an integer or 'sum'")
    else:
        channel = _as_index(channel, shape[1], "channel")
        channel_selection = channel
        data = np.asarray(dataset[:, channel]).reshape(1, 1, shape[0])
        axes = ("time_bin", "channel")

    trace_kind = _trace_kind(dataset, kind)
    selection = {
        "dataset": trace_kind,
        "channel": channel_selection,
    }
    attrs = _signal_attrs(
        handle,
        dataset,
        axes,
        selection,
        samples=shape[0],
        kind=trace_kind,
    )
    if trace_kind in {"reference", "irf"}:
        attrs["reference"] = True
        attrs["irf"] = trace_kind == "irf"

    coords = {"H": _histogram_coordinates(handle, dataset, shape[0])}
    return Signal(data, ("Y", "X", "H"), coords, attrs)


def _trace_kind(
    dataset: h5py.Dataset, kind: Literal["data", "reference", "irf"]
) -> Literal["data", "reference", "irf"]:
    if kind in {"reference", "irf"}:
        return kind
    path = dataset.name.lower()
    attrs = _read_attrs(dataset)
    value = str(attrs.get("trace_kind", "")).lower()
    if "irf" in value or "irf" in path:
        return "irf"
    if "reference" in value or "ref" in value or "reference" in path:
        return "reference"
    return "data"


def _signal_attrs(
    handle: h5py.File,
    dataset: h5py.Dataset,
    axes: tuple[str, ...],
    selection: dict[str, Any],
    *,
    samples: int,
    kind: str,
) -> dict[str, Any]:
    data_attrs = _read_attrs(dataset)
    parent_attrs = _read_attrs(dataset.parent)
    metadata_attrs = _metadata_attrs(handle, dataset, parent_attrs, data_attrs)
    timing_attrs = _read_first_attrs(
        handle,
        "raw/metadata/acquisition/timing",
        f"{_run_path(dataset)}/metadata",
    )
    calibration_attrs = _calibration_attrs(handle, dataset)
    file_attrs = _read_attrs(handle)

    attrs: dict[str, Any] = {
        "h5_dataset": dataset.name,
        "h5_shape": tuple(int(size) for size in dataset.shape),
        "h5_axes": axes,
        "h5_selection": selection,
        "h5_kind": kind,
        "h5_attrs": {
            "file": file_attrs,
            "group": parent_attrs,
            "metadata": metadata_attrs,
            "timing": timing_attrs,
            "calibration": calibration_attrs,
            "data": data_attrs,
        },
        "samples": int(samples),
    }

    frequency = _first_attr(
        data_attrs,
        parent_attrs,
        metadata_attrs,
        timing_attrs,
        calibration_attrs,
        file_attrs,
        names=("frequency", "Frequency", "laser_frequency_mhz"),
    )
    if frequency is not None:
        attrs["frequency"] = frequency

    lifetime = _reference_lifetime_ns(
        handle,
        data_attrs,
        parent_attrs,
        metadata_attrs,
        timing_attrs,
        calibration_attrs,
    )
    if lifetime is not None:
        attrs["reference_lifetime_ns"] = lifetime

    return attrs


def _append_output_product_infos(
    infos: list[DatasetInfo],
    seen: set[tuple[str, str]],
    handle: h5py.File,
    default_data: str | None,
) -> None:
    if "output" not in handle or not isinstance(handle["output"], h5py.Group):
        return
    output = handle["output"]
    for run_id in sorted(output):
        run = output[run_id]
        if not isinstance(run, h5py.Group) or "products" not in run:
            continue
        products = run["products"]
        if not isinstance(products, h5py.Group):
            continue
        for name in sorted(products):
            node = products[name]
            if not isinstance(node, h5py.Dataset):
                continue
            kind = _dataset_info_kind(node)
            if kind != "data":
                continue
            _append_dataset_info(
                infos,
                seen,
                handle,
                node.name,
                kind=kind,
                is_default=_same_h5_path(default_data, node.name),
            )


def _append_virtual_channel_infos(
    infos: list[DatasetInfo],
    seen: set[tuple[str, str]],
    handle: h5py.File,
) -> None:
    if "output/virtual_channels" in handle:
        group = handle["output/virtual_channels"]
        if isinstance(group, h5py.Group):
            for kind in sorted(group):
                subgroup = group[kind]
                if not isinstance(subgroup, h5py.Group):
                    continue
                for name in sorted(subgroup):
                    node = subgroup[name]
                    if isinstance(node, h5py.Dataset):
                        _append_dataset_info(
                            infos,
                            seen,
                            handle,
                            node.name,
                            kind="data",
                            is_default=False,
                        )

    if "output" not in handle or not isinstance(handle["output"], h5py.Group):
        return
    for name in sorted(handle["output"]):
        if not name.startswith(("data_channel_", "data_aux_channel_")):
            continue
        node = handle["output"][name]
        if isinstance(node, h5py.Dataset):
            _append_dataset_info(
                infos,
                seen,
                handle,
                node.name,
                kind="data",
                is_default=False,
            )


def _append_dataset_info(
    infos: list[DatasetInfo],
    seen: set[tuple[str, str]],
    handle: h5py.File,
    path: str,
    *,
    kind: Literal["data", "reference", "irf"],
    is_default: bool,
) -> None:
    dataset = _dataset_at(handle, path)
    key = (_normalize_path(dataset.name), kind)
    if key in seen:
        return
    seen.add(key)

    axes = _axis_order(dataset)
    infos.append(
        DatasetInfo(
            path=dataset.name,
            label=_dataset_label(dataset, kind, is_default),
            shape=tuple(int(size) for size in dataset.shape),
            axes=axes,
            kind=kind,
            is_default=is_default,
        )
    )


def _dataset_info_kind(
    dataset: h5py.Dataset,
) -> Literal["data", "reference", "irf"]:
    attrs = _read_attrs(dataset)
    output_type = str(attrs.get("output_type", "")).strip().lower()
    trace_kind = str(attrs.get("trace_kind", "")).strip().lower()
    path = dataset.name.lower()
    if output_type == "trace" or dataset.ndim in (1, 2):
        if "irf" in trace_kind or "irf" in path:
            return "irf"
        if "reference" in trace_kind or "ref" in trace_kind:
            return "reference"
    return "data"


def _dataset_label(
    dataset: h5py.Dataset,
    kind: Literal["data", "reference", "irf"],
    is_default: bool,
) -> str:
    path = dataset.name.strip("/")
    if kind == "reference":
        label = "Reference"
    elif kind == "irf":
        label = "IRF"
    elif path.startswith("output/virtual_channels/"):
        parts = path.split("/")
        label = f"Raw view: {parts[-2]}/{parts[-1]}"
    elif path.startswith("output/") and "/products/" in path:
        parts = path.split("/")
        label = (
            f"output/{parts[1]}"
            if len(parts) >= 4 and parts[3] == "image"
            else path
        )
    elif path.startswith("raw/"):
        label = f"Raw view: {path}"
    else:
        label = path

    if is_default and kind == "data":
        return f"{label} (default)"
    return label


def _axis_order(dataset: h5py.Dataset) -> tuple[str, ...]:
    attrs = _read_attrs(dataset)
    value = attrs.get("axis_order") or attrs.get("axes")
    if isinstance(value, str):
        axes = tuple(axis.strip() for axis in value.split(","))
        if len(axes) == dataset.ndim and all(axes):
            return axes

    axes = []
    for index in range(dataset.ndim):
        axis = attrs.get(f"axis_{index}")
        if not isinstance(axis, str) or not axis:
            axes = []
            break
        axes.append(axis)
    if len(axes) == dataset.ndim:
        return tuple(axes)

    fallback = _AXIS_FALLBACKS.get(dataset.ndim)
    if fallback is not None:
        return fallback
    return ()


def _axis_index(axes: tuple[str, ...], names: set[str]) -> int | None:
    for index, axis in enumerate(axes):
        if str(axis).strip().lower() in names:
            return index
    return None


def _required_axis_index(
    axes: tuple[str, ...], names: set[str], description: str
) -> int:
    index = _axis_index(axes, names)
    if index is None:
        raise ValueError(f"HDF5 axes {axes!r} do not contain {description}")
    return index


def _histogram_coordinates(
    handle: h5py.File, dataset: h5py.Dataset, size: int
) -> np.ndarray:
    attrs = _read_attrs(dataset)
    for name in ("H", "h", "T", "t", "time", "times", "histogram_bins"):
        if name in attrs:
            coord = np.asarray(attrs[name])
            if coord.ndim == 1 and coord.size == size:
                return coord

    for name in (
        "time_axis_path",
        "source_time_axis_path",
        "time_axis_source",
    ):
        coord = _axis_coordinates(handle, attrs.get(name), size)
        if coord is not None:
            return coord

    for path in _default_time_axis_paths(dataset):
        coord = _axis_coordinates(handle, path, size)
        if coord is not None:
            return coord

    all_attrs = [
        attrs,
        _read_attrs(dataset.parent),
        _metadata_attrs(handle, dataset, _read_attrs(dataset.parent), attrs),
        _read_first_attrs(handle, "raw/metadata/acquisition/timing"),
        _calibration_attrs(handle, dataset),
    ]
    time_bin = _first_attr(
        *all_attrs,
        names=(
            "time_bin_ns",
            "digital_time_bin_ns",
            "timebin_in_ns",
            "bin_width_ns",
        ),
    )
    if time_bin is not None:
        try:
            time_bin_float = float(time_bin)
        except (TypeError, ValueError):
            time_bin_float = np.nan
        if np.isfinite(time_bin_float):
            return np.arange(size, dtype=np.float64) * time_bin_float

    return np.arange(size)


def _axis_coordinates(
    handle: h5py.File, path: Any, size: int
) -> np.ndarray | None:
    if not isinstance(path, str) or not path.strip("/"):
        return None
    key = _normalize_path(path)
    if key not in handle:
        return None
    coord = np.asarray(handle[key])
    if coord.ndim == 1 and coord.size == size:
        return coord
    return None


def _default_time_axis_paths(dataset: h5py.Dataset) -> tuple[str, ...]:
    path = dataset.name.strip("/")
    if path.startswith("raw/"):
        return ("raw/axes/digital_time_ns",)
    if path.startswith("calibration/"):
        return ("calibration/axes/time_ns",)
    if path.startswith("output/"):
        parts = path.split("/")
        if len(parts) > 1:
            return (f"output/{parts[1]}/axes/time_ns",)
    return ()


def _metadata_attrs(
    handle: h5py.File,
    dataset: h5py.Dataset,
    parent_attrs: dict[str, Any],
    data_attrs: dict[str, Any],
) -> dict[str, Any]:
    return _read_first_attrs(
        handle,
        data_attrs.get("metadata_path", ""),
        parent_attrs.get("metadata_path", ""),
        f"{_run_path(dataset)}/metadata",
        "raw/metadata",
    )


def _calibration_attrs(
    handle: h5py.File, dataset: h5py.Dataset
) -> dict[str, Any]:
    path = dataset.name.strip("/")
    if path.startswith("calibration/results/"):
        parts = path.split("/")
        if len(parts) >= 3:
            return _read_first_attrs(handle, "/".join(parts[:3]))

    product = _product_name(dataset)
    if product:
        return _read_first_attrs(handle, f"calibration/results/{product}")
    return {}


def _reference_lifetime_ns(
    handle: h5py.File, *attrs: dict[str, Any]
) -> float | None:
    value = _first_attr(
        *attrs,
        names=(
            "reference_lifetime_ns",
            "tau_reference_ns",
            "tau_ref_input_ns",
        ),
    )
    lifetime = _first_finite_float(value)
    if lifetime is not None:
        return lifetime

    for product in ("spad", "aux"):
        path = f"calibration/results/{product}/fit/tau_reference_ns"
        if path not in handle:
            continue
        lifetime = _first_finite_float(handle[path][()])
        if lifetime is not None:
            return lifetime
    return None


def _first_attr(*attrs: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for attr in attrs:
        for name in names:
            if name in attr:
                return attr[name]
    return None


def _first_finite_float(value: Any) -> float | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = array[np.isfinite(array) & (array > 0.0)]
    if finite.size == 0:
        return None
    return float(finite[0])


def _product_name(dataset: h5py.Dataset) -> str | None:
    attrs = _read_attrs(dataset)
    product = attrs.get("product_name")
    if isinstance(product, str) and product:
        return product
    source_key = attrs.get("source_key") or attrs.get("source_data_key")
    if source_key == "data":
        return "spad"
    if source_key == "data_channels_extra":
        return "aux"

    path = dataset.name.strip("/")
    if path.startswith("raw/spad") or path.endswith("/products/spad"):
        return "spad"
    if path.startswith("raw/aux") or path.endswith("/products/aux"):
        return "aux"
    return None


def _run_path(dataset: h5py.Dataset) -> str:
    parts = dataset.name.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "output":
        return "/".join(parts[:2])
    return ""


def _read_first_attrs(handle: h5py.File, *paths: Any) -> dict[str, Any]:
    for path in paths:
        if not isinstance(path, str) or not path.strip("/"):
            continue
        key = _normalize_path(path)
        if key in handle:
            return _read_attrs(handle[key])
    return {}


def _read_attrs(obj: Any) -> dict[str, Any]:
    return {
        str(key): _normalize_value(value) for key, value in obj.attrs.items()
    }


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _normalize_value(value[()])
        if value.dtype.kind == "S":
            return [_normalize_value(item) for item in value.tolist()]
        if value.dtype.kind == "U":
            return [str(item) for item in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return _normalize_value(value.item())
    return value


def _normalize_path(path: str) -> str:
    key = str(path).strip()
    if not key:
        raise ValueError("HDF5 path is empty")
    return key.strip("/")


def _as_text(value: Any) -> str:
    value = _normalize_value(value)
    return value if isinstance(value, str) else str(value)


def _as_index(index: int, size: int, name: str) -> int:
    try:
        index = int(index)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if index < 0:
        index += size
    if index < 0 or index >= size:
        raise IndexError(f"{name}={index!r} is out of bounds [0, {size - 1}]")
    return index


def _same_h5_path(first: str | None, second: str | None) -> bool:
    if first is None or second is None:
        return False
    return _normalize_path(first) == _normalize_path(second)
