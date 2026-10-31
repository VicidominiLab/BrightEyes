"""Equivalence cases for `brighteyes_mcs_reader`."""

import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from _capture import error as _error
from _capture import printed as _printed
from _capture import warned as _warned
from _inputs import mcs_h5_file, mcs_legacy_h5_file, mcs_output_h5_file
from _registry import Case

READER = "brighteyes_mcs_reader.reader"
LEGACY = "brighteyes_mcs_reader.reader_legacy"

PROPERTIES = (
    "pxdwelltime",
    "frametime",
    "framerate",
    "dx",
    "dy",
    "dz",
    "pxszizes",
    "pxsizes",
    "nmicroim",
    "ndatapoints",
    "duration",
    "dfd_freq",
    "dfd_nbins",
)


def _run(build: Callable[[Path], Path], use: Callable[[Any, Path], Any]):
    def call(f: Callable[..., Any]) -> Any:
        with tempfile.TemporaryDirectory() as folder:
            return use(f, build(Path(folder) / "file.h5"))

    return call


def _with(build: Callable[[Path], Path], **kwargs: Any):
    return _run(build, lambda f, path: f(path, **kwargs))


def _fails(build: Callable[[Path], Path], *args: Any, **kwargs: Any):
    return _run(build, lambda f, path: _error(lambda: f(path, *args, **kwargs)))


def _properties(metadata: Any) -> dict[str, Any]:
    return {
        name: _warned(lambda name=name: getattr(metadata, name)) for name in PROPERTIES
    }


def _metadata_of(build: Callable[[Path], Path]):
    def use(cls: Callable[..., Any], path: Path) -> Any:
        with h5py.File(path, "r") as handle:
            metadata = cls(handle)
        return metadata, _properties(metadata)

    return _run(build, use)


def _printed_metadata(f: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = mcs_legacy_h5_file(Path(folder) / "file.h5")
        with h5py.File(path, "r") as handle:
            metadata = import_module(f.__module__).MCSMetadata(handle)
        return _printed(lambda: f(metadata))


def _loaded_h5(f: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = mcs_legacy_h5_file(Path(folder) / "file.h5")
        dataset, metadata = f(path, key="data_channels_extra", data_format="h5")
        try:
            content = (dataset.name, dataset.shape, dataset[...])
        finally:
            dataset.file.close()
        return content, metadata, _properties(metadata)


def _open_source(f: Callable[..., Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = mcs_h5_file(Path(folder) / "file.h5")
        with h5py.File(path, "r") as handle:
            return f(handle, "aux")


def _signal(cls: Callable[..., Any]) -> Any:
    signal = cls(
        np.arange(6.0).reshape(1, 2, 3),
        ("Y", "X", "H"),
        {"H": np.arange(3) * 0.5},
        {"samples": 3},
    )
    return signal, cls(np.zeros(2), ("H",))


def _info(cls: Callable[..., Any]) -> Any:
    full = cls(
        "/raw/spad",
        "Raw view: raw/spad",
        (1, 2, 3),
        ("y", "x", "time_bin"),
        "reference",
        True,
    )
    return full, cls("/raw/aux", "aux", (4,))


def _bit_files(f: Callable[..., Any]) -> Any:
    names = ("DFD_80M200.bit", "x/y\\run_40M91.bit", "none.bit", "5M5.bit")
    return [_warned(lambda name=name: f(name)) for name in names] + [
        f("DFD_80M200.bit", default_cycle_mhz=20)
    ]


CASES: list[Case] = [
    Case(f"{READER}:Signal", _signal),
    Case(f"{READER}:DatasetInfo", _info),
    Case(f"{READER}:is_brightemcs_h5", _with(mcs_h5_file), "current schema"),
    Case(f"{READER}:is_brightemcs_h5", _with(mcs_legacy_h5_file), "legacy layout"),
    Case(f"{READER}:is_brighteyes_mcs_h5", _with(mcs_output_h5_file), "current schema"),
    Case(f"{READER}:is_brighteyes_mcs_h5", _with(mcs_legacy_h5_file), "legacy layout"),
    Case(f"{READER}:list_datasets", _with(mcs_h5_file), "raw only"),
    Case(f"{READER}:list_datasets", _with(mcs_output_h5_file), "with output"),
    Case(f"{READER}:list_datasets", _fails(mcs_legacy_h5_file), "legacy layout"),
    Case(f"{READER}:read_signal", _with(mcs_h5_file), "default"),
    Case(
        f"{READER}:read_signal",
        _with(mcs_h5_file, channel=2, time=0, depth=0),
        "channel",
    ),
    Case(f"{READER}:read_signal", _with(mcs_h5_file, channel=-1), "last channel"),
    Case(f"{READER}:read_signal", _with(mcs_h5_file, channel="sum"), "channel sum"),
    Case(
        f"{READER}:read_signal",
        _with(mcs_h5_file, dataset="/raw/aux", channel=1),
        "aux dataset",
    ),
    Case(
        f"{READER}:read_signal",
        _with(mcs_h5_file, dataset="raw/aux", channel="sum"),
        "aux sum",
    ),
    Case(f"{READER}:read_signal", _with(mcs_output_h5_file), "output default"),
    Case(
        f"{READER}:read_signal",
        _with(mcs_output_h5_file, dataset="output/virtual_channels/spad/channel_1"),
        "virtual channel path",
    ),
    Case(
        f"{READER}:read_signal",
        _with(mcs_output_h5_file, dataset="reference", channel=1),
        "reference channel",
    ),
    Case(
        f"{READER}:read_signal",
        _with(mcs_output_h5_file, dataset="reference", channel="sum"),
        "reference sum",
    ),
    Case(
        f"{READER}:read_signal",
        _with(mcs_output_h5_file, dataset="irf", channel=None),
        "irf",
    ),
    Case(
        f"{READER}:read_signal",
        _fails(mcs_h5_file, channel=9),
        "channel out of range",
    ),
    Case(
        f"{READER}:read_signal",
        _fails(mcs_h5_file, dataset="/raw/missing"),
        "missing dataset",
    ),
    Case(f"{READER}:read_signal", _fails(mcs_h5_file, time=3), "time out of range"),
    Case(f"{READER}:read_signal", _fails(mcs_legacy_h5_file), "legacy layout"),
    Case(f"{READER}:load_raw", _with(mcs_h5_file), "spad"),
    Case(f"{READER}:load_raw", _with(mcs_h5_file, kind="aux"), "aux"),
    Case(
        f"{READER}:load_raw",
        _with(
            mcs_h5_file,
            kind="spad",
            selection=(0, 0, slice(1, 3), slice(None), slice(0, 4), 2),
        ),
        "selection",
    ),
    Case(f"{READER}:load_raw", _open_source, "open file"),
    Case(f"{READER}:load_raw", _fails(mcs_h5_file, kind="analog"), "missing analog"),
    Case(f"{READER}:load_raw", _fails(mcs_h5_file, kind="other"), "invalid kind"),
    Case(f"{READER}:load_default_output_spad", _with(mcs_output_h5_file), "default"),
    Case(
        f"{READER}:load_default_output_spad",
        _with(mcs_output_h5_file, selection=(0, 0, 1)),
        "selection",
    ),
    Case(
        f"{READER}:load_default_output_spad",
        _fails(mcs_h5_file),
        "no output group",
    ),
    Case(
        f"{READER}:load_virtual_channel",
        _run(mcs_output_h5_file, lambda f, path: f(path, "spad", 2)),
        "spad",
    ),
    Case(
        f"{READER}:load_virtual_channel",
        _run(
            mcs_output_h5_file,
            lambda f, path: f(path, "aux", 1, selection=(0, 0, slice(0, 2))),
        ),
        "aux selection",
    ),
    Case(
        f"{READER}:load_virtual_channel",
        _fails(mcs_output_h5_file, "spad", -1),
        "negative index",
    ),
    Case(
        f"{READER}:load_virtual_channel",
        _fails(mcs_output_h5_file, "spad", "x"),
        "non-integer index",
    ),
    Case(
        f"{READER}:load_virtual_channel",
        _fails(mcs_output_h5_file, "spad", 9),
        "missing channel",
    ),
    Case(f"{READER}:load_default_irf", _with(mcs_output_h5_file), "default"),
    Case(
        f"{READER}:load_default_irf",
        _with(mcs_output_h5_file, selection=slice(2, 6)),
        "selection",
    ),
    Case(f"{READER}:load_default_irf", _fails(mcs_h5_file), "no output group"),
    Case(f"{READER}:load_default_ref", _with(mcs_output_h5_file), "default"),
    Case(
        f"{READER}:load_default_ref",
        _with(mcs_output_h5_file, selection=(slice(None), 1)),
        "selection",
    ),
    Case(f"{READER}:load_default_ref", _fails(mcs_h5_file), "no output group"),
    Case(f"{LEGACY}:MCSMetadata", _metadata_of(mcs_h5_file), "current schema"),
    Case(f"{LEGACY}:MCSMetadata", _metadata_of(mcs_legacy_h5_file), "legacy layout"),
    Case(f"{LEGACY}:metadata", _metadata_of(mcs_legacy_h5_file), "alias"),
    Case(
        f"{LEGACY}:MCSMetadata.parse_dfd_metadata_from_bitfile_name",
        _bit_files,
        "bit file names",
    ),
    Case(f"{LEGACY}:MCSMetadata.Print", _printed_metadata),
    Case(
        f"{LEGACY}:metadata_load",
        _run(mcs_legacy_h5_file, lambda f, path: _properties(f(path))),
        "legacy layout",
    ),
    Case(
        f"{LEGACY}:metadata_load",
        _run(mcs_h5_file, lambda f, path: _properties(f(path))),
        "current schema",
    ),
    Case(
        f"{LEGACY}:metadata_print",
        _run(mcs_legacy_h5_file, lambda f, path: _printed(lambda: f(path))),
    ),
    Case(f"{LEGACY}:load", _with(mcs_h5_file), "default key"),
    Case(f"{LEGACY}:load", _with(mcs_h5_file, key="aux"), "aux key"),
    Case(f"{LEGACY}:load", _with(mcs_legacy_h5_file, key="data"), "legacy data"),
    Case(
        f"{LEGACY}:load",
        _with(mcs_legacy_h5_file, key="data_channels_extra"),
        "legacy extra",
    ),
    Case(f"{LEGACY}:load", _loaded_h5, "h5 format"),
    Case(f"{LEGACY}:load", _fails(mcs_h5_file, data_format="other"), "invalid format"),
    Case(f"{LEGACY}:load", _fails(mcs_h5_file, key="analog"), "missing key"),
]

SKIPPED: dict[str, str] = {}
