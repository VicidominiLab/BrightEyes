"""Read-only helpers for BrightEyes MCS HDF5 files."""

from __future__ import annotations

from . import reader, reader_legacy
from .reader import (
    DatasetInfo,
    Signal,
    is_brightemcs_h5,
    is_brighteyes_mcs_h5,
    list_datasets,
    load_default_irf,
    load_default_output_spad,
    load_default_ref,
    load_raw,
    load_virtual_channel,
    read_signal,
)
from .reader_legacy import (
    MCSMetadata,
    load,
    metadata,
    metadata_load,
    metadata_print,
)

__all__ = [
    "DatasetInfo",
    "MCSMetadata",
    "Signal",
    "is_brightemcs_h5",
    "is_brighteyes_mcs_h5",
    "list_datasets",
    "load",
    "load_default_irf",
    "load_default_output_spad",
    "load_default_ref",
    "load_raw",
    "load_virtual_channel",
    "metadata",
    "metadata_load",
    "metadata_print",
    "read_signal",
    "reader",
    "reader_legacy",
]
