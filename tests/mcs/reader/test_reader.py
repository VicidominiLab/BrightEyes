# Origin: VicidominiLab/BrightEyes-MCS-Reader, tests/test_reader.py at 8b81d62.
"""Tests for the BrightEyes-MCS HDF5 reader."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from brighteyes.mcs.reader import (
    is_brightemcs_h5,
    list_datasets,
    read_signal,
)

h5py = pytest.importorskip("h5py")


def _write_h5(path):
    raw = np.arange(2 * 3 * 4 * 5 * 8 * 2, dtype=np.uint16).reshape(
        2, 3, 4, 5, 8, 2
    )
    output = raw.sum(axis=-1)
    reference = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)
    irf = reference + 100.0

    with h5py.File(path, "w") as handle:
        handle.attrs["schema_name"] = "brighteyes_mcs_file"
        handle.attrs["data_format_version"] = "0.0.6"
        handle.attrs["default"] = "/raw/spad"

        raw_group = handle.create_group("raw")
        raw_group.attrs["metadata_path"] = "/raw/metadata"
        raw_group.attrs["axes_path"] = "/raw/axes"
        data = raw_group.create_dataset("spad", data=raw)
        data.attrs["axis_order"] = (
            "repetition,z,y,x,time_bin,detector_channel"
        )
        data.attrs["source_key"] = "data"
        data.attrs["time_axis_path"] = "/raw/axes/digital_time_ns"
        data.attrs["metadata_path"] = "/raw/metadata"
        data.attrs["string_array"] = np.array([b"alpha", b"beta"])

        metadata = raw_group.create_group("metadata")
        metadata.attrs["laser_frequency_mhz"] = 80.0
        metadata.attrs["digital_time_bin_ns"] = 12.5 / 8
        timing = metadata.create_group("acquisition").create_group("timing")
        timing.attrs["laser_frequency_mhz"] = 80.0
        timing.attrs["digital_time_bin_ns"] = 12.5 / 8

        axes = raw_group.create_group("axes")
        axes.create_dataset(
            "digital_time_ns",
            data=np.linspace(0.0, 12.5, 8, endpoint=False),
        )

        calibration = handle.create_group("calibration")
        calibration.create_group("axes").create_dataset(
            "time_ns",
            data=np.linspace(0.0, 12.5, 8, endpoint=False),
        )
        result = calibration.create_group("results").create_group("spad")
        result.attrs["laser_frequency_mhz"] = 80.0
        result.attrs["time_axis_path"] = "/calibration/axes/time_ns"
        fit = result.create_group("fit")
        fit.create_dataset("tau_reference_ns", data=np.array([2.7, 2.7]))
        aligned = result.create_group("aligned")
        aligned.create_dataset("reference_trace", data=reference)
        aligned.create_dataset("irf_trace", data=irf)

        out = handle.create_group("output")
        out.attrs["default"] = "/output/sum_channels_001/products/spad"
        out.attrs["default_run"] = "/output/sum_channels_001"
        out.attrs["default_ref_trace_id"] = (
            "/output/sum_ref_001/products/trace"
        )
        out.attrs["default_irf_trace_id"] = (
            "/output/sum_irf_001/products/trace"
        )

        run = out.create_group("sum_channels_001")
        run.attrs["output_type"] = "image_tool"
        run.create_group("metadata").attrs["laser_frequency_mhz"] = 80.0
        run.create_group("axes").create_dataset(
            "time_ns", data=np.linspace(0.0, 12.5, 8, endpoint=False)
        )
        product = run.create_group("products").create_dataset(
            "spad", data=output
        )
        product.attrs["axis_order"] = "repetition,z,y,x,time_bin"
        product.attrs["time_axis_path"] = (
            "/output/sum_channels_001/axes/time_ns"
        )
        product.attrs["metadata_path"] = "/output/sum_channels_001/metadata"

        for name, kind, trace in (
            ("sum_ref_001", "sum_reference_trace", reference.sum(axis=1)),
            ("sum_irf_001", "sum_irf_trace", irf.sum(axis=1)),
        ):
            trace_run = out.create_group(name)
            trace_run.attrs["output_type"] = "trace_tool"
            trace_run.attrs["trace_kind"] = kind
            trace_run.create_group("axes").create_dataset(
                "time_ns", data=np.linspace(0.0, 12.5, 8, endpoint=False)
            )
            trace_product = trace_run.create_group(
                "products"
            ).create_dataset("trace", data=trace)
            trace_product.attrs["output_type"] = "trace"
            trace_product.attrs["trace_kind"] = kind
            trace_product.attrs["time_axis_path"] = (
                f"/output/{name}/axes/time_ns"
            )

    return raw, output, reference.sum(axis=1), irf.sum(axis=1)


def test_is_brightemcs_h5_and_list_datasets(tmp_path):
    filename = tmp_path / "sample.h5"
    _write_h5(filename)

    assert is_brightemcs_h5(filename)
    infos = list_datasets(filename)

    assert infos[0].path == "/output/sum_channels_001/products/spad"
    assert infos[0].kind == "data"
    assert infos[0].is_default
    assert any(info.kind == "reference" for info in infos)
    assert any(info.kind == "irf" for info in infos)


def test_read_default_output_signal(tmp_path):
    filename = tmp_path / "sample.h5"
    _, output, _, _ = _write_h5(filename)

    signal = read_signal(filename, time=1, depth=2)

    assert signal.dims == ("Y", "X", "H")
    assert signal.data.shape == (4, 5, 8)
    assert_array_equal(signal.data, output[1, 2])
    assert_array_equal(
        signal.coords["H"], np.linspace(0.0, 12.5, 8, endpoint=False)
    )
    assert signal.attrs["frequency"] == 80.0
    assert signal.attrs["reference_lifetime_ns"] == 2.7
    assert signal.attrs["h5_selection"] == {
        "time": 1,
        "depth": 2,
        "channel": None,
    }


def test_read_raw_channel_index_and_sum(tmp_path):
    filename = tmp_path / "sample.h5"
    raw, _, _, _ = _write_h5(filename)

    signal = read_signal(
        filename, dataset="/raw/spad", time=-1, depth=-1, channel=1
    )
    assert_array_equal(signal.data, raw[-1, -1, :, :, :, 1])
    assert signal.attrs["h5_selection"] == {
        "time": 1,
        "depth": 2,
        "channel": 1,
    }

    signal = read_signal(
        filename, dataset="/raw/spad", time=1, depth=2, channel="sum"
    )
    assert_array_equal(signal.data, raw[1, 2].sum(axis=-1))
    assert signal.attrs["h5_selection"]["channel"] == "sum"


def test_read_reference_and_irf_aliases(tmp_path):
    filename = tmp_path / "sample.h5"
    _, _, reference, irf = _write_h5(filename)

    signal = read_signal(filename, dataset="reference")
    assert signal.data.shape == (1, 1, 8)
    assert_array_equal(signal.data, reference.reshape(1, 1, 8))
    assert signal.attrs["reference"] is True
    assert signal.attrs["irf"] is False

    signal = read_signal(filename, dataset="irf")
    assert_array_equal(signal.data, irf.reshape(1, 1, 8))
    assert signal.attrs["reference"] is True
    assert signal.attrs["irf"] is True


def test_explicit_dataset_path_and_byte_string_attrs(tmp_path):
    filename = tmp_path / "sample.h5"
    raw, _, _, _ = _write_h5(filename)

    signal = read_signal(filename, dataset="raw/spad", time=0, depth=0)

    assert_array_equal(signal.data, raw[0, 0, :, :, :, 0])
    attrs = signal.attrs["h5_attrs"]["data"]
    assert attrs["string_array"] == ["alpha", "beta"]


def test_malformed_axes_raise(tmp_path):
    filename = tmp_path / "bad.h5"
    with h5py.File(filename, "w") as handle:
        handle.attrs["schema_name"] = "brighteyes_mcs_file"
        handle.attrs["data_format_version"] = "0.0.6"
        handle.create_group("raw").create_group("metadata")
        handle["raw"].create_group("axes")
        data = handle.create_dataset("raw/spad", data=np.zeros((2, 3, 4)))
        data.attrs["axis_order"] = "a,b,c"

    with pytest.raises(ValueError, match="y axis"):
        read_signal(filename)


def test_missing_default_raises(tmp_path):
    filename = tmp_path / "bad.h5"
    with h5py.File(filename, "w") as handle:
        handle.attrs["schema_name"] = "brighteyes_mcs_file"
        handle.attrs["data_format_version"] = "0.0.6"
        handle.create_group("raw").create_group("metadata")
        handle["raw"].create_group("axes")

    with pytest.raises(ValueError, match="default data"):
        read_signal(filename)
