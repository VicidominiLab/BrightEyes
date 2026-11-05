import random
from collections import Counter

import numpy as np
import pandas as pd
import pytest
import torch
from _compare import assert_same
from _inputs import mcs_h5_file, ttm_new_protocol_bytes, ttm_raw_frame
from _registry import Case, new_path, resolve, run

from brighteyes.mcs.reader import load_raw, read_signal


class _Box:
    def __init__(self, value):
        self.value = value


def _same_named_boxes(old_value, new_value):
    """Build two instances of different classes that share one name."""

    def make(value):
        class Box:
            def __init__(self, value):
                self.value = value

        return Box(value)

    return make(old_value), make(new_value)


def _same_named_callables(old_value, new_value):
    """Build two callable instances of different classes that share one name."""

    def make(value):
        class Net:
            def __init__(self, value):
                self.value = value

            def __call__(self):
                return self.value

        return Net(value)

    return make(old_value), make(new_value)


def test_compare_rejects_different_values():
    """Raise AssertionError for values, dtypes, keys, types and attributes that differ."""
    cases = [
        (np.array([1.0, 2.0]), np.array([1.0, 2.5])),
        (np.array([1, 2]), np.array([1.0, 2.0])),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 2.0, 3.5])),
        (np.array([1, "a"], dtype=object), np.array([1, "b"], dtype=object)),
        ({"a": 1}, {"b": 1}),
        ((1, 2), [1, 2]),
        (True, 1),
        (pd.Series([1.0]), pd.Series([1.0 + 1e-9])),
        (pd.DataFrame({"x": [1.0]}), pd.DataFrame({"x": [1.0 + 1e-9]})),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 3], mask=[0, 1])),
        (np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 0])),
        (np.ma.array([1, 2], mask=[0, 1]), np.array([1, 2])),
        (np.matrix([[1, 2]]), np.array([[1, 2]])),
        _same_named_boxes(1, 2),
        _same_named_callables(1, 2),
    ]
    for old, new in cases:
        with pytest.raises(AssertionError):
            assert_same(old, new)


def test_compare_accepts_equal_values():
    """Accept equal arrays with NaN, data frames, tensors, objects and self references."""
    nested = {"a": np.array([1.0, np.nan]), "b": [np.float64(2.0), (1, "x", None)]}
    assert_same(
        nested, {"a": np.array([1.0, np.nan]), "b": [np.float64(2.0), (1, "x", None)]}
    )
    frame = pd.DataFrame({"x": [1, 2], "y": [0.5, np.nan]})
    assert_same(frame, frame.copy())
    assert_same(torch.arange(3.0), torch.arange(3.0))
    assert_same(np.ma.array([1, 2], mask=[0, 1]), np.ma.array([1, 2], mask=[0, 1]))
    assert_same(*_same_named_boxes(1, 1))
    assert_same(*_same_named_callables(1, 1))

    old, new = _Box(None), _Box(None)
    old.value, new.value = old, new
    assert_same(old, new)


def test_compare_names_the_path_of_a_difference():
    """Put the path of the differing element in the failure message."""
    with pytest.raises(AssertionError, match=r"result\[\"g\"\]\[1\]"):
        assert_same({"g": [1, 2]}, {"g": [1, 3]})


def test_exact_relocations_preserve_class_members(monkeypatch):
    """Resolve a class member after its containing class moves."""
    monkeypatch.setattr(
        "_registry.paths", {"sample.old:Counter": "collections:Counter"}
    )
    method = resolve(new_path("sample.old:Counter.most_common"))
    assert method(Counter("aab"), 1) == [("a", 2)]


def test_exact_relocations_distinguish_same_named_functions(monkeypatch):
    """Resolve identical old names to their distinct destination functions."""
    monkeypatch.setattr(
        "_registry.paths",
        {"sample.first:reduce": "builtins:sum", "sample.second:reduce": "builtins:max"},
    )
    assert resolve(new_path("sample.first:reduce"))([1, 2, 3]) == 6
    assert resolve(new_path("sample.second:reduce"))([1, 2, 3]) == 3


def test_declared_removal_has_no_new_path(monkeypatch):
    """Refuse to resolve a definition intentionally removed from the package."""
    monkeypatch.setattr(
        "_registry.removed",
        {"brighteyes_ism.analysis.Tools_lib:low_pass": "removed from imaging"},
    )
    with pytest.raises(ValueError, match="removed name"):
        new_path("brighteyes_ism.analysis.Tools_lib:low_pass")


def test_compare_rejects_unknown_type():
    """Raise TypeError naming the type of an object without vars()."""
    with pytest.raises(TypeError, match="memoryview"):
        assert_same(memoryview(b"ab"), memoryview(b"ab"))


def test_run_resolves_new_path():
    """Map each old package prefix to its new location."""
    assert new_path("brighteyes_ism.dataio.mcs:x") == "brighteyes.imaging.io.mcs:x"
    assert (
        new_path("brighteyes_ism.analysis.APR_lib:hann2d")
        == "brighteyes.imaging.analysis.APR_lib:hann2d"
    )
    assert new_path("brighteyes_flism.tools:x") == "brighteyes.imaging.flism.tools:x"
    assert (
        new_path("brighteyes_ffs.fcs.fcs_funcs:A.b")
        == "brighteyes.spectroscopy.fcs.fcs_funcs:A.b"
    )


def test_new_path_maps_the_new_packages():
    """Map each moved package prefix to its location and refuse a lookalike name."""
    assert (
        new_path("brighteyes_mcs_reader.reader:read_signal")
        == "brighteyes.mcs.reader.reader:read_signal"
    )
    assert (
        new_path("brighteyes_mcs_dataprep.h5_output_writers:x")
        == "brighteyes.mcs.dataprep.h5_output_writers:x"
    )
    assert (
        new_path("libttp.ttpCython:timeProcessRAW")
        == "brighteyes.ttm._ttp:timeProcessRAW"
    )
    assert new_path("libttp.ttp:fitGaus") == "brighteyes.ttm.ttp:fitGaus"
    assert new_path("s2ism.s2ism:x") == "brighteyes.imaging.s2ism.s2ism:x"
    assert new_path("flus2ISM_lib:x") == "brighteyes.imaging.flus2ism.flus2ISM_lib:x"
    assert new_path("varreg.x:y") == "brighteyes.imaging.varreg.x:y"
    assert new_path("psf_ism_distiller.x:y") == "brighteyes.imaging.psf_distiller.x:y"
    assert new_path("birfi.x:y") == "brighteyes.imaging.birfi.x:y"
    with pytest.raises(ValueError):
        new_path("s2ism_extra:x")


def test_run_seeds_each_call():
    """Seed the random generators identically before the old and the new call."""
    run(
        Case(
            "brighteyes_ism.analysis.Tools_lib:sigmoid",
            call=lambda f: (
                f.__qualname__,
                np.random.rand(3),
                random.random(),
                torch.rand(3),
            ),
        )
    )


def test_run_detects_a_change():
    """Raise when the old and the new object give different results."""
    with pytest.raises(AssertionError):
        run(
            Case(
                "brighteyes_ism.analysis.Tools_lib:sigmoid", call=lambda f: f.__module__
            )
        )


def test_mcs_h5_file_is_readable(tmp_path):
    """Open the generated file with the reader and read a signal and the raw payload."""
    path = mcs_h5_file(tmp_path / "sample.h5")
    assert read_signal(path).data.shape == (4, 5, 8)
    assert load_raw(path, "aux").shape == (1, 1, 4, 5, 8, 2)


def test_ttm_new_protocol_bytes_has_every_id_class():
    """Hold channel words, the five flag words and `0x7FFF` fillers in the stream."""
    words = np.frombuffer(ttm_new_protocol_bytes(), dtype=np.uint16)
    ids = set((words >> 8) & 0x7F)
    assert {0, 3, 123, 124, 125, 126, 127} <= ids
    assert 0x7FFF in words
    assert np.count_nonzero(words != 0x7FFF) > 1000


def test_ttm_raw_frame_has_the_cython_dtypes():
    """Give the frame the exact column and index dtypes the Cython functions index."""
    frame = ttm_raw_frame()
    uint8 = ["t_L", "t_0", "t_3", "valid_tdc_0", "valid_tdc_3", "valid_tdc_L"]
    uint8 += ["scan_enable", "line_enable", "pixel_enable"]
    assert all(frame[name].dtype == np.uint8 for name in uint8)
    assert frame["step"].dtype == np.uint16
    assert frame["cumulativeStep"].dtype == np.int64
    assert frame.index.dtype == np.int64
    assert len(frame) >= 1000
