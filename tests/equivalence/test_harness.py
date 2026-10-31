import random

import numpy as np
import pandas as pd
import pytest
import torch
from _compare import assert_same
from _registry import Case, new_path, run


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


def test_run_seeds_each_call():
    """Seed the random generators identically before the old and the new call."""
    run(
        Case(
            "brighteyes_ism.analysis.Tools_lib:sigmoid",
            call=lambda f: (f.__qualname__, np.random.rand(3), random.random()),
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
