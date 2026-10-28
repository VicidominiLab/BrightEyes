import random

import numpy as np
import pandas as pd
import pytest
import torch

from ._compare import assert_same
from ._registry import Case, new_path, run


def _box(value: int) -> object:
    class Box:
        def __init__(self, value: int) -> None:
            self.value = value

    return Box(value)


def test_compare_rejects_different_values() -> None:
    """Raise for different values, dtypes, keys, sequence types and attributes."""
    pairs = [
        (np.array([1.0, 2.0]), np.array([1.0, 2.5])),
        (np.array([1, 2]), np.array([1.0, 2.0])),
        ({"a": 1}, {"b": 1}),
        ((1, 2), [1, 2]),
        (_box(1), _box(2)),
    ]
    for old, new in pairs:
        with pytest.raises(AssertionError):
            assert_same(old, new)


def test_compare_accepts_equal_values() -> None:
    """Accept nested arrays with NaN, data frames, tensors and objects with equal attributes."""
    frame = pd.DataFrame({"a": [1, 2]})
    assert_same({"g": [np.array([1.0, np.nan]), (1, "a", None)]}, {"g": [np.array([1.0, np.nan]), (1, "a", None)]})
    assert_same(frame, frame.copy())
    assert_same(torch.arange(3.0), torch.arange(3.0))
    assert_same(_box(1), _box(1))


def test_compare_rejects_unknown_type() -> None:
    """Raise `TypeError` naming a type that cannot be compared."""
    with pytest.raises(TypeError, match="memoryview"):
        assert_same(memoryview(b"ab"), memoryview(b"ab"))


def test_compare_names_the_path() -> None:
    """Name the position of a difference inside a nested result."""
    with pytest.raises(AssertionError, match=r"result\['g'\]\[1\]"):
        assert_same({"g": [1, 2]}, {"g": [1, 3]})


def test_run_resolves_new_path() -> None:
    """Map each old package prefix to its new location."""
    assert new_path("brighteyes_ism.dataio.mcs:x") == "brighteyes.imaging.io.mcs:x"
    assert new_path("brighteyes_ism.analysis.APR_lib:f") == "brighteyes.imaging.analysis.APR_lib:f"
    assert new_path("brighteyes_flism.tools_phasor:f") == "brighteyes.imaging.flism.tools_phasor:f"
    assert new_path("brighteyes_ffs.fcs.atimes2corr:f") == "brighteyes.spectroscopy.fcs.atimes2corr:f"


def test_run_seeds_each_call() -> None:
    """Pass when both calls see identically seeded random generators."""
    run(Case("brighteyes_ism.analysis.Tools_lib:sigmoid", call=lambda f: (f.__qualname__, np.random.rand(3), random.random())))


def test_run_detects_a_change() -> None:
    """Raise when the old and the moved object give different results."""
    with pytest.raises(AssertionError):
        run(Case("brighteyes_ism.analysis.Tools_lib:sigmoid", call=lambda f: f.__module__))
