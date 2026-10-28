import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _fail(path: str, message: str) -> None:
    raise AssertionError(f"{path}: {message}")


def _equal_or_both_nan(old: Any, new: Any) -> bool:
    return bool(old == new) or (
        isinstance(old, float) and math.isnan(old) and math.isnan(new)
    )


def assert_same(old: object, new: object, path: str = "result") -> None:
    """Raise `AssertionError` unless `new` equals `old`, exactly and recursively."""
    if isinstance(old, Figure | Axes) and isinstance(new, Figure | Axes):
        return
    if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
        assert_same(old.detach().cpu().numpy(), new.detach().cpu().numpy(), path)
    elif isinstance(old, np.ndarray | np.generic):
        if type(new) is not type(old):
            _fail(path, f"type {type(old).__name__} != {type(new).__name__}")
        try:
            np.testing.assert_array_equal(new, old, strict=True)
        except AssertionError as error:
            _fail(path, str(error))
    elif isinstance(old, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(new, old)  # type: ignore[arg-type]
        except (AssertionError, TypeError) as error:
            _fail(path, str(error))
    elif isinstance(old, pd.Series):
        try:
            pd.testing.assert_series_equal(new, old)  # type: ignore[arg-type]
        except (AssertionError, TypeError) as error:
            _fail(path, str(error))
    elif isinstance(old, Mapping):
        if not isinstance(new, Mapping) or list(old) != list(new):
            _fail(path, f"keys differ: {list(old)!r} != {new!r}")
        for key in old:
            assert_same(old[key], new[key], f"{path}[{key!r}]")  # type: ignore[index]
    elif isinstance(old, list | tuple):
        if type(old) is not type(new):
            _fail(path, f"type {type(old).__name__} != {type(new).__name__}")
        if len(old) != len(new):  # type: ignore[arg-type]
            _fail(path, f"length {len(old)} != {len(new)}")  # type: ignore[arg-type]
        for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):  # type: ignore[call-overload]
            assert_same(old_item, new_item, f"{path}[{index}]")
    elif old is None or isinstance(old, int | float | complex | str | bytes):
        if type(old) is not type(new):
            _fail(path, f"type {type(old).__name__} != {type(new).__name__}")
        if not (
            _equal_or_both_nan(old.real, new.real)
            and _equal_or_both_nan(old.imag, new.imag)  # type: ignore[union-attr]
            if isinstance(old, complex)
            else _equal_or_both_nan(old, new)
        ):
            _fail(path, f"{old!r} != {new!r}")
    elif isinstance(old, BaseException):
        if type(old).__name__ != type(new).__name__:
            _fail(path, f"class {type(old).__name__} != {type(new).__name__}")
        assert_same(old.args, getattr(new, "args", None), f"{path}.args")
    elif callable(old) and hasattr(old, "__qualname__"):
        if old.__qualname__ != getattr(new, "__qualname__", None):
            _fail(
                path, f"{old.__qualname__!r} != {getattr(new, '__qualname__', new)!r}"
            )
    else:
        _compare_objects(old, new, path)


def _compare_objects(old: Any, new: Any, path: str) -> None:
    try:
        old_vars = vars(old)
    except TypeError:
        raise TypeError(
            f"{path}: cannot compare an object of type {type(old).__name__}"
        ) from None
    if type(old).__name__ != type(new).__name__:
        _fail(path, f"class {type(old).__name__} != {type(new).__name__}")
    assert_same(old_vars, vars(new), f"{path}.__dict__")
