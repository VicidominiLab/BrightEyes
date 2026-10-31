import inspect
from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _same_name(old: object, new: object) -> bool:
    return type(old).__name__ == type(new).__name__


def _key(key: object) -> str:
    return f'["{key}"]' if isinstance(key, str) else f"[{key!r}]"


def _fail(path: str, message: str) -> AssertionError:
    return AssertionError(f"{path}: {message}")


def _kinds(old: object, new: object) -> str:
    return f"{type(old).__name__} vs {type(new).__name__}"


def _check_array(old: np.ndarray, new: object, path: str) -> None:
    if isinstance(old, np.ndarray) and old.dtype == object and isinstance(new, np.ndarray):
        if old.shape != new.shape or new.dtype != object:
            raise _fail(path, f"object arrays differ: {old.shape} {old.dtype} vs {new.shape} {new.dtype}")
        for index in np.ndindex(old.shape):
            _compare(old[index], new[index], f"{path}[{', '.join(map(str, index))}]", set())
        return
    try:
        np.testing.assert_array_equal(new, old, strict=True)
    except AssertionError as error:
        raise _fail(path, str(error)) from None


def _compare(old: object, new: object, path: str, seen: set[tuple[int, int]]) -> None:
    match old:
        case Figure() | Axes():
            if not isinstance(new, Figure | Axes):
                raise _fail(path, _kinds(old, new))
        case torch.Tensor():
            if not isinstance(new, torch.Tensor):
                raise _fail(path, _kinds(old, new))
            _check_array(old.detach().cpu().numpy(), new.detach().cpu().numpy(), path)
        case np.ndarray() | np.generic():
            if type(old) is not type(new):
                raise _fail(path, _kinds(old, new))
            if isinstance(old, np.ma.MaskedArray):
                _check_array(np.ma.getmaskarray(old), np.ma.getmaskarray(new), path + ".mask")
                _check_array(np.ma.getdata(old), np.ma.getdata(new), path + ".data")
            else:
                _check_array(old, new, path)
        case pd.DataFrame() | pd.Series():
            if type(old) is not type(new):
                raise _fail(path, _kinds(old, new))
            check = pd.testing.assert_frame_equal if isinstance(old, pd.DataFrame) else pd.testing.assert_series_equal
            try:
                check(old, new, check_exact=True)
            except AssertionError as error:
                raise _fail(path, str(error)) from None
        case Mapping():
            if not isinstance(new, Mapping) or set(old) != set(new):
                raise _fail(path, f"keys differ, {_kinds(old, new)}")
            if (id(old), id(new)) in seen:
                return
            seen.add((id(old), id(new)))
            for key in old:
                _compare(old[key], new[key], path + _key(key), seen)
        case list() | tuple():
            if not _same_name(old, new) or len(old) != len(new):
                raise _fail(path, f"sequences differ, {_kinds(old, new)}")
            if (id(old), id(new)) in seen:
                return
            seen.add((id(old), id(new)))
            for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
                _compare(old_item, new_item, f"{path}[{index}]", seen)
        case float() | complex():
            if not _same_name(old, new) or not (old == new or (old != old and new != new)):
                raise _fail(path, f"{old!r} vs {new!r}")
        case bool() | int() | str() | bytes() | None:
            if not _same_name(old, new) or old != new:
                raise _fail(path, f"{old!r} vs {new!r}")
        case _ if inspect.isroutine(old) or isinstance(old, type):
            if getattr(old, "__qualname__", None) != getattr(new, "__qualname__", None):
                raise _fail(path, f"{old!r} vs {new!r}")
        case _:
            try:
                old_attributes = vars(old)
            except TypeError:
                raise TypeError(f"{path}: cannot compare an object of type {type(old).__name__}") from None
            if not _same_name(old, new):
                raise _fail(path, _kinds(old, new))
            if (id(old), id(new)) in seen:
                return
            seen.add((id(old), id(new)))
            _compare(old_attributes, vars(new), path + ".__dict__", seen)


def assert_same(old: object, new: object) -> None:
    """Raise `AssertionError` unless `old` and `new` hold equal results.

    Arrays, tensors, data frames, mappings, sequences, scalars and the
    attributes of same-named objects are compared exactly. Figures count as
    equal. A failure message names the path of the first difference.

    Raises
    ------
    TypeError
        If the objects are of a type that cannot be compared.
    """
    _compare(old, new, "result", set())
