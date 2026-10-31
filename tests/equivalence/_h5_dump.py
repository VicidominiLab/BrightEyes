from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _clean(value: Any, folder: str) -> Any:
    match value:
        case str():
            posix = Path(folder).as_posix()
            for variant in (folder, folder.replace("\\", "\\\\"), posix):
                value = value.replace(variant, "<dir>")
            return value
        case np.ndarray() if value.dtype == object:
            return np.array([_clean(item, folder) for item in value.tolist()], object)
        case _:
            return value


def h5_contents(
    path: str | Path, folder: str | Path, drop: frozenset[str] = frozenset()
) -> dict[str, Any]:
    """Return every group, dataset and attribute of an HDF5 file keyed by path.

    Strings holding `folder` have it replaced by `<dir>`, and attributes named
    in `drop` are left out.
    """
    folder = str(folder)
    contents: dict[str, Any] = {}

    def visit(name: str, node: h5py.Group | h5py.Dataset) -> None:
        attrs = {
            key: _clean(value, folder)
            for key, value in node.attrs.items()
            if key not in drop
        }
        data = _clean(node[()], folder) if isinstance(node, h5py.Dataset) else None
        contents["/" + name] = (type(node).__name__, attrs, data)

    with h5py.File(path, "r") as handle:
        visit("", handle)
        handle.visititems(visit)
    return contents
