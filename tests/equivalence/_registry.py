import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from typing import Any, Final

import numpy as np
import torch
from _compare import assert_same

SKIP_FILE_IO: Final[str] = "reads or writes files"
SKIP_PLOT_ONLY: Final[str] = "only draws a figure"
SKIP_GUI: Final[str] = "needs a running Qt application"

PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("brighteyes_ism.dataio", "brighteyes.imaging.io"),
    ("brighteyes_ism", "brighteyes.imaging"),
    ("brighteyes_flism", "brighteyes.imaging.flism"),
    ("brighteyes_ffs", "brighteyes.spectroscopy"),
)


@dataclass(frozen=True)
class Case:
    """One comparison between a name of the old packages and its moved copy."""

    old: str
    call: Callable[[Any], Any]
    label: str = ""
    compare: Callable[[object, object], None] = assert_same


def new_path(old: str) -> str:
    """Return the location in `brighteyes` of the old `module:qualname` path."""
    module, separator, qualname = old.partition(":")
    for old_prefix, new_prefix in PREFIXES:
        if module == old_prefix or module.startswith(old_prefix + "."):
            return new_prefix + module[len(old_prefix) :] + separator + qualname
    raise ValueError(f"no known package prefix in {old!r}")


def resolve(path: str) -> Any:
    """Import the object a `module:qualname` path names."""
    module, _, qualname = path.partition(":")
    return reduce(getattr, qualname.split("."), import_module(module))


def run(case: Case) -> None:
    """Call the old and the moved object with equal seeds and compare the results."""
    results = []
    for path in (case.old, new_path(case.old)):
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        results.append(case.call(resolve(path)))
    case.compare(*results)
