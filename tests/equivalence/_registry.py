import importlib
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any, Final

import numpy as np
import torch

from ._compare import assert_same

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
    old: str
    call: Callable[[Any], Any]
    label: str = ""
    compare: Callable[[object, object], None] = assert_same


def new_path(old: str) -> str:
    """Return the `module:qualname` path in the moved package for an old path."""
    for old_prefix, new_prefix in PREFIXES:
        if old.startswith((old_prefix + ".", old_prefix + ":")):
            return new_prefix + old[len(old_prefix) :]
    raise ValueError(f"unknown package in {old!r}")


def resolve(path: str) -> Any:
    """Import the object named by a `module:qualname` path."""
    module_name, qualname = path.split(":")
    return reduce(getattr, qualname.split("."), importlib.import_module(module_name))


def _seed() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def run(case: Case) -> None:
    """Call the old and the moved object with equal seeds and compare the results."""
    _seed()
    old = case.call(resolve(case.old))
    _seed()
    new = case.call(resolve(new_path(case.old)))
    case.compare(old, new)
