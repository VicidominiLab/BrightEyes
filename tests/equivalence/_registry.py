import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from importlib import import_module
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from _compare import assert_same
from _relocations import paths, removed

REFERENCE_DIR: Final[Path] = Path(__file__).parent / "reference"

SKIP_FILE_IO: Final[str] = "reads or writes files"
SKIP_PLOT_ONLY: Final[str] = "only draws a figure"
SKIP_GUI: Final[str] = "needs a running Qt application"

PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("brighteyes_mcs_reader", "brighteyes.mcs.reader"),
    ("brighteyes_mcs_dataprep", "brighteyes.mcs.dataprep"),
    ("libttp.ttpCython", "brighteyes.ttm._ttp"),
    ("libttp", "brighteyes.ttm"),
    ("s2ism", "brighteyes.imaging.s2ism"),
    ("flus2ISM_lib", "brighteyes.imaging.flus2ism.flus2ISM_lib"),
    ("varreg", "brighteyes.imaging.varreg"),
    ("psf_ism_distiller", "brighteyes.imaging.psf_distiller"),
    ("birfi", "brighteyes.imaging.birfi"),
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
    """Return the location in `brighteyes` of the old `module:qualname` path.

    Raises
    ------
    ValueError
        If the original name was removed or has no known package prefix.
    """
    module, separator, qualname = old.partition(":")
    name, member_separator, member = qualname.partition(".")
    original = f"{module}:{name}"
    if original in removed:
        raise ValueError(f"removed name: {original}: {removed[original]}")
    if original in paths:
        return paths[original] + member_separator + member
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
        target = resolve(path)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        results.append(case.call(target))
    case.compare(*results)
