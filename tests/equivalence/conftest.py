import sys

import libttp.ttpCython
import matplotlib
import numpy as np
from _registry import REFERENCE_DIR

matplotlib.use("Agg")

if REFERENCE_DIR.exists():
    sys.path.insert(0, str(REFERENCE_DIR))


class _WritableNumpy:
    """Numpy whose `asarray` returns a writable copy."""

    def asarray(self, x, *args, **kwargs):
        return np.array(x, copy=True)

    def __getattr__(self, name):
        return getattr(np, name)


# The old module was built for Python 2 names and writable arrays: Python 3 has
# no `long`, and pandas 3 copy-on-write columns are read-only, which its typed
# memoryviews refuse.
libttp.ttpCython.long = int
libttp.ttpCython.np = _WritableNumpy()
