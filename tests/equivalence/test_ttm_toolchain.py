import contextlib
import io

import libttp.ttpCython
import numpy as np
import pytest
from _compare import assert_same
from _inputs import ttm_raw_frame

from brighteyes.ttm import _ttp


def test_read_only_frame_matches_patched_old_module():
    """Index a pandas 3 frame of read-only columns like the old module with writable copies."""
    frame = ttm_raw_frame()
    with contextlib.redirect_stdout(io.StringIO()):
        new = _ttp.analysisForImg(frame, 10, frame["cumulativeStep"])
        old = libttp.ttpCython.analysisForImg(frame, 10, frame["cumulativeStep"])
    assert_same(old, new)


def test_line_enable_on_first_row_starts_a_line():
    """Count a line at a first row whose line_enable is 1."""
    frame = ttm_raw_frame()
    frame.loc[0, "line_enable"] = np.uint8(1)
    with contextlib.redirect_stdout(io.StringIO()):
        _, arr_py, *_ = _ttp.analysisForImg(frame, 10, frame["cumulativeStep"])
    # The old module left last_le uninitialised, so its result here was
    # undefined; the MSVC build in the reference group gave 0.
    assert arr_py[0] == 1


# The old module divides by a zero progress-bar period in C on 1 to 99 rows,
# which ends the process with 0xC0000094 instead of raising.
def test_raw_pairs_on_fewer_than_100_rows_divide_by_zero():
    """Raise ZeroDivisionError for timeProcessRAW on 50 rows."""
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ZeroDivisionError):
        _ttp.timeProcessRAW(ttm_raw_frame().iloc[:50], 0)


def test_image_indices_on_fewer_than_100_rows_divide_by_zero():
    """Raise ZeroDivisionError for analysisForImg on 50 rows."""
    frame = ttm_raw_frame().iloc[:50]
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ZeroDivisionError):
        _ttp.analysisForImg(frame, 10, frame["cumulativeStep"])
