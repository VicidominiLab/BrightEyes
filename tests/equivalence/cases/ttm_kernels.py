"""Equivalence cases for `libttp.ttpCython`."""

import contextlib
import io
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from _inputs import ttm_new_protocol_bytes, ttm_raw_frame
from _registry import Case

TTP = "libttp.ttpCython"


def _printed(call: Callable[[], Any]) -> tuple[Any, str]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        result = call()
    return result, stdout.getvalue()


def _error(call: Callable[[], Any]) -> str:
    try:
        _printed(call)
    except (ArithmeticError, LookupError, TypeError, ValueError) as error:
        return type(error).__name__
    raise AssertionError("no exception raised")


def _on_stream(function: Callable[..., Any], **kwargs: Any) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "stream.bin"
        path.write_bytes(ttm_new_protocol_bytes())
        return _printed(lambda: function(str(path), **kwargs))


def _negative_step() -> pd.Series:
    steps = ttm_raw_frame()["cumulativeStep"].copy()
    steps.iloc[500] = steps.iloc[0] - 100
    return steps


def _jump_at_500(offset: int) -> pd.Series:
    steps = ttm_raw_frame()["cumulativeStep"].copy()
    steps.iloc[500] = steps.iloc[499] + offset
    return steps


def _longer_steps() -> pd.Series:
    steps = ttm_raw_frame()["cumulativeStep"]
    return pd.concat([steps, pd.Series(np.arange(100) + 10**6)], ignore_index=True)


def _on_flat(
    f: Callable[..., Any], steps: list[int], step_per_pixel: int, pixel_enable: int = 0
) -> Any:
    """Call `f` on rows that start a frame on row 0, never start a line, and step by `steps`."""
    rows = len(steps)
    scan_enable = np.zeros(rows, dtype=np.uint8)
    scan_enable[0] = 1
    frame = pd.DataFrame(
        {
            "scan_enable": scan_enable,
            "line_enable": np.zeros(rows, dtype=np.uint8),
            "pixel_enable": np.full(rows, pixel_enable, dtype=np.uint8),
            "cumulativeStep": np.array(steps, dtype=np.int64),
        }
    )
    return f(frame, step_per_pixel, frame["cumulativeStep"])


INT64 = np.iinfo(np.int64)


def _with_column(name: str, dtype: type) -> pd.DataFrame:
    frame = ttm_raw_frame()
    frame[name] = frame[name].astype(dtype)
    return frame


def _image(step_per_pixel: object) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        frame = ttm_raw_frame()
        return _printed(lambda: f(frame, step_per_pixel, frame["cumulativeStep"]))

    return call


def _image_error(step_per_pixel: object) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        frame = ttm_raw_frame()
        return _error(lambda: f(frame, step_per_pixel, frame["cumulativeStep"]))

    return call


CASES = [
    Case(f"{TTP}:getLengthFileNewProtocol", lambda f: _on_stream(f)),
    Case(
        f"{TTP}:getLengthFileNewProtocol",
        lambda f: _on_stream(f, dtype=np.uint8),
        "uint8",
    ),
    Case(f"{TTP}:timeProcessNewProtocol", lambda f: _on_stream(f), "49 channels"),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _on_stream(f, CHANNELS=25),
        "25 channels",
    ),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _on_stream(f, file_offset=10, file_last=700),
        "file range",
    ),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _on_stream(f, CHANNELS=49.5),
        "fractional channels",
    ),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _on_stream(f, CHANNELS=-2),
        "negative columns wrap around",
    ),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _error(lambda: _on_stream(f, CHANNELS=-5)),
        "columns out of bounds",
    ),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _error(lambda: _on_stream(f, CHANNELS=65531)),
        "MAXWORD above uint16",
    ),
    Case(
        f"{TTP}:timeProcessNewProtocol",
        lambda f: _error(lambda: _on_stream(f, CHANNELS=-6)),
        "MAXWORD below zero",
    ),
    Case(
        f"{TTP}:timeProcessRAW",
        lambda f: _printed(lambda: f(ttm_raw_frame(), 0)),
        "channel 0",
    ),
    Case(
        f"{TTP}:timeProcessRAW",
        lambda f: _printed(lambda: f(ttm_raw_frame(), 3, verbose=True)),
        "channel 3, verbose",
    ),
    Case(
        f"{TTP}:timeProcessRAW",
        lambda f: _printed(lambda: f(ttm_raw_frame().iloc[:0], 0)),
        "no rows",
    ),
    Case(
        f"{TTP}:timeProcessRAW",
        lambda f: _error(lambda: f(_with_column("t_L", np.int16), 0)),
        "t_L as int16",
    ),
    Case(f"{TTP}:analysisForImg", _image(10), "frame"),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _error(
            lambda: f(ttm_raw_frame(), 10, _negative_step()),
        ),
        "negative pixelCorr",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _printed(lambda: f(ttm_raw_frame(), 10, _longer_steps())),
        "cumulativeStep longer than the frame",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _error(
            lambda: f(
                ttm_raw_frame(), 10, ttm_raw_frame()["cumulativeStep"].iloc[:2000]
            )
        ),
        "cumulativeStep shorter than the frame",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _error(lambda: f(ttm_raw_frame(), 1, _jump_at_500(2**40))),
        "pixelCorr outside int32",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _error(lambda: f(ttm_raw_frame(), 10, _jump_at_500(70000 * 10))),
        "pixelCorr above uint16",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _error(lambda: _on_flat(f, [0] * 70000, 1, pixel_enable=1)),
        "pixel above uint16",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _printed(
            lambda: _on_flat(f, [INT64.max] + [INT64.min + n for n in range(99)], 1)
        ),
        "step difference wraps around int64",
    ),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _printed(lambda: _on_flat(f, [100] + [95] * 99, 10)),
        "small negative distance truncates toward zero",
    ),
    Case(f"{TTP}:analysisForImg", _image_error(0), "StepPerPixel_in 0"),
    Case(f"{TTP}:analysisForImg", _image(10.5), "StepPerPixel_in float"),
    Case(f"{TTP}:analysisForImg", _image_error(2**63), "StepPerPixel_in above int64"),
    Case(f"{TTP}:analysisForImg", _image_error("10"), "StepPerPixel_in str"),
    Case(
        f"{TTP}:analysisForImg",
        lambda f: _error(
            lambda: f(
                _with_column("pixel_enable", np.int16),
                10,
                ttm_raw_frame()["cumulativeStep"],
            ),
        ),
        "pixel_enable as int16",
    ),
]

SKIPPED: dict[str, str] = {}
