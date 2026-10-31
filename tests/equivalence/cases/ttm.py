"""Equivalence cases for `libttp.ttp`."""

import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from _capture import error, printed
from _h5_dump import h5_contents
from _inputs import (
    ttm_calibration_h5_file,
    ttm_new_protocol_bytes,
    ttm_parsed_bytes,
    ttm_raw_bytes,
    ttm_raw_frame,
)
from _registry import SKIP_PLOT_ONLY, Case

TTP = "libttp.ttp"


def _scrub(value: Any, folder: str) -> Any:
    match value:
        case str():
            for variant in (
                folder,
                folder.replace("\\", "\\\\"),
                Path(folder).as_posix(),
            ):
                value = value.replace(variant, "<dir>")
            return value
        case h5py.Empty():
            return ("Empty", str(value.dtype))
        case dict():
            return {key: _scrub(item, folder) for key, item in value.items()}
        case tuple():
            return tuple(_scrub(item, folder) for item in value)
        case list():
            return [_scrub(item, folder) for item in value]
        case _:
            return value


def _in_folder(body: Callable[[Path], Any]) -> Any:
    with tempfile.TemporaryDirectory() as folder:
        return _scrub(body(Path(folder)), folder)


def _dumps(folder: Path) -> dict[str, Any]:
    return {
        str(path.relative_to(folder)): h5_contents(path, folder)
        for path in sorted(folder.rglob("*.h5"))
    }


def _on_file(
    name: str, content: Callable[[], bytes], **kwargs: Any
) -> Callable[[Callable[..., Any]], Any]:
    def call(f: Callable[..., Any]) -> Any:
        def body(folder: Path) -> Any:
            path = folder / name
            path.write_bytes(content())
            return printed(lambda: f(str(path), **kwargs))

        return _in_folder(body)

    return call


def _stream(**kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    return _on_file("stream.bin", ttm_new_protocol_bytes, **kwargs)


def _raw(**kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    return _on_file("data.raw", ttm_raw_bytes, **kwargs)


def _size(f: Callable[..., Any]) -> int:
    def body(folder: Path) -> int:
        (folder / "data").write_bytes(ttm_parsed_bytes())
        return f(str(folder / "data"))

    return _in_folder(body)


def _convert(
    first: Callable[[], Any], **kwargs: Any
) -> Callable[[Callable[..., Any]], Any]:
    channels = kwargs.pop("list_of_channels", [])

    def call(f: Callable[..., Any]) -> Any:
        def body(folder: Path) -> Any:
            output = folder / "out.h5"
            result = printed(
                lambda: f(
                    first(),
                    filenameOutputHDF5=str(output),
                    list_of_channels=channels,
                    **kwargs,
                )
            )
            return result, _dumps(folder)

        return _in_folder(body)

    return call


def _convert_file(raw: bool, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    channels = kwargs.pop("list_of_channels", [])

    def call(f: Callable[..., Any]) -> Any:
        def body(folder: Path) -> Any:
            path = folder / "data-1"
            path.write_bytes((ttm_raw_bytes if raw else ttm_parsed_bytes)())
            result = printed(
                lambda: f(
                    str(path),
                    fileInputRAW=raw,
                    list_of_channels=channels,
                    **kwargs,
                )
            )
            return result, _dumps(folder)

        return _in_folder(body)

    return call


def _failing(
    case: Callable[[Callable[..., Any]], Any],
) -> Callable[[Callable[..., Any]], str]:
    return lambda f: error(lambda: case(f))


def _described(parsed: tuple[Any, Any, np.dtype]) -> tuple[Any, Any, Any]:
    data, source, dtype = parsed
    return data, source, dtype.descr


def _with_scipy(f: Callable[..., Any], *args: Any) -> Any:
    namespace = f.__globals__
    namespace["scipy"] = scipy
    try:
        return f(*args)
    finally:
        del namespace["scipy"]


def _timing() -> dict[str, np.ndarray]:
    frame = ttm_raw_frame().iloc[:1000]
    steps = frame["step"].to_numpy().astype(np.int64)
    return {
        "t_L": frame["t_L"].to_numpy().astype(np.int64),
        "t_Ch": frame["t_3"].to_numpy().astype(np.int64),
        "S_L": steps,
        "S_Ch": steps[::-1].copy() % 7,
    }


def _histogram_peak() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = np.arange(100, dtype=np.float64)
    y = 400.0 * np.exp(-((x - 42.0) ** 2) / (2 * 6.0**2))
    return x, y + rng.poisson(5.0, size=x.size)


def _on_calibration_file(
    f: Callable[[str], Any],
) -> Any:
    def body(folder: Path) -> Any:
        path = ttm_calibration_h5_file(folder / "calib.h5")
        return printed(lambda: f(str(path)))

    return _in_folder(body)


def _calibrate(**kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    def call(f: Callable[..., Any]) -> Any:
        try:
            return _on_calibration_file(lambda path: f(path, [1, 0, 3, 5], **kwargs))
        finally:
            plt.close("all")

    return call


def _calib_dict() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "ch_L": np.sort(rng.random(150)),
        "ch_0": np.sort(rng.random(150)) * 0.9,
        "ch_3": np.sort(rng.random(150)) * 0.8,
    }


def _apply(**kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    return lambda f: _on_calibration_file(lambda path: f(path, _calib_dict(), **kwargs))


def _tables(channels: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows = 300
    return [
        pd.DataFrame(
            {
                "arr_frame": rng.integers(0, 2, size=rows),
                "arr_px": rng.integers(0, 4, size=rows),
                "arr_py": rng.integers(0, 4, size=rows),
                f"dt_{channel}": rng.random(rows) * 25000.0,
            }
        )
        for channel in range(channels)
    ]


def _receiver(**kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    def call(f: Callable[..., Any]) -> Any:
        def body(folder: Path) -> Any:
            script = folder / "receiver.py"
            script.write_text(
                "import sys\nsys.stderr.write('log line\\nFilename: data-123\\n')\n"
            )
            return printed(
                lambda: f(str(script), programm=sys.executable, cwd=folder, **kwargs)
            )

        return _in_folder(body)

    return call


def _counts() -> np.ndarray:
    return ttm_raw_frame()["t_0"].to_numpy().astype(np.float64) + 3.0


CASES = [
    Case(f"{TTP}:get_data_for_useconds", _receiver(), "stand-in receiver"),
    Case(
        f"{TTP}:get_data_for_useconds",
        _receiver(verbose=True),
        "stand-in receiver, verbose",
    ),
    Case(f"{TTP}:dtypeAll", lambda f: (f().descr, f().itemsize)),
    Case(
        f"{TTP}:readParsedFile",
        lambda f: _on_file("data", ttm_parsed_bytes)(lambda path: _described(f(path))),
    ),
    Case(f"{TTP}:readParsedFilePANDAS", _on_file("data", ttm_parsed_bytes)),
    Case(f"{TTP}:getSizeData", _size),
    Case(f"{TTP}:getLengthFileNewProtocol", _stream()),
    Case(f"{TTP}:getLengthFileNewProtocol", _stream(dtype=np.uint8), "uint8"),
    Case(f"{TTP}:readNewProtocolFileToPandas", _stream(), "49 channels"),
    Case(
        f"{TTP}:readNewProtocolFileToPandas",
        _stream(CHANNELS=25),
        "25 channels",
    ),
    Case(
        f"{TTP}:readNewProtocolFileToPandas",
        _stream(CHANNELS=25, reorder_channels=True),
        "25 channels reordered",
    ),
    Case(
        f"{TTP}:readNewProtocolFileToPandas",
        _stream(CHANNELS=37, reorder_channels=True),
        "37 channels reordered",
    ),
    Case(
        f"{TTP}:readNewProtocolFileToPandas",
        _stream(reorder_channels=True),
        "reorder unsupported for 49 channels",
    ),
    Case(
        f"{TTP}:readNewProtocolFileToPandas",
        _stream(force_16bit_step=False, extract_failsafe=True),
        "28-bit step and failsafe",
    ),
    Case(
        f"{TTP}:readNewProtocolFileToPandas",
        _stream(file_offset=10, file_last=700),
        "file range",
    ),
    Case(f"{TTP}:readRAWfileToPandas", _raw(), "file"),
    Case(
        f"{TTP}:readRAWfileToPandas",
        _raw(offset=320, count=500),
        "file offset and count",
    ),
    Case(
        f"{TTP}:readRAWfileToPandas",
        lambda f: printed(lambda: f(buffer=ttm_raw_bytes())),
        "buffer",
    ),
    Case(
        f"{TTP}:readRAWfileToPandas",
        lambda f: printed(
            lambda: f(buffer=ttm_raw_bytes((5, 17, 2999)), removeNoData=False)
        ),
        "buffer keeping no-data rows",
    ),
    Case(
        f"{TTP}:readRAWfileToPandas",
        lambda f: printed(lambda: f(buffer=ttm_raw_bytes((5, 17, 2999)))),
        "buffer dropping no-data rows",
    ),
    Case(
        f"{TTP}:find_nearest_idx",
        lambda f: (f([1.0, 4.0, 9.0, 16.0], 7.0), f(np.array([3, 1, 2]), 1.9)),
    ),
    Case(
        f"{TTP}:autoCalibrateTAP",
        lambda f: printed(
            lambda: f(ttm_raw_frame()["t_L"].to_numpy().astype(np.int64), 4166.0)
        ),
    ),
    Case(
        f"{TTP}:autoCalibrateTAP",
        lambda f: printed(
            lambda: f(
                ttm_raw_frame()["t_3"].to_numpy().astype(np.int64),
                4166.0,
                debugPlots=True,
            )
        ),
        "debug plots",
    ),
    Case(
        f"{TTP}:dt_with_autoCalibratedTAP",
        lambda f: printed(lambda: f(_timing(), 4166.0)),
    ),
    Case(f"{TTP}:dt_with_fixConstant", lambda f: f(_timing(), 4166.0)),
    Case(f"{TTP}:dt_with_fixConstant", lambda f: f(_timing(), 4166.0, kC4=40.0), "kC4"),
    Case(f"{TTP}:fast_histInt", lambda f: f(_counts()), "default minimum"),
    Case(f"{TTP}:fast_histInt", lambda f: f(_counts(), dmin=0), "zero minimum"),
    Case(f"{TTP}:fast_histInt", lambda f: f(_counts(), xrange=(50, 200)), "range"),
    Case(
        f"{TTP}:fitGaus",
        lambda f: printed(lambda: _with_scipy(f, *_histogram_peak())),
    ),
    Case(f"{TTP}:convertFromRAWBuffer", _convert(ttm_raw_bytes), "no channels"),
    Case(
        f"{TTP}:convertFromRAWBuffer",
        _convert(
            ttm_raw_bytes,
            ignorePixelLineFrame=True,
            compressionLevel=1,
            sysclk_MHz=200.0,
            laser_MHz=50.0,
            metadata={"sample": np.asarray(7)},
        ),
        "no image, compression, metadata",
    ),
    Case(
        f"{TTP}:convertFromRAWBuffer",
        _failing(_convert(ttm_raw_bytes, list_of_channels=[0, 3])),
        "channels hit the t_L column name bug",
    ),
    Case(f"{TTP}:convertFromPandasDataFrame", _convert(ttm_raw_frame), "no channels"),
    Case(
        f"{TTP}:convertFromPandasDataFrame",
        _convert(
            ttm_raw_frame,
            ignorePixelLineFrame=True,
            step_max=70000,
            dwell_time_us=500.0,
            metadata={"sample": np.asarray(7)},
        ),
        "no image, step_max, metadata",
    ),
    Case(
        f"{TTP}:convertFromPandasDataFrame",
        _failing(_convert(ttm_raw_frame, list_of_channels=[0, 3])),
        "channels hit the t_L column name bug",
    ),
    Case(
        f"{TTP}:convertFromPandasDataFrame",
        _failing(_convert(ttm_raw_frame, coincidence_analysis=True)),
        "coincidence analysis",
    ),
    Case(f"{TTP}:convertDataRAW", _convert_file(False), "parsed file"),
    Case(
        f"{TTP}:convertDataRAW",
        _convert_file(True, ignorePixelLineFrame=True, compressionLevel=1),
        "RAW file, no image",
    ),
    Case(
        f"{TTP}:convertDataRAW",
        _convert_file(False, chunk_start=200, chunk_stop=2000),
        "parsed file chunk",
    ),
    Case(
        f"{TTP}:convertDataRAW",
        _failing(_convert_file(False, coincidence_analysis=True)),
        "coincidence analysis",
    ),
    Case(
        f"{TTP}:convertDataRAW",
        _failing(_convert_file(False, list_of_channels=[0, 3])),
        "channels hit the t_L column name bug",
    ),
    Case(f"{TTP}:binwidth_normalized", lambda f: f(np.array([3, 1, 4, 1, 5, 9, 2, 6]))),
    Case(
        f"{TTP}:binwidth_normalized",
        lambda f: f(np.random.default_rng(0).random(150)),
        "float counts",
    ),
    Case(f"{TTP}:calculateCalibFromH5", _calibrate(), "channel 1 missing"),
    Case(
        f"{TTP}:calculateCalibFromH5",
        _calibrate(return_numeric_channels=True, plots=True),
        "numeric channels and plots",
    ),
    Case(f"{TTP}:applyCalibDict", _apply(), "channel 0"),
    Case(
        f"{TTP}:applyCalibDict",
        _apply(channel=3, getAbsoluteTime=True),
        "channel 3 absolute time",
    ),
    Case(
        f"{TTP}:image_4d_fast",
        lambda f: f(4, 8, 2, _tables(2), 25000.0),
        "two channels",
    ),
    Case(
        f"{TTP}:image_4d_fast",
        lambda f: f(4, 8, 1, _tables(1), 25000.0, bin_ir=np.array([2.0])),
        "one channel with bin_ir",
    ),
    Case(
        f"{TTP}:image_4d_fast",
        lambda f: error(
            lambda: f(4, 8, 2, _tables(2), 25000.0, bin_ir=np.array([1.0, 2.0]))
        ),
        "bin_ir array of two",
    ),
]

SKIPPED: dict[str, str] = {
    f"{TTP}:decoration_sysclk_laser_ps": SKIP_PLOT_ONLY,
}
