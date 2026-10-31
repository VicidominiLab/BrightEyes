# Ported from VicidominiLab/libttp, libttp/ttpCython.pyx at 7fe694a.
"""Decode time-tagger streams, with the per-event loops in `_ttp_core`."""

import math
import os

import numpy as np
import tqdm as tq

from . import _ttp_core


def _c_integer(value, dtype):
    """Return `value` as a Python integer within the range of `dtype`.

    A value that is not an `int` is converted with its type's `__int__`.

    Raises
    ------
    TypeError
        If `value` has no integer conversion.
    OverflowError
        If the integer does not fit `dtype`.
    """
    if not isinstance(value, int):
        try:
            convert = type(value).__int__
        except AttributeError:
            raise TypeError("an integer is required") from None
        value = convert(value)
    limits = np.iinfo(dtype)
    if value < limits.min:
        raise OverflowError(f"can't convert negative value to {limits.dtype}")
    if value > limits.max:
        raise OverflowError(f"value too large to convert to {limits.dtype}")
    return value


def _typed(values, dtype):
    """Return `values` as a one-dimensional array of `dtype`, without converting it.

    Raises
    ------
    ValueError
        If the array is not one-dimensional or its dtype is not `dtype`.
    """
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype != dtype:
        raise ValueError(
            f"Buffer dtype mismatch, expected '{np.dtype(dtype)}' but got "
            f"'{array.dtype}' with {array.ndim} dimensions"
        )
    return array


def _check_period(length, pbar_period):
    """Raise `ZeroDivisionError` when there are rows but the progress period is zero."""
    if length > 0 and pbar_period == 0:
        raise ZeroDivisionError("integer modulo by zero")


def getLengthFileNewProtocol(filename, dtype=np.uint16):
    """Return the length of `filename` in units of `dtype`."""
    stats = os.stat(filename)
    # bytes / (2 bytes/uint16)
    return int(stats.st_size // np.dtype(dtype).itemsize)


def timeProcessNewProtocol(filename, CHANNELS=49, file_offset=0, file_last=-1):
    """Read a new-protocol stream and return one row of words per event."""
    print(file_offset, file_last)
    print(
        "File total size in",
        str(np.dtype(np.uint16)),
        "units:",
        getLengthFileNewProtocol(filename, dtype=np.uint16),
    )
    if file_last == -1:
        count = -1
    else:
        count = file_last - file_offset
    print(count)
    dataIn = np.fromfile(filename, dtype=np.uint16, offset=file_offset * 2, count=count)
    print("File read size in", str(np.dtype(np.uint16)), "units:", len(dataIn))

    MAXWORD = _c_integer(CHANNELS + 5, np.uint16)
    print("Channels:", CHANNELS)
    print("MAXWORD:", MAXWORD)

    dataIn = dataIn[dataIn != 0b0111_1111_1111_1111]

    ID = np.right_shift(np.bitwise_and(dataIn, 0b0111_1111_0000_0000), 8)
    valids = np.right_shift(np.bitwise_and(dataIn, 0b1000_0000_0000_0000), 15)
    data = np.bitwise_and(dataIn, 0b0000_0000_1111_1111)
    maxevents = np.count_nonzero(ID == 127) + 1

    print(maxevents)
    # An integer id is below CHANNELS exactly when it is below its ceiling.
    dataout, current_pointer = _ttp_core.new_protocol_events(
        ID, valids, data, math.ceil(CHANNELS), MAXWORD, maxevents
    )
    return dataout[:current_pointer]


def timeProcessRAW(datapre, channel_number, verbose=False):
    """Pair the photons of one channel with the laser pulse that follows each."""
    write_en_tdc_name = "valid_tdc_%d" % channel_number  # noqa: UP031
    channel = "t_%d" % channel_number  # noqa: UP031
    if verbose:
        print("Data processing...")
    datapre = datapre[(datapre[write_en_tdc_name] > 0) | (datapre["valid_tdc_L"] > 0)]
    print("...")
    length = len(datapre["t_L"])

    t_L = _typed(datapre["t_L"], np.uint8)
    t_1 = _typed(datapre[channel], np.uint8)
    step = _typed(datapre["step"], np.uint16)
    valid_1 = _typed(datapre[write_en_tdc_name], np.uint8)
    valid_L = _typed(datapre["valid_tdc_L"], np.uint8)
    cumulative_step = _typed(datapre["cumulativeStep"], np.int64)
    theIndex = _typed(datapre.index.to_numpy(), np.int64)

    pbar = tq.tqdm(total=length)  # progress bar
    pbar_period = int(length / 100)
    print("starting loop")
    _check_period(length, pbar_period)
    tL_v, t1_v, SL_64_v, S1_64_v, theIndex_v, countGood = _ttp_core.raw_tdc_pairs(
        t_L, t_1, step, valid_1, valid_L, cumulative_step, theIndex
    )
    for _ in range(0, length, pbar_period or 1):
        pbar.update(pbar_period)

    print("Data ready, conversion to array")
    dataout = {}

    dataout["t_L"] = tL_v[0:countGood]
    dataout["t_Ch"] = t1_v[0:countGood]

    dataout["S_L"] = SL_64_v[0:countGood]
    dataout["S_Ch"] = S1_64_v[0:countGood]

    dataout["index"] = theIndex_v[0:countGood]
    if verbose:
        print(dataout)
    if verbose:
        print("Data processed!")
    return dataout


def analysisForImg(dataframeInput, StepPerPixel_in, cumulativeStep):
    """Return the pixel, line, frame, corrected pixel and index of every row."""
    StepPerPixel = _c_integer(StepPerPixel_in, np.int64)

    length = len(dataframeInput["scan_enable"])

    scan_enable = _typed(dataframeInput["scan_enable"], np.uint8)
    line_enable = _typed(dataframeInput["line_enable"], np.uint8)
    pixel_enable = _typed(dataframeInput["pixel_enable"], np.uint8)
    cumulative_step = _typed(cumulativeStep.astype(np.int64), np.int64)
    theIndex = _typed(dataframeInput.index.to_numpy(), np.int64)
    stepNewLine = _c_integer(cumulativeStep[0], np.int64)

    print("Arrays copied into analysisForImg")

    pbar = tq.tqdm(total=length)  # progress bar
    pbar_period = int(length / 100)
    _check_period(length, pbar_period)
    arr_px_v, arr_py_v, arr_frame_v, arr_px_corr_v, arr_index_v, frame = (
        _ttp_core.image_indices(
            scan_enable,
            line_enable,
            pixel_enable,
            cumulative_step,
            theIndex,
            stepNewLine,
            StepPerPixel,
        )
    )
    for _ in range(0, length, pbar_period or 1):
        pbar.update(pbar_period)
        pbar.set_description(f"Current frame: {frame} \t")

    print("Total Frame:", frame)
    return arr_px_v, arr_py_v, arr_frame_v, arr_px_corr_v, arr_index_v
