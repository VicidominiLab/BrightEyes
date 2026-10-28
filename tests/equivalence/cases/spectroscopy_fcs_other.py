import importlib

import numpy as np

from .._inputs import decay_histogram, photon_arrival_times, time_trace
from .._registry import SKIP_FILE_IO, SKIP_PLOT_ONLY, Case

P = "brighteyes_ffs.fcs"
ASH = f"{P}.add_sum_column"
ALH = f"{P}.align_lifetime_hist"
AFT = f"{P}.arrivaltimes2filteredtimetrace"
A2T = f"{P}.arrivaltimes2timetrace"
A2H = f"{P}.atimes2hist"
ATD = f"{P}.atimes_data"
BHC = f"{P}.bh_atimes2csv"
B2A = f"{P}.binfile2arrivaltimes"
B2D = f"{P}.binfile2data"
C2C = f"{P}.corr2csv"
CIK = f"{P}.corr_input_from_kw"
D2P = f"{P}.data2phasor"
ESD = f"{P}.extract_spad_data"
ESP = f"{P}.extract_spad_photon_streams"
F2A = f"{P}.fcs2arrivaltimes"
FFILE = f"{P}.fcs_file"
FLG = f"{P}.fcs_load_g"
FB2H = f"{P}.fcsbin2h5"
FD2V = f"{P}.fcsdata2video"
GFI = f"{P}.get_fcs_info"
LTH = f"{P}.lifetimehist"
MAC = f"{P}.meas_to_analogcount"
MTC = f"{P}.meas_to_count"
PCS = f"{P}.plot_corrs_scheme"
PFP = f"{P}.plot_fingerprint"
PIT = f"{P}.plot_intensity_traces"
PPC = f"{P}.plot_pycorrfit"
PLT = f"{P}.plots"
RBA = f"{P}.read_bh_asc"
RBH = f"{P}.read_bh_hist"
RBS = f"{P}.read_bh_spc"
RBF = f"{P}.read_binfile"
U64 = f"{P}.u64_to_counts"


def _error(function, *args, **kwargs) -> BaseException:
    try:
        function(*args, **kwargs)
    except Exception as error:
        return error
    raise AssertionError("the call did not raise")


def _sib(obj, module: str, name: str):
    """Return `name` from the sibling module of the package that `obj` lives in."""
    package = obj.__module__.rsplit(".", 1)[0]
    return getattr(importlib.import_module(f"{package}.{module}"), name)


def _atimes(f, n_channels: int = 25, weights: int = 0, macrotime: float = 1e-5):
    """Arrival-time object with `n_channels` channels and `weights` extra weight columns."""
    obj = _sib(f, "atimes_data", "ATimesData")()
    obj.macrotime = macrotime
    rng = np.random.default_rng(1)
    for index, (name, array) in enumerate(photon_arrival_times().items()):
        if index >= n_channels:
            break
        if weights:
            array = np.column_stack((array, rng.random((len(array), weights))))
        setattr(obj, name, array)
    return obj


def _atimes_of(f, weights: int = 0, tcspc: bool = True):
    """Arrival-time object like `_atimes`, with the `tcspc` flag set."""
    obj = _atimes(f, weights=weights)
    obj.tcspc = tcspc
    return obj


def _with(obj, **attributes):
    """Set attributes on an object and return it."""
    for name, value in attributes.items():
        setattr(obj, name, value)
    return obj


def _summary(obj):
    """Read every property and the filter-less array conversion of an arrival-time object."""
    return (
        obj.data,
        obj.all_channels,
        obj.num_channels,
        obj.num_filters,
        obj.to_time_channel_array(obj.num_filters - 1 if obj.num_filters else 0),
        obj,
    )


def _inplace(function, obj, *args, **kwargs):
    """Call an in-place function and return the object it changed."""
    function(obj, *args, **kwargs)
    return obj


def _hist_object(f, n: int = 25):
    """Arrival-time object holding `hist0` ... columns of (time, counts)."""
    obj = _sib(f, "atimes_data", "ATimesData")()
    for channel, counts in enumerate(decay_histogram()[:n]):
        setattr(obj, f"hist{channel}", np.column_stack((np.arange(256) * 100, counts)))
    return obj


def _filters(n_det: int = 25, n_bins: int = 64, n_filters: int = 2) -> np.ndarray:
    filters = np.random.default_rng(4).random((n_det, n_bins, n_filters))
    return filters / filters.sum(axis=1, keepdims=True)


def _spad(cols: int = 26) -> np.ndarray:
    return np.random.default_rng(2).poisson(2.0, (300, cols)).astype(np.uint32)


def _spad_5d() -> np.ndarray:
    return np.random.default_rng(2).poisson(2.0, (2, 3, 3, 4, 25)).astype(np.uint32)


def _arrivals(f, unique_channel: bool = False, sum_unique: bool = False):
    """Photon bins per channel: det0 repeats a bin, det1 repeats one unless asked, det2 is empty."""
    obj = _sib(f, "binfile2arrivaltimes", "ArrivalTimesObject")()
    macro = photon_arrival_times()
    for channel in range(25):
        setattr(
            obj,
            f"det{channel}",
            [int(t) for t in np.sort(macro[f"det{channel}"][:, 0] // 1000)],
        )
    obj.det1 = (
        list(range(0, 700, 7)) if unique_channel else [t // 2 * 7 for t in range(100)]
    )
    obj.det2 = []
    obj.sum = (
        list(range(0, 900, 3))
        if sum_unique
        else [int(t) for t in np.sort(macro["det3"][:, 0] // 1000)]
    )
    obj.dwellTime = []
    return obj


def _lifetime_object(f):
    """Arrival times with microtimes (ps) spanning several laser periods."""
    obj = _sib(f, "atimes_data", "ATimesData")()
    rng = np.random.default_rng(6)
    for channel in range(4):
        macro = np.sort(rng.integers(0, 10**6, 300))
        micro = rng.integers(0, 40000, 300)
        setattr(obj, f"det{channel}", np.column_stack((macro, micro)).astype(np.int64))
    return obj


def _empty_object(f):
    obj = f()
    obj.value = 1
    return obj


def _matrix_to_count(f) -> np.ndarray:
    values = np.random.default_rng(3).integers(
        0, np.iinfo(np.uint64).max, (64, 2), dtype=np.uint64
    )
    out = np.zeros((64, 25), dtype=np.uint16)
    f(values, out)
    return out


def _bytes(*values: int) -> np.ndarray:
    return np.array(values, dtype=np.uint8)


ALL_BUT_FIVE = [
    "central",
    "sum3",
    "sum5",
    "chess0",
    "chess1",
    "picentral",
    "piring1",
    "piring2",
    "piring3",
    "airycentral",
    "airyring1",
    "airyring2",
    "airyring3",
    "prismcentral",
    "prismsum3",
    "prismsum5",
    "prismsum7",
]
SPAD_2D = [
    "sum_all",
    "central",
    "sum3",
    "sum5left",
    "sum5right",
    "sum5top",
    "sum5bottom",
    "allbuthot",
    "sum5",
    "outer",
    "chess0",
    "chess1",
    "chess3a",
    "chess3b",
    "upper_left",
    "lower_right",
]
SPAD_5D = ["sum3_5D", "allbuthot_5D"]
SWITCH_2D = [
    "central",
    "center",
    "sum3",
    "allbuthot",
    "sum5",
    "sum",
    "outer",
    "chess0",
    "chess1",
    "chess3a",
    "chess3b",
    "upper_left",
    "lower_right",
    "sum5left",
    "sum5right",
    "sum5top",
    "sum5bottom",
]
SWITCH_5D = ["sum3_5D", "allbuthot_5D"]
CORR_INPUT = [
    ("all autocorrelations", "square 5x5"),
    ("all autocorrelations", "pda-23"),
    ("all autocorrelations", "airyscan 32"),
    ("fcs", "5x5"),
    ("spot-variation fcs", "square 5x5"),
    ("spot-variation fcs", "pda-23"),
    ("spot-variation fcs", "airyscan 32"),
    ("svfcs", "nsparc"),
    ("spot-variation fcs - extended", "square 5x5"),
    ("spot-variation fcs - extended", "pda-23"),
    ("spot-variation fcs - extended", "airyscan 32"),
    ("svfcs-ext", "luminosa"),
    ("pair-correlation fcs", "square 5x5"),
    ("pair-correlation fcs", "pda-23"),
    ("pair-correlation fcs", "airyscan 32"),
    ("all cross-correlations", "square 5x5"),
    ("all cross-correlations", "pda-23"),
    ("all cross-correlations", "airyscan 32"),
    ("xcorr", "pda23"),
    ("Cross-Correlation FFS", "AIRYSCAN"),
    ("crossall", "airyscan"),
    ("crossall-ext", "square 5x5"),
    ("crossall-ext", "pda-23"),
    ("crossall-ext", "airyscan 32"),
    ("all autocorrelations", "unknown detector"),
    ("unknown correlation", "square 5x5"),
]


CASES: list[Case] = [
    # add_sum_column
    Case(f"{ASH}:add_sum_column", lambda f: f(time_trace()[:50])),
    # align_lifetime_hist
    Case(f"{ALH}:align_hist", lambda f: _inplace(f, _hist_object(f)), "default_window"),
    Case(f"{ALH}:align_hist", lambda f: _inplace(f, _hist_object(f), n=5), "window_5"),
    # arrivaltimes2filteredtimetrace
    Case(f"{AFT}:ATimesData", _empty_object),
    Case(
        f"{AFT}:arrival_times_2_filtered_time_trace",
        lambda f: f(_with(_atimes(f), microtime=1e-9), _filters(), 4e-9, 1.0),
        "multi_detector",
    ),
    Case(
        f"{AFT}:arrival_times_2_filtered_time_trace",
        lambda f: f(
            _with(_atimes(f, n_channels=1), microtime=1e-9), _filters(1)[0], 4e-9, 2.0
        ),
        "single_detector_2d_filter",
    ),
    Case(
        f"{AFT}:atimes_filtered",
        lambda f: _inplace(
            f, _with(_atimes(f), microtime=1e-9, microbintime=2e-9), _filters()
        ),
        "micro_in_time_units",
    ),
    Case(
        f"{AFT}:atimes_filtered",
        lambda f: _inplace(f, _atimes(f), _filters(), micro_bin=True),
        "micro_in_bins_clipped",
    ),
    Case(
        f"{AFT}:atimes_filtered",
        lambda f: _inplace(
            f,
            _with(_atimes(f, n_channels=1), microtime=1e-9, microbintime=4e-9),
            _filters(1)[0],
            verbose=True,
        ),
        "single_detector_2d_filter_verbose",
    ),
    Case(
        f"{AFT}:atimes_filtered",
        lambda f: _inplace(
            f, _atimes(f, weights=2), _filters(25, 64, 3), micro_bin=True
        ),
        "replaces_existing_weight_columns",
    ),
    # arrivaltimes2timetrace
    Case(f"{A2T}:atimes2timetrace", lambda f: f(_arrivals(f)), "repeated_bins"),
    # a channel or a sum without repeated bins assigns a (N, 1) array to a (N,) column, which numpy rejects
    Case(
        f"{A2T}:atimes2timetrace",
        lambda f: _error(f, _arrivals(f, unique_channel=True)),
        "unique_channel",
    ),
    Case(
        f"{A2T}:atimes2timetrace",
        lambda f: _error(f, _arrivals(f, sum_unique=True)),
        "unique_sum",
    ),
    Case(
        f"{A2T}:atimes2timetrace_bh",
        lambda f: f(photon_arrival_times()["det0"][:, 0], 1000.0),
        "bins_1000",
    ),
    Case(
        f"{A2T}:atimes2timetrace_bh",
        lambda f: f(photon_arrival_times()["det5"][:, 0].astype(float), 12345.5),
        "float_times",
    ),
    # atimes2hist
    Case(f"{A2H}:atimes_data_2_hist", lambda f: _inplace(f, _atimes(f)), "auto_bin"),
    Case(
        f"{A2H}:atimes_data_2_hist",
        lambda f: _inplace(f, _atimes(f), bintime=8),
        "bintime_8",
    ),
    Case(
        f"{A2H}:atimes_data_2_hist",
        lambda f: _inplace(
            f, _atimes(f), bintime=8, ttm_correction=True, laser_freq=1e10
        ),
        "ttm_correction",
    ),
    # atimes_data
    Case(
        f"{ATD}:ATimesData",
        lambda f: _summary(_atimes_of(f)),
        "properties_tcspc",
    ),
    Case(
        f"{ATD}:ATimesData",
        lambda f: _summary(_atimes_of(f, weights=2, tcspc=False)),
        "properties_weights_no_tcspc",
    ),
    Case(
        f"{ATD}:ATimesData.to_time_channel_array",
        lambda f: f(_atimes_of(f)),
        "all_channels",
    ),
    Case(
        f"{ATD}:ATimesData.to_time_channel_array",
        lambda f: f(_atimes_of(f, weights=2), 1),
        "second_filter",
    ),
    Case(
        f"{ATD}:ATimesData.to_time_channel_array",
        lambda f: f(
            _atimes_of(f), 0, [[12], [6, 7, 8, 11, 12, 13, 16, 17, 18], list(range(25))]
        ),
        "channel_groups",
    ),
    Case(
        f"{ATD}:ATimesData.to_time_channel_array",
        lambda f: _error(f, _atimes_of(f), 0, [[12], [30]]),
        "invalid_channel",
    ),
    Case(f"{ATD}:ATimesData.duration", lambda f: f(_atimes_of(f)), "last_photon"),
    Case(
        f"{ATD}:ATimesData.duration",
        lambda f: f(_atimes_of(f), subtract_start_time=True),
        "subtract_start",
    ),
    Case(
        f"{ATD}:ATimesData.duration",
        lambda f: f(_atimes_of(f), return_period=True),
        "period",
    ),
    Case(
        f"{ATD}:ATimesData.get_channels", lambda f: f(_hist_object(f), "hist"), "hist"
    ),
    Case(
        f"{ATD}:ATimesData.calc_hist",
        lambda f: _inplace(f, _atimes_of(f)),
        "auto_bin",
    ),
    Case(
        f"{ATD}:ATimesData.calc_hist",
        lambda f: _inplace(f, _atimes_of(f), bintime=8),
        "bintime_8",
    ),
    Case(
        f"{ATD}:ATimesData.calc_hist",
        lambda f: _inplace(
            f, _atimes_of(f), bintime=8, ttm_correction=True, laser_freq=1e10
        ),
        "ttm_correction",
    ),
    Case(
        f"{ATD}:ATimesData.align_hist",
        lambda f: _inplace(f, _hist_object(f)),
        "default_window",
    ),
    Case(
        f"{ATD}:ATimesData.align_hist",
        lambda f: _inplace(f, _hist_object(f), n=5),
        "window_5",
    ),
    Case(f"{ATD}:atimes_data_2_channels", lambda f: f(_atimes(f, n_channels=12))),
    Case(f"{ATD}:atimes_data_2_duration", lambda f: f(_atimes(f)), "auto_macrotime"),
    Case(
        f"{ATD}:atimes_data_2_duration",
        lambda f: f(_atimes(f), 1e-12),
        "explicit_macrotime",
    ),
    Case(
        f"{ATD}:atimes_data_2_duration",
        lambda f: f(_atimes(f), 1e-12, subtract_start_time=True),
        "subtract_start",
    ),
    Case(
        f"{ATD}:atimes_data_2_duration",
        lambda f: f(_atimes(f), 1e-12, return_period=True),
        "period",
    ),
    Case(f"{ATD}:atimes_data_attr_2_ch", lambda f: f("det31")),
    # binfile2arrivaltimes, binfile2data, read_bh_*, other empty classes
    Case(f"{B2A}:ArrivalTimesObject", _empty_object),
    Case(f"{B2D}:ArrivalTimes", _empty_object),
    Case(f"{F2A}:ATimesData", _empty_object),
    Case(f"{GFI}:infoObj", _empty_object),
    Case(f"{PPC}:PyCorrfit_data", _empty_object),
    Case(f"{RBA}:DataObj", _empty_object),
    Case(f"{RBH}:DataObj", _empty_object),
    Case(f"{RBS}:BHspcInfo", _empty_object),
    # binfile2data byte decoding
    Case(f"{B2D}:getaddress", lambda f: f(_bytes(0b10100000, 0, 0, 0)), "address_5"),
    Case(f"{B2D}:getaddress", lambda f: f(_bytes(0b00011111, 1, 2, 3)), "address_0"),
    Case(f"{B2D}:gettime", lambda f: f(_bytes(0, 0, 0b00001010, 0b01010101)), "time"),
    Case(f"{B2D}:gettime", lambda f: f(_bytes(0, 0, 0b11111111, 255)), "time_max"),
    Case(f"{B2D}:getchannel", lambda f: f(_bytes(0, 0b00000001, 0, 0)), "channel_1"),
    Case(f"{B2D}:getchannel", lambda f: f(_bytes(0, 0b00000010, 0, 0)), "channel_2"),
    Case(f"{B2D}:getchannel", lambda f: f(_bytes(0b00000001, 0, 0, 0)), "channel_3"),
    Case(f"{B2D}:getchannel", lambda f: f(_bytes(0b00000010, 0, 0, 0)), "channel_4"),
    Case(f"{B2D}:getchannel", lambda f: f(_bytes(0, 0, 0, 0)), "no_channel"),
    Case(f"{B2D}:getchannel", lambda f: f(_bytes(0b00000011, 0, 0, 0)), "two_channels"),
    # corr_input_from_kw
    *[
        Case(f"{CIK}:get_corr_input", lambda f, c=c, d=d: f(c, d), f"{c}|{d}")
        for c, d in CORR_INPUT
    ],
    # data2phasor
    Case(f"{D2P}:data2phasor", lambda f: f(_hist_object(f))),
    # extract_spad_data
    Case(f"{ESD}:extract_spad_data", lambda f: f(_spad(), "sum_all"), "sum_all"),
    Case(f"{ESD}:extract_spad_data", lambda f: f(_spad(), "C3+7+8+12"), "custom_sum"),
    Case(f"{ESD}:extract_spad_data", lambda f: f(_spad(), "C5"), "custom_single"),
    Case(f"{ESD}:extract_spad_data", lambda f: f(_spad(), "N1+3+12"), "all_but"),
    *[
        Case(f"{ESD}:extract_spad_data", lambda f, k=k: f(_spad(49), k), f"keyword_{k}")
        for k in ALL_BUT_FIVE
    ],
    *[
        Case(f"{ESD}:extract_spad_data", lambda f, m=m: f(_spad(), m), f"channel_{m}")
        for m in (0, 12, 24)
    ],
    *[
        Case(
            f"{ESD}:extract_spad_data",
            lambda f, m=m: f(_spad(), m),
            f"all_but_channel_{-m}",
        )
        for m in (-1, -3, -25, -26)
    ],
    # "sum" and "outer" are documented modes that keyword_2_ch lacks, so the old code raises too
    Case(
        f"{ESD}:extract_spad_data",
        lambda f: _error(f, _spad(), "sum"),
        "sum_unsupported",
    ),
    Case(
        f"{ESD}:extract_spad_data",
        lambda f: _error(f, _spad(), "outer"),
        "outer_unsupported",
    ),
    *[Case(f"{ESD}:{name}", lambda f: f(_spad())) for name in SPAD_2D],
    *[Case(f"{ESD}:{name}", lambda f: f(_spad_5d())) for name in SPAD_5D],
    *[
        Case(f"{ESD}:switch_mode_string", lambda f, m=m: f(_spad(), m), m)
        for m in SWITCH_2D
    ],
    *[
        Case(f"{ESD}:switch_mode_string", lambda f, m=m: f(_spad_5d(), m), m)
        for m in SWITCH_5D
    ],
    Case(
        f"{ESD}:switch_mode_string",
        lambda f: _error(f, _spad(), "nothing"),
        "invalid_mode",
    ),
    # extract_spad_photon_streams
    Case(
        f"{ESP}:extract_spad_photon_streams",
        lambda f: f(_atimes(f, weights=2), "C3+4+7+12"),
        "custom_sum",
    ),
    Case(
        f"{ESP}:extract_spad_photon_streams",
        lambda f: f(_atimes(f, weights=2), 5),
        "single",
    ),
    Case(
        f"{ESP}:extract_spad_photon_streams",
        lambda f: f(_atimes(f, weights=2), "sum3"),
        "keyword_sum3",
    ),
    Case(
        f"{ESP}:extract_spad_photon_streams",
        lambda f: f(_atimes(f, weights=1), "central"),
        "keyword_central",
    ),
    Case(
        f"{ESP}:extract_spad_photon_streams",
        lambda f: _error(f, _atimes(f, weights=2), "nothing"),
        "unknown_keyword",
    ),
    # fcs_load_g
    Case(
        f"{FLG}:strip_g_fname",
        lambda f: f("data/FCS_beads_central.csv", "FCS_beads_"),
        "print",
    ),
    Case(
        f"{FLG}:strip_g_fname",
        lambda f: f("data/FCS_beads_sum3.csv", "FCS_beads_", False),
        "quiet",
    ),
    # fcsbin2h5
    Case(
        f"{FB2H}:numpy_timetrace_2_atimes",
        lambda f: f(time_trace()[:1000], 1e-6),
    ),
    Case(f"{FB2H}:photon_arrivals", lambda f: f(time_trace()[:1000], 3), "channel_3"),
    Case(
        f"{FB2H}:photon_arrivals",
        lambda f: f(np.zeros((50, 4), dtype=np.uint32), 1),
        "no_photons",
    ),
    # get_fcs_info
    Case(f"{GFI}:get_metafile_from_file", lambda f: f("data/run1.bin"), "bin"),
    Case(
        f"{GFI}:get_metafile_from_file",
        lambda f: f("data/run1.bin", "_meta.txt"),
        "bin_custom",
    ),
    Case(f"{GFI}:get_metafile_from_file", lambda f: f("data/run1.h5"), "h5"),
    Case(
        f"{GFI}:get_fcs_info_singleparam",
        lambda f: f("a: 1\nDwell time: 2.5\nduration: 10", "Dwell time"),
        "middle_line",
    ),
    Case(
        f"{GFI}:get_fcs_info_singleparam",
        lambda f: f("a: 1\nDwell time: 2.5\nduration: 10", "duration"),
        "last_line",
    ),
    # lifetimehist
    Case(f"{LTH}:lifetimehist", lambda f: _inplace(f, _lifetime_object(f)), "default"),
    Case(
        f"{LTH}:lifetimehist",
        lambda f: _inplace(
            f, _lifetime_object(f), m_bins=64, laser_freq=40e6, microtime=1e-12
        ),
        "bins_64_40MHz",
    ),
    # meas_to_count
    Case(f"{MTC}:matrix_to_count", _matrix_to_count),
    Case(
        f"{MTC}:reshape_to_5d",
        lambda f: f(np.arange(2 * 3 * 4 * 5 * 25).reshape(-1, 25), 2, 3, 4, 5),
    ),
    Case(
        f"{MTC}:reshape_to_6d",
        lambda f: f(np.arange(2 * 2 * 3 * 4 * 5 * 25).reshape(-1, 25), 2, 2, 3, 4, 5),
        "time_axis",
    ),
    Case(
        f"{MTC}:reshape_to_6d",
        lambda f: f(np.arange(1 * 2 * 3 * 4 * 25).reshape(-1, 25), 1, 2, 3, 4),
        "default_time",
    ),
    # plot_fingerprint
    Case(
        f"{PFP}:plot_fingerprint_airyscan",
        lambda f: f(np.arange(32) * 3.0 + np.random.default_rng(0).random(32)),
        "default",
    ),
    Case(
        f"{PFP}:plot_fingerprint_airyscan",
        lambda f: f(np.arange(32), cmap="viridis", plot=True),
        "plot",
    ),
    Case(f"{PFP}:plot_fingerprint_airyscan", lambda f: f(np.full(32, 4)), "constant"),
    Case(
        f"{PFP}:plot_fingerprint_luminosa",
        lambda f: f(np.random.default_rng(0).random(23)),
        "23_elements",
    ),
    Case(
        f"{PFP}:plot_fingerprint_luminosa",
        lambda f: f(np.random.default_rng(0).random(32), plot=True),
        "32_elements_plot",
    ),
    Case(f"{PFP}:plot_fingerprint_luminosa", lambda f: f(np.full(23, 2.0)), "constant"),
    Case(f"{PFP}:plot_fingerprint5x5", lambda f: f(time_trace()[:500]), "array_2d"),
    Case(
        f"{PFP}:plot_fingerprint5x5",
        lambda f: f(time_trace()[:500], show_perc="numbers", normalize=True),
        "array_numbers_normalized",
    ),
    Case(
        f"{PFP}:plot_fingerprint5x5",
        lambda f: f(time_trace()[:500], show_perc=False, vminmax=[0, 2000]),
        "array_no_perc_fixed_range",
    ),
    Case(
        f"{PFP}:plot_fingerprint5x5",
        lambda f: f(time_trace()[:500], vminmax=[0, 2000]),
        "array_perc_fixed_range",
    ),
    Case(
        f"{PFP}:plot_fingerprint5x5",
        lambda f: f(time_trace()[:500].sum(axis=0), dtype="int64"),
        "array_1d",
    ),
    Case(f"{PFP}:plot_fingerprint5x5", lambda f: f(_atimes(f)), "atimes_25"),
    Case(
        f"{PFP}:plot_fingerprint5x5",
        lambda f: f(_atimes(f, n_channels=21)),
        "atimes_21",
    ),
    Case(
        f"{PFP}:plot_fingerprint5x5",
        lambda f: f(
            np.random.default_rng(0).poisson(3.0, (200, 49)).astype(np.int64), n_ch=49
        ),
        "array_7x7",
    ),
    # u64_to_counts
    Case(f"{U64}:u64_to_counts", lambda f: f(5), "below_threshold"),
    Case(f"{U64}:u64_to_counts", lambda f: f(31), "threshold_31"),
    Case(f"{U64}:u64_to_counts", lambda f: f(32), "threshold_32"),
    Case(f"{U64}:u64_to_counts", lambda f: f(33), "cluster_0_small"),
    Case(f"{U64}:u64_to_counts", lambda f: f(48), "cluster_1_small"),
    Case(
        f"{U64}:u64_to_counts", lambda f: f(13091560953979690842), "documented_example"
    ),
    Case(f"{U64}:u64_to_counts", lambda f: f(2**64 - 1), "all_bits_set"),
    Case(
        f"{U64}:u64_to_counts",
        lambda f: f(np.uint64(0x0123456789ABCDE0)),
        "numpy_uint64_cluster_0",
    ),
    Case(
        f"{U64}:u64_to_counts",
        lambda f: f(np.uint64(0x0123456789ABCDF0)),
        "numpy_uint64_cluster_1",
    ),
    Case(f"{U64}:binary_to_counts", lambda f: f(["1", "0", "1", "1"], 3)),
    # plot_pycorrfit: np.float was removed in numpy 1.24, so the old code raises as well
    Case(
        f"{PPC}:read_pycorrfit_singleparam",
        lambda f: _error(f, "#   n\t2.5\n#   SP\t3\n", "#   n\t", "\n"),
    ),
]


SKIPPED: dict[str, str] = {
    f"{A2T}:atimes2data": SKIP_FILE_IO,
    f"{ATD}:load_atimes_data": SKIP_FILE_IO,
    f"{ATD}:write_atimes_data": SKIP_FILE_IO,
    f"{ATD}:atimes_file_2_nch": SKIP_FILE_IO,
    f"{ATD}:load_atimes_data_pandas": SKIP_FILE_IO,
    f"{ATD}:load_atimes_data_newprot": SKIP_FILE_IO,
    f"{ATD}:atimes_h5_to_hdf5": SKIP_FILE_IO,
    f"{ATD}:atimes_raw_2_h5all": SKIP_FILE_IO,
    f"{ATD}:atimes_raw_2_h5": SKIP_FILE_IO,
    f"{ATD}:h5_atimes_2_data": SKIP_FILE_IO,
    f"{BHC}:bh_atimes2csv": SKIP_FILE_IO,
    f"{B2A}:binfile2arrivaltimes": SKIP_FILE_IO,
    f"{B2D}:binfile2data": SKIP_FILE_IO,
    f"{B2D}:binfile2dataAle": SKIP_FILE_IO,
    f"{C2C}:corr2csv": SKIP_FILE_IO,
    f"{FFILE}:load_file": SKIP_FILE_IO,
    f"{FLG}:load_g": SKIP_FILE_IO,
    f"{FB2H}:fcsbin2h5": SKIP_FILE_IO,
    f"{FB2H}:fcsbin2h5_atimes": SKIP_FILE_IO,
    f"{FB2H}:fcsbin2atimes": SKIP_FILE_IO,
    f"{FB2H}:photon_data_2_h5": SKIP_FILE_IO,
    f"{FD2V}:fcsdata2video": SKIP_FILE_IO,
    f"{FD2V}:fcstraces2video": SKIP_FILE_IO,
    f"{FD2V}:partpos2video": SKIP_FILE_IO,
    f"{GFI}:get_fcs_info": SKIP_FILE_IO,
    f"{GFI}:get_file_info": SKIP_FILE_IO,
    f"{GFI}:get_file_info_txt": SKIP_FILE_IO,
    f"{GFI}:get_file_info_h5": SKIP_FILE_IO,
    f"{GFI}:get_file_info_ptu": SKIP_FILE_IO,
    f"{GFI}:get_file_info_czi": SKIP_FILE_IO,
    f"{GFI}:get_file_info_tiff": SKIP_FILE_IO,
    f"{GFI}:get_duration_from_h5_timetagging_data": SKIP_FILE_IO,
    f"{MAC}:file_to_analogcount": SKIP_FILE_IO,
    f"{MTC}:file_to_count": SKIP_FILE_IO,
    f"{MTC}:file_to_fcs_count": SKIP_FILE_IO,
    f"{MTC}:czi2h5": SKIP_FILE_IO,
    f"{MTC}:mat2h5": SKIP_FILE_IO,
    f"{MTC}:numpy2h5": SKIP_FILE_IO,
    f"{MTC}:bin2np": SKIP_FILE_IO,
    f"{MTC}:image2h5": SKIP_FILE_IO,
    f"{MTC}:locseries2h5": SKIP_FILE_IO,
    f"{MTC}:all_timetr_2_h5": SKIP_FILE_IO,
    f"{MTC}:bin2h5": SKIP_FILE_IO,
    f"{MTC}:all_bin_images_2_tiff": SKIP_FILE_IO,
    f"{MTC}:image2tiff": SKIP_FILE_IO,
    f"{MTC}:all_bin_images_2_h5": SKIP_FILE_IO,
    f"{PCS}:plot_correlations_microimage": SKIP_PLOT_ONLY,
    f"{PCS}:plot_det_element_map": SKIP_PLOT_ONLY,
    f"{PFP}:plot_fingerprint7x7": SKIP_PLOT_ONLY,
    f"{PFP}:plot_det_dist": SKIP_PLOT_ONLY,
    f"{PIT}:plot_intensity_traces": SKIP_PLOT_ONLY,
    f"{PPC}:plot_pycorr_all": SKIP_FILE_IO,
    f"{PPC}:plot_g_csv": SKIP_FILE_IO,
    f"{PPC}:plot_pycorrfit": SKIP_FILE_IO,
    f"{PPC}:read_pycorrfit": SKIP_FILE_IO,
    f"{PPC}:plot_pycorrfit3": SKIP_FILE_IO,
    f"{PLT}:plot_timetrace": SKIP_PLOT_ONLY,
    f"{PLT}:plot_atimetrace": SKIP_PLOT_ONLY,
    f"{PLT}:plot_atimeshist": SKIP_PLOT_ONLY,
    f"{PLT}:plot_corrs": SKIP_PLOT_ONLY,
    f"{PLT}:plot_corrs_av": SKIP_PLOT_ONLY,
    f"{PLT}:plot_corrs_fit": SKIP_PLOT_ONLY,
    f"{PLT}:plot_difflaw": SKIP_PLOT_ONLY,
    f"{PLT}:plot_flow_heat_map": SKIP_PLOT_ONLY,
    f"{PLT}:plot_anisotropy_heat_map": SKIP_PLOT_ONLY,
    f"{PLT}:plot_fida_hist": SKIP_PLOT_ONLY,
    f"{RBA}:read_asc": SKIP_FILE_IO,
    f"{RBH}:read_hist": SKIP_FILE_IO,
    f"{RBS}:read_spc": SKIP_FILE_IO,
    f"{RBF}:read_binfile": SKIP_FILE_IO,
}
