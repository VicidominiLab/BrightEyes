"""Equivalence cases for `brighteyes_ffs.fcs`, first half of the modules."""

import contextlib
import io
from importlib import import_module
from types import SimpleNamespace

import numpy as np
from _inputs import (
    CHANNELS,
    correlation_curve,
    decay_histogram,
    ism_series,
    photon_arrival_times,
    photon_streams,
    time_channel_array,
    time_trace,
)
from _registry import SKIP_FILE_IO, SKIP_PLOT_ONLY, Case
from matplotlib import pyplot as plt

FCS = "brighteyes_ffs.fcs"
ADD_SUM = f"{FCS}.add_sum_column"
ALIGN = f"{FCS}.align_lifetime_hist"
ATIMES_FILTERED = f"{FCS}.arrivaltimes2filteredtimetrace"
ATIMES_TRACE = f"{FCS}.arrivaltimes2timetrace"
ATIMES_CORR = f"{FCS}.atimes2corr"
ATIMES_PARALLEL = f"{FCS}.atimes2corrparallel"
ATIMES_HIST = f"{FCS}.atimes2hist"
ATIMES_DATA = f"{FCS}.atimes_data"
BH = f"{FCS}.bh_atimes2csv"
BIN_ATIMES = f"{FCS}.binfile2arrivaltimes"
BIN_DATA = f"{FCS}.binfile2data"
CORR_CSV = f"{FCS}.corr2csv"
CORR_KW = f"{FCS}.corr_input_from_kw"
PHASOR = f"{FCS}.data2phasor"
DETECTORS = f"{FCS}.detectors"
DISTANCE = f"{FCS}.distance2detelements"
SPAD_DATA = f"{FCS}.extract_spad_data"
SPAD_STREAMS = f"{FCS}.extract_spad_photon_streams"
FCS_ATIMES = f"{FCS}.fcs2arrivaltimes"
FCS_CORR = f"{FCS}.fcs2corr"
DIFFTIME = f"{FCS}.fcs2difftime"
ANALYTICAL = f"{FCS}.fcs_analytical"
FILE = f"{FCS}.fcs_file"
FIT = f"{FCS}.fcs_fit"
LOAD_G = f"{FCS}.fcs_load_g"
POLAR = f"{FCS}.fcs_polar"

SPLIT = 5e-8
MACROTIME = 1e-12


def _sibling(obj, name):
    return import_module(obj.__module__.rpartition(".")[0] + "." + name)


def _printed(call, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = call(*args, **kwargs)
    return result, buffer.getvalue()


def _drawn(call):
    plt.close("all")
    result = call()
    plt.close("all")
    return result


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _atimes(obj, channels=range(CHANNELS), filters=0, tcspc=True):
    data = _sibling(obj, "atimes_data").ATimesData()
    data.tcspc = tcspc
    weights = _rng()
    for channel, stream in photon_streams().items():
        if channel in channels:
            setattr(data, f"det{channel}", np.column_stack((stream, weights.uniform(0, 1, (len(stream), filters)))))
    return data


def _in_place(function, data, *args, **kwargs):
    function(data, *args, **kwargs)
    return data


def _marked(cls, name):
    instance = cls()
    setattr(instance, name, [1, 2])
    return instance


def _micro_data(n_channels=3, microtime=1e-10):
    rng = _rng()
    data = SimpleNamespace(macrotime=1e-6, microtime=microtime, microbintime=1e-9)
    for channel in range(n_channels):
        macro = np.sort(rng.integers(0, 200_000, size=60))
        micro = rng.integers(0, 250, size=60)
        setattr(data, f"det{channel}", np.column_stack((macro, micro)))
    return data


def _filter_functions(n_detectors=3, bins=30, n_filters=2):
    rng = _rng()
    filters = rng.uniform(0.1, 1.0, size=(n_detectors, bins, n_filters))
    return filters / filters.sum(axis=2, keepdims=True)


def _histograms(n_channels=3):
    bins = np.arange(256.0)
    data = SimpleNamespace()
    for channel in range(n_channels):
        counts = np.roll(decay_histogram(), 20 * channel)
        setattr(data, f"hist{channel}", np.column_stack((bins, counts)))
    return data


def _arrival_bins():
    rng = _rng()
    data = SimpleNamespace()
    for det in range(25):
        times = np.sort(rng.choice(500, size=20 + det, replace=False))
        setattr(data, f"det{det}", np.sort(np.append(times, times[:3])))
    data.det1 = np.array([], dtype=int)
    data.sum = np.sort(np.concatenate([getattr(data, f"det{det}") for det in range(25)]))
    return data


def _chunk_correlations(names=("a", "b"), n_chunks=3):
    rng = _rng()
    tau = np.logspace(-6, 0, 20)
    return SimpleNamespace(
        **{f"{name}_chunk{chunk}": np.column_stack((tau, rng.uniform(0.1, 1, 20))) for name in names for chunk in range(n_chunks)}
    )


def _correlations(obj, names, n_chunks=3, n_filters=1):
    rng = _rng()
    tau = np.logspace(-6, 0, 20)
    G = _sibling(obj, "fcs2corr").Correlations()
    for name in names:
        chunks = np.zeros((n_chunks, 20, 2 + n_filters))
        chunks[..., 0] = tau
        chunks[..., 1:] = rng.uniform(0.1, 1, (n_chunks, 20, 1 + n_filters))
        G.add_corr_chunks(name, chunks)
        G.list_of_g_out.append(name)
    return G


def _all_pair_averages():
    rng = _rng()
    tau = np.logspace(-6, 0, 12)
    return SimpleNamespace(
        **{
            f"det{a}x{b}_average": np.column_stack((tau, rng.uniform(0.1, 1, 12), rng.uniform(0, 0.1, 12)))
            for a in range(25)
            for b in range(25)
        }
    )




def _tau() -> np.ndarray:
    return correlation_curve()[0]


_FITS = {
    "fitfun": SimpleNamespace(
        model="fcs_analytical",
        args=(2.0, 1e-3, 5.0, 0.0),
        start=[1.0, 5e-4, 5.0, 0.01, 0.0, 1.0],
        fit=[1, 1, 0, 1, 0, 0],
        lower=[0, 1e-6, 1, -1, 0, 0],
        upper=[1e3, 1, 50, 1, 1e3, 3],
    ),
    "fitfun_an": SimpleNamespace(
        model="fcs_analytical",
        args=(2.0, 1e-3, 5.0, 0.0, 0.0, 0.0, 0.9),
        start=[1.0, 5e-4, 5.0, 0.0, 1.0],
        fit=[1, 1, 0, 0, 1],
        lower=[0, 1e-6, 1, -1, 0.3],
        upper=[1e3, 1, 50, 1, 1.5],
    ),
    "fitfun_2c": SimpleNamespace(
        model="fcs_2c_analytical",
        args=(2.0, 1e-3, 3e-2, 0.6, 1.0, 0.1, 5e-6, 5.0, 0.0, 0.0, 0.0),
        start=[1.0, 0.5, 10.0, 0.5, 1.0, 0.1, 5.0, 5.0, 0.0, 0.0, 0.0],
        fit=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        lower=[0, 1e-3, 1e-3, 0, 0.1, 0, 0.1, 1, -1, 0, 0],
        upper=[1e3, 1e3, 1e4, 1, 10, 0.9, 1e3, 50, 1, 1e3, 3],
    ),
    "fitfun_an_2c": SimpleNamespace(
        model="fcs_analytical_2c_anomalous",
        args=(2.0, 1e-3, 3e-2, 1.0, 0.8, 0.6, 0.1, 5e-6, 5.0, 0.0, 1.5),
        start=[1.0, 5e-4, 3e-2, 1.0, 0.8, 0.5, 1.5, 0.1, 5e-6, 5.0, 0.0],
        fit=[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        lower=[0, 1e-6, 1e-6, 0.3, 0.3, 0, 0.1, 0, 1e-7, 1, -1],
        upper=[1e3, 1, 1, 1.5, 1.5, 1, 10, 0.9, 1e-3, 50, 1],
    ),
    "fitfun_free_diffusion_2d": SimpleNamespace(
        model="fcs_2c_2d_analytical",
        args=(2.0, 1e-3, 3e-2, 0.6, 1.0, 0.1, 5e-6, 0.0, 0.0, 0.0),
        start=[1.0, 0.5, 10.0, 0.5, 1.0, 0.1, 5.0, 0.0, 0.0, 0.0],
        fit=[1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        lower=[0, 1e-3, 1e-3, 0, 0.1, 0, 0.1, -1, 0, -3],
        upper=[1e3, 1e3, 1e4, 1, 10, 0.9, 1e3, 1, 1e3, 3],
    ),
    "fitfun_circfcs": SimpleNamespace(
        model="fcs_circular_scanning",
        args=(2.0, 1e-3, 2.5e-7, 5.0, 1e-3, 2e-7, 0.0),
        start=[1.0, 0.5, 2.5e-7, 5.0, 2e-7, 1e-3, 0.01],
        fit=[1, 1, 0, 0, 0, 0, 1],
        lower=[0, 1e-3, 1e-9, 1, 0, 1e-6, -1],
        upper=[1e3, 1e3, 1e-5, 50, 1e-5, 1, 1],
    ),
    "fitfun_dualfocus": SimpleNamespace(
        model="fcs_dualfocus",
        args=(2.0, 1e-10, 2.5e-7, 5.0, 3e-7, 0.0, 0.0),
        start=[1.0, 50.0, 250.0, 5.0, 300.0, 0.0, 0.0, 0.0, 0.01],
        fit=[1, 1, 0, 0, 0, 0, 0, 0, 1],
        lower=[0, 1, 10, 1, 0, 0, -1e3, -1e3, -1],
        upper=[1e3, 1e5, 1e4, 50, 1e4, 1e4, 1e3, 1e3, 1],
    ),
    "fitfun_finitelength": SimpleNamespace(
        model="fcs_finitelength",
        args=(2.0, 1e-3, 5.0, 1e4, 10.0, 1e-6),
        start=[1.0, 0.5, 5.0, 10.0, 1e-3, 1e4],
        fit=[1, 1, 0, 0, 0, 0],
        lower=[0, 1e-3, 1.5, 1, 1e-6, 1],
        upper=[1e3, 1e3, 50, 100, 1, 1e7],
    ),
    "fitfun_nanosecond_fcs": SimpleNamespace(
        model="nanosecond_fcs_analytical",
        args=(0.5, 0.3, 2e-6, 0.2, 2e-5, 0.1, 1e-6, 0.1, 1e-5, 1e-3, 5.0),
        start=[0.3, 0.3, 2e-6, 0.2, 2e-5, 0.1, 1e-6, 0.1, 1e-5, 5e-4, 5.0],
        fit=[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        lower=[0, 0, 1e-9, 0, 1e-9, 0, 1e-9, 0, 1e-9, 1e-6, 1],
        upper=[10, 1, 1e-3, 1, 1e-2, 1, 1e-3, 1, 1e-2, 1, 50],
    ),
    "fitfun_uncoupled_reaction_diffusion": SimpleNamespace(
        model="uncoupled_reaction_diffusion",
        args=(0.5, 1e-3, 5.0, 0.6, 100.0),
        start=[0.3, 5e-4, 5.0, 0.5, 100.0],
        fit=[1, 1, 0, 1, 0],
        lower=[0, 1e-6, 1, 0, 0],
        upper=[10, 1, 50, 1, 1e4],
    ),
}


def _fit_data(function, kind):
    spec = _FITS[kind]
    tau = _tau()
    model = getattr(_sibling(function, "fcs_analytical"), spec.model)
    curve = model(tau, *spec.args) + _rng().normal(0, 1e-3, tau.shape)
    arrays = {key: np.asarray(getattr(spec, key), dtype=float) for key in ("start", "lower", "upper")}
    return tau, curve, SimpleNamespace(fit=np.asarray(spec.fit), **arrays)


def _weights() -> np.ndarray:
    return np.linspace(0.5, 2.0, 50)


def _residual(function, kind, weights=1):
    tau, curve, s = _fit_data(function, kind)
    return function(s.start[s.fit == 1], s.start[s.fit == 0], s.fit, tau, curve, weights)


def _fit(function, kind, fitfun, weights=1, global_param=None, plot=-1):
    tau, curve, s = _fit_data(function, kind)
    return function(
        curve,
        tau,
        fitfun(_sibling(function, "fcs_fit")),
        s.fit,
        s.start,
        s.lower,
        s.upper,
        plot,
        weights=weights,
        global_param=global_param,
    )


def _global_dualfocus(function):
    module = _sibling(function, "fcs_analytical")
    tau = _tau()
    noise = _rng().normal(0, 1e-3, (50, 2))
    curves = np.column_stack(
        [module.fcs_dualfocus(tau, 2.0, 1e-10, 2.5e-7, 5.0, rhox, 0.0, 0.0, 0.0, 0.0) for rhox in (3e-7, 4.5e-7)]
    )
    start = np.array(
        [[1.0, 1.0], [50.0, 50.0], [250.0, 250.0], [5.0, 5.0], [300.0, 450.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.01, 0.01]]
    )
    spec = _FITS["fitfun_dualfocus"]
    return tau, curves + noise, start, np.asarray(spec.fit), np.asarray(spec.lower, dtype=float), np.asarray(spec.upper, dtype=float)


GLOBAL_PARAM = [True, True, True, True, False, False, True, True, True]


def _fit_corr_inputs(function, curves=1):
    model = _sibling(function, "fcs_analytical").fcs_analytical
    tau = _tau()
    noise = _rng().normal(0, 1e-3, (50, curves))
    amplitudes = [2.0, 3.0][:curves]
    data = np.column_stack([model(tau, n, 1e-3, 5.0, 0.0) for n in amplitudes]) + noise
    return model, tau, data


def _fit_info_inputs():
    tau = _tau()
    rng = _rng()
    G = SimpleNamespace(
        **{f"{name}_averageX": np.column_stack((tau, rng.uniform(0.1, 1, 50))) for name in ("a", "b")}
    )
    return (
        G,
        ["a", "b"],
        np.arange(9, dtype=float),
        100 + np.arange(9, dtype=float),
        np.array([1, 1, 1, 0, 1, 0, 1, 0, 1]),
        np.linspace(1, 7, 7),
        [10.0, 20.0],
        [1.0, 2.0],
    )


def _spatial_correlations():
    rng = _rng()
    names = ("TB", "BT", "LR", "RL", "LT", "TL", "BR", "RB", "LB", "BL", "TR", "RT")
    return SimpleNamespace(
        dwellTime=1e-6,
        spatialCorr=rng.uniform(0.1, 1, (5, 5, 20)),
        crossCenterAv=rng.uniform(0.1, 1, (20, 6)),
        **{f"twofocus{name}_average": np.column_stack((np.arange(20.0), rng.uniform(0.1, 1, 20))) for name in names},
    )


def _directions(count):
    return _rng().uniform(0.5, 2.0, (40, count))


def _oscillating_curve():
    x = np.linspace(0, 1, 80)
    return np.exp(-3 * x) + 0.3 * np.sin(10 * x) + 1


def _fit_result(n_fitted):
    rng = _rng()
    return SimpleNamespace(x=rng.uniform(1, 2, n_fitted), fun=rng.normal(0, 1e-3, 50))


def _spad_input(mode):
    data = time_trace()
    if mode == "sum":
        return np.column_stack((data, data.sum(axis=1)))
    if mode.endswith("_5D"):
        return ism_series()[None]
    return data


CASES: list[Case] = []
SKIPPED: dict[str, str] = {}

CASES += [
    Case(f"{ADD_SUM}:add_sum_column", lambda f: f(time_trace()[:200].astype(np.int64))),
]

CASES += [
    Case(f"{ALIGN}:align_hist", lambda f: _in_place(f, _histograms(), n=9), "n=9"),
    Case(f"{ALIGN}:align_hist", lambda f: _in_place(f, _histograms()), "default window"),
]

CASES += [
    Case(f"{ATIMES_FILTERED}:ATimesData", lambda cls: _marked(cls, "det0")),
    Case(
        f"{ATIMES_FILTERED}:arrival_times_2_filtered_time_trace",
        lambda f: _printed(f, _micro_data(), _filter_functions(), 1e-9, 1e-2),
    ),
    Case(
        f"{ATIMES_FILTERED}:atimes_filtered",
        lambda f: _in_place(f, _micro_data(), _filter_functions()),
        "micro times in time units",
    ),
    Case(
        f"{ATIMES_FILTERED}:atimes_filtered",
        lambda f: _printed(_in_place, f, _micro_data(), _filter_functions(), micro_bin=True, verbose=True),
        "micro times in bins",
    ),
    Case(
        f"{ATIMES_FILTERED}:atimes_filtered",
        lambda f: _in_place(f, _micro_data(1), _filter_functions(1)[0]),
        "single detector 2D filter",
    ),
]

CASES += [
    Case(f"{ATIMES_TRACE}:atimes2timetrace", lambda f: _printed(f, _arrival_bins())),
    Case(f"{ATIMES_TRACE}:atimes2timetrace_bh", lambda f: f(photon_arrival_times().astype(np.float64), 500.0)),
]

CASES += [
    Case(f"{ATIMES_CORR}:Correlations", lambda cls: _marked(cls, "det0")),
    Case(
        f"{ATIMES_CORR}:atimes_2_corrs",
        lambda f: _printed(f, _atimes(f), [12, "sum3", "C6+7"], accuracy=8, taumax=1e-8, split=SPLIT),
        "unfiltered",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corrs",
        lambda f: _printed(f, _atimes(f, filters=1), [12, "sum3"], accuracy=8, taumax=1e-8, split=SPLIT),
        "filtered",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corrs",
        lambda f: _printed(
            lambda: f(_atimes(f), [12], taumax=3e-10, perform_coarsening=False, logtau=False, split=SPLIT)
        ),
        "linear lag times",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corrs",
        lambda f: _printed(
            lambda: f(
                _atimes(f, range(21)),
                ["av"],
                accuracy=4,
                taumax=1e-8,
                split=SPLIT,
            )
        ),
        "average of 21 channels",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corr",
        lambda f: f(photon_arrival_times(), photon_arrival_times(), [1], [1], MACROTIME, accuracy=16, taumax=1e-8),
        "autocorrelation",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corr",
        lambda f: f(photon_arrival_times(), photon_arrival_times()[1::2], [1], [1], MACROTIME, accuracy=16, taumax=1e-8),
        "crosscorrelation",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corr",
        lambda f: f(
            photon_arrival_times(),
            photon_arrival_times()[1::2],
            _rng().uniform(0.5, 1.5, 10_000),
            _rng().uniform(0.5, 1.5, 5_000),
            MACROTIME,
            accuracy=16,
            taumax=1e-8,
        ),
        "weighted",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corr",
        lambda f: f(
            photon_arrival_times(),
            photon_arrival_times()[1::2],
            [1],
            [1],
            MACROTIME,
            accuracy=8,
            taumax=3e-10,
            logtau=False,
            min_step=2,
        ),
        "linear lag times",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corr",
        lambda f: f(
            photon_arrival_times(),
            photon_arrival_times(),
            [1],
            [1],
            MACROTIME,
            accuracy=8,
            taumax=1e-8,
            perform_coarsening=False,
            min_step=4,
        ),
        "no coarsening",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_corr_single",
        lambda f: f(
            np.unique(photon_arrival_times() // 1000),
            np.unique(photon_arrival_times()[1::2] // 1000),
            _rng().uniform(0.5, 1.5, np.unique(photon_arrival_times() // 1000).size),
            _rng().uniform(0.5, 1.5, np.unique(photon_arrival_times()[1::2] // 1000).size),
            5,
            4,
        ),
    ),
    Case(f"{ATIMES_CORR}:atimes_2_all_crosscorr", lambda f: f(time_channel_array(), MACROTIME, accuracy=8), "default"),
    Case(
        f"{ATIMES_CORR}:atimes_2_all_crosscorr",
        lambda f: f(time_channel_array(), MACROTIME, accuracy=4, taumax=3e-11, logtau=False, perform_coarsening=False),
        "linear lag times",
    ),
    Case(
        f"{ATIMES_CORR}:atimes_2_all_crosscorr",
        lambda f: f(time_channel_array(), MACROTIME, accuracy=8, taumax=1e-9, min_step=2),
        "initial coarsening",
    ),
    Case(f"{ATIMES_CORR}:all_crosscorr_at_tau", lambda f: f(time_channel_array(), 7.0)),
    Case(f"{ATIMES_CORR}:time_coarsening", lambda f: f(photon_arrival_times(), _rng().uniform(0.5, 1.5, 10_000))),
    Case(f"{ATIMES_CORR}:time_coarsening_all_ch", lambda f: f(time_channel_array())),
    Case(f"{ATIMES_CORR}:generate_list_of_tau_values", lambda f: f(1e5, 8, 2, True), "logarithmic"),
    Case(f"{ATIMES_CORR}:generate_list_of_tau_values", lambda f: f(120.7, 8, 0, False), "linear"),
]


def _parallel(function, data, list_of_g, **kwargs):
    return function(data, list_of_g, accuracy=8, split=SPLIT, parallel_prefer="threads", print_info=False, **kwargs)


def _add_cross_all(function, averaging, names, **kwargs):
    G = _sibling(function, "fcs2corr").Correlations()
    chunks = _rng().uniform(0.1, 1, (2, 3, 12, 5, 5))
    function(G, chunks, np.logspace(-6, 0, 12), averaging, names, channels_order=[0, 1, 2, 3, 4], **kwargs)
    return G


CASES += [
    Case(f"{ATIMES_PARALLEL}:convert_crossall_to_list_of_g", lambda f: f(4), "all channels"),
    Case(f"{ATIMES_PARALLEL}:convert_crossall_to_list_of_g", lambda f: f(12, start_ch=9), "start channel"),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f), ["central", "sum3"], list_of_g_out=["c", "s3"]),
        "keywords",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f), [12, "x0612"], list_of_g_out=["single", "pair"]),
        "channel and pair",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f), ["C6+7", "C6+7xC11+12"], list_of_g_out=["sum", "cross sum"]),
        "custom sums",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f, filters=2), ["central", "x0612"], list_of_g_out=["c", "pair"]),
        "all filters",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f, filters=2), ["sum3"], list_of_g_out=["s3"], filters=[0, 2]),
        "selected filters",
    ),
    Case(f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel", lambda f: _parallel(f, _atimes(f, range(5)), ["crossAll"]), "crossAll"),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f, range(5)), ["crossAll", [[0, 1], [2, 3, 4]]]),
        "crossAll channel groups",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(
            f, _atimes(f, range(5)), ["crossAll"], averaging=["0x0+1x1", "0x1+1x0+2x3"], list_of_g_out=["auto", "cross"]
        ),
        "crossAll averaged",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f, range(5)), ["crossAll"], averaging=["0x0+1x1", "2x3"]),
        "crossAll averaged with default names",
    ),
    Case(
        f"{ATIMES_PARALLEL}:atimes_2_corrs_parallel",
        lambda f: _parallel(f, _atimes(f, range(5), filters=1, tcspc=False), ["crossAll"]),
        "crossAll without microtimes",
    ),
    Case(
        f"{ATIMES_PARALLEL}:parallel_g",
        lambda f: f(photon_streams()[12][:, 0], [1], MACROTIME, 0, SPLIT, 8, 1e-8, True, True, 0, 1),
        "unfiltered",
    ),
    Case(
        f"{ATIMES_PARALLEL}:parallel_g",
        lambda f: f(
            photon_streams()[12][:, 0], _rng().uniform(0.5, 1.5, 400), MACROTIME, 1, SPLIT, 8, 1e-8, True, True, 0, 1
        ),
        "filtered",
    ),
    Case(
        f"{ATIMES_PARALLEL}:parallel_gx",
        lambda f: f(
            photon_streams()[12][:, 0], [1], photon_streams()[7][:, 0], [1], MACROTIME, 0, SPLIT, 8, 1e-8, True, True, 0, 1
        ),
        "unfiltered",
    ),
    Case(
        f"{ATIMES_PARALLEL}:parallel_gx",
        lambda f: f(
            photon_streams()[12][:, 0],
            _rng().uniform(0.5, 1.5, 400),
            photon_streams()[7][:, 0],
            _rng().uniform(0.2, 1.0, 400),
            MACROTIME,
            1,
            SPLIT,
            8,
            1e-8,
            True,
            True,
            0,
            1,
        ),
        "filtered",
    ),
    Case(f"{ATIMES_PARALLEL}:extract_atimes_for_corr", lambda f: f(_atimes(f), 12, SPLIT, print_info=False), "channel"),
    Case(f"{ATIMES_PARALLEL}:extract_atimes_for_corr", lambda f: f(_atimes(f), "x0612", SPLIT, print_info=False), "pair"),
    Case(f"{ATIMES_PARALLEL}:extract_atimes_for_corr", lambda f: f(_atimes(f), "sum3", SPLIT, print_info=False), "keyword"),
    Case(f"{ATIMES_PARALLEL}:extract_atimes_for_corr", lambda f: f(_atimes(f), "C6+7", SPLIT, print_info=False), "custom sum"),
    Case(
        f"{ATIMES_PARALLEL}:extract_atimes_for_corr",
        lambda f: f(_atimes(f), "C6+7xC11+12", SPLIT, print_info=False),
        "custom sum pair",
    ),
    Case(
        f"{ATIMES_PARALLEL}:calc_average_correlation",
        lambda f: _in_place(f, _chunk_correlations()),
    ),
    Case(
        f"{ATIMES_PARALLEL}:add_cross_all_to_G",
        lambda f: _add_cross_all(f, ["0x0+1x1", "2x3"], ["auto", "cross"]),
        "averaging",
    ),
    Case(
        f"{ATIMES_PARALLEL}:add_cross_all_to_G",
        lambda f: _add_cross_all(f, None, None, c0_all=[0, 1], c1_all=[3, 4]),
        "pairs without averaging",
    ),
]


CASES += [
    Case(f"{ATIMES_HIST}:atimes_data_2_hist", lambda f: _in_place(f, _atimes(f)), "automatic bin time"),
    Case(f"{ATIMES_HIST}:atimes_data_2_hist", lambda f: _in_place(f, _atimes(f), bintime=250.0), "bin time"),
    Case(
        f"{ATIMES_HIST}:atimes_data_2_hist",
        lambda f: _in_place(f, _atimes(f), bintime=250.0, ttm_correction=True, laser_freq=40e6),
        "TTM correction",
    ),
]


def _atimes_data_without_microtimes(cls):
    data = _atimes(cls, filters=1, tcspc=False)
    return data.data, data.to_time_channel_array(), data


def _atimes_data_methods(cls):
    data = _atimes(cls, filters=1)
    plain = _atimes(cls)
    plain.calc_hist(bintime=250.0)
    plain.align_hist(n=5)
    corrected = _atimes(cls)
    corrected.calc_hist(bintime=250.0, ttm_correction=True, laser_freq=40e6)
    automatic = _atimes(cls)
    automatic.calc_hist()
    return (
        data.data,
        data.all_channels,
        data.num_channels,
        data.num_filters,
        data.to_time_channel_array(),
        data.to_time_channel_array(1, [[0, 1], [12, 13, 14]]),
        data.duration(),
        data.duration(subtract_start_time=True, return_period=True),
        data.get_channels(),
        plain.get_channels("hist"),
        plain,
        corrected,
        automatic,
        data,
    )


CASES += [
    Case(f"{ATIMES_DATA}:ATimesData", _atimes_data_methods, "with microtimes"),
    Case(f"{ATIMES_DATA}:ATimesData", _atimes_data_without_microtimes, "without microtimes"),
    Case(f"{ATIMES_DATA}:atimes_data_2_duration", lambda f: f(_atimes(f)), "default"),
    Case(f"{ATIMES_DATA}:atimes_data_2_duration", lambda f: f(_atimes(f), 1e-9, subtract_start_time=True), "subtract start"),
    Case(f"{ATIMES_DATA}:atimes_data_2_duration", lambda f: f(_atimes(f), return_period=True), "period"),
    Case(f"{ATIMES_DATA}:atimes_data_2_channels", lambda f: f(_atimes(f, range(3, 9)))),
    Case(f"{ATIMES_DATA}:atimes_data_attr_2_ch", lambda f: f("det31")),
]

SKIPPED |= {
    f"{ATIMES_TRACE}:atimes2data": SKIP_FILE_IO,
    f"{ATIMES_PARALLEL}:atimes_file_2_corr": SKIP_FILE_IO,
    f"{ATIMES_DATA}:atimes_file_2_nch": SKIP_FILE_IO,
    f"{ATIMES_DATA}:atimes_h5_to_hdf5": SKIP_FILE_IO,
    f"{ATIMES_DATA}:atimes_raw_2_h5": SKIP_FILE_IO,
    f"{ATIMES_DATA}:atimes_raw_2_h5all": SKIP_FILE_IO,
    f"{ATIMES_DATA}:h5_atimes_2_data": SKIP_FILE_IO,
    f"{ATIMES_DATA}:load_atimes_data": SKIP_FILE_IO,
    f"{ATIMES_DATA}:load_atimes_data_newprot": SKIP_FILE_IO,
    f"{ATIMES_DATA}:load_atimes_data_pandas": SKIP_FILE_IO,
    f"{ATIMES_DATA}:write_atimes_data": SKIP_FILE_IO,
    f"{BH}:bh_atimes2csv": SKIP_FILE_IO,
    f"{BIN_ATIMES}:binfile2arrivaltimes": SKIP_FILE_IO,
    f"{BIN_DATA}:binfile2data": SKIP_FILE_IO,
    f"{BIN_DATA}:binfile2dataAle": SKIP_FILE_IO,
    f"{CORR_CSV}:corr2csv": SKIP_FILE_IO,
    f"{FILE}:load_file": SKIP_FILE_IO,
    f"{LOAD_G}:load_g": SKIP_FILE_IO,
}

CASES += [
    Case(f"{BIN_ATIMES}:ArrivalTimesObject", lambda cls: _marked(cls, "det0")),
    Case(f"{BIN_DATA}:ArrivalTimes", lambda cls: _marked(cls, "pixel0")),
    Case(
        f"{BIN_DATA}:getaddress",
        lambda f: [f([byte, 0, 0, 0]) for byte in (0b00000000, 0b00100000, 0b01100000, 0b11111111, 0b10101010)],
    ),
    Case(
        f"{BIN_DATA}:gettime",
        lambda f: [f([0, 0, high, low]) for high, low in ((0, 0), (0b00001111, 255), (0b00000101, 0b10110001), (0b11110010, 7))],
    ),
    Case(
        f"{BIN_DATA}:getchannel",
        lambda f: [f([high, low, 0, 0]) for high, low in ((0, 1), (0, 2), (1, 0), (2, 0), (0, 0), (1, 1), (3, 3))],
    ),
]

CASES += [
    Case(f"{CORR_KW}:get_corr_input", lambda f, c=corr: [f(c, det) for det in ("square 5x5", "PDA-23", "airyscan 32")], corr)
    for corr in (
        "all autocorrelations",
        "spot-variation fcs",
        "spot-variation fcs - extended",
        "pair-correlation fcs",
        "all cross-correlations",
        "crossall-ext",
        "unknown",
    )
] + [
    Case(f"{CORR_KW}:get_corr_input", lambda f, c=corr, d=det: f(c, d), f"{corr}, {det}")
    for corr, det in (("fcs", "nsparc"), ("svfcs", "luminosa"), ("xcorr", "airyscan"), ("svfcs-ext", "5x5"), ("fcs", "pda23"))
]

CASES += [
    Case(f"{PHASOR}:data2phasor", lambda f: _in_place(f, _histograms())),
]

CASES += [
    Case(f"{DETECTORS}:detector_element_coordinates", lambda f, d=detector: (f(d), f(d, element=3)), detector)
    for detector in ("airyscan", "Luminosa", "pda-23", "nsparc", "5x5", "gi", "7x7", "prism", "unknown")
] + [
    Case(f"{DETECTORS}:distance_between_elements", lambda f, d=detector: f(d, 3, 17), detector)
    for detector in ("airyscan", "luminosa", "nsparc", "prism")
]

CASES += [
    Case(f"{DISTANCE}:distance2detelements", lambda f: f(3, 18)),
    Case(f"{DISTANCE}:spad_coord_from_det_numb", lambda f: f(13)),
    Case(f"{DISTANCE}:spad_shift_vector_crosscorr", lambda f: f(), "default"),
    Case(f"{DISTANCE}:spad_shift_vector_crosscorr", lambda f: f([1, -1], 3), "3x3 region"),
]

SPAD_MODES = {
    "number": 3,
    "negative number": -5,
    "sum_all": "sum_all",
    "custom sum": "C3+7+8+12",
    "custom single channel": "C5",
    "all but listed": "N1+3+12",
    "keyword single channel": "central",
    "keyword sum": "sum3",
    "keyword all channels": "sum5",
    "keyword chess": "chess0",
    "keyword airyscan": "airyring1",
}

SWITCH_MODES = (
    "central",
    "center",
    "sum3",
    "sum3_5D",
    "allbuthot",
    "allbuthot_5D",
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
)

CASES += [
    Case(f"{SPAD_DATA}:extract_spad_data", lambda f, m=mode: _printed(f, _spad_input("data"), m), label)
    for label, mode in SPAD_MODES.items()
] + [
    Case(f"{SPAD_DATA}:switch_mode_string", lambda f, m=mode: _printed(f, _spad_input(m), m), mode)
    for mode in SWITCH_MODES
]

CASES += [
    Case(f"{SPAD_DATA}:{name}", lambda f, n=name: _printed(f, _spad_input(n if n.endswith("_5D") else "data")))
    for name in (
        "central",
        "sum3",
        "sum3_5D",
        "allbuthot",
        "allbuthot_5D",
        "sum5",
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
    )
] + [
    Case(f"{SPAD_DATA}:sum_all", lambda f: f(_spad_input("sum"))),
]

CASES += [
    Case(f"{SPAD_STREAMS}:extract_spad_photon_streams", lambda f: f(_atimes(f), "C3+4+7+12"), "custom sum"),
    Case(f"{SPAD_STREAMS}:extract_spad_photon_streams", lambda f: f(_atimes(f), 7), "single channel"),
    Case(f"{SPAD_STREAMS}:extract_spad_photon_streams", lambda f: f(_atimes(f), "sum3"), "keyword"),
    Case(f"{SPAD_STREAMS}:extract_spad_photon_streams", lambda f: f(_atimes(f, filters=2), "sum3"), "with filter weights"),
]

CASES += [
    Case(f"{FCS_ATIMES}:ATimesData", lambda cls: _marked(cls, "det0")),
]

CASES += [
    Case(f"{DIFFTIME}:g2difftime", lambda f: f(np.column_stack(correlation_curve())), "default smoothing"),
    Case(f"{DIFFTIME}:g2difftime", lambda f: f(np.column_stack(correlation_curve()), smoothing=3), "smoothing 3"),
]

CASES += [
    Case(f"{LOAD_G}:strip_g_fname", lambda f: _printed(f, "folder/FCS_beads_sum3.csv", "FCS_beads_")),
    Case(f"{LOAD_G}:strip_g_fname", lambda f: _printed(f, "FCS_beads_det12x7.csv", "FCS_", False), "quiet"),
]

CASES += [
    Case(f"{POLAR}:sig", lambda f: f(np.linspace(-3, 3, 30), 0.5, 2.0)),
    Case(f"{POLAR}:g2polar", lambda f: f(_directions(4), Nr=48), "4 directions"),
    Case(f"{POLAR}:g2polar", lambda f: f(_directions(6), Nr=48), "6 directions"),
    Case(f"{POLAR}:g2flow", lambda f: f(_directions(4), Nr=48), "square"),
    Case(f"{POLAR}:g2flow", lambda f: f(_directions(4), Nr=48, detector="airy"), "airy"),
    Case(f"{POLAR}:g2flow", lambda f: f(_directions(6), Nr=48, detector="airy6"), "airy6"),
    Case(f"{POLAR}:g2polar_old", lambda f: f(_oscillating_curve()), "default smoothing"),
    Case(f"{POLAR}:g2polar_old", lambda f: f(_oscillating_curve(), smoothing=5), "smoothing 5"),
]

CASES += [
    Case(f"{ANALYTICAL}:fcs_analytical", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 0.01), "plain"),
    Case(f"{ANALYTICAL}:fcs_analytical", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 0.01, 1e-7, -1.05, 0.9), "afterpulsing and anomalous diffusion"),
    Case(f"{ANALYTICAL}:fcs_analytical", lambda f: f(np.float64(1e-3), 2.0, 1e-3, 5.0, 0.01), "scalar lag time"),
    Case(
        f"{ANALYTICAL}:fcs_2c_analytical",
        lambda f: f(_tau(), 2.0, 1e-3, 3e-2, 0.6, alpha=1.5, T=0.1, tautrip=5e-6, SF=4.0, offset=0.01, A=1e-8, B=-1.0),
    ),
    Case(
        f"{ANALYTICAL}:fcs_analytical_2c_anomalous",
        lambda f: f(_tau(), 2.0, 1e-3, 3e-2, 0.9, 0.8, 0.6, 0.1, 5e-6, 4.0, 0.01, 1.5),
    ),
    Case(
        f"{ANALYTICAL}:fcs_analytical_2c_anomalous_c",
        lambda f: f(_tau(), 0.5, 15.0, 2.0, 0.9, 0.8, 0.6, 0.1, 5e-6, 0.25, 4.0, 0.01, 1.5),
    ),
    Case(
        f"{ANALYTICAL}:fcs_dualfocus",
        lambda f: f(_tau(), 2.0, 1e-10, 2.5e-7, 5.0, 3e-7, 1.5e-7, 0.01, vx=2e-5, vy=-1e-5),
    ),
    Case(
        f"{ANALYTICAL}:fcs_dualfocus_c",
        lambda f: f(_tau(), 0.5, 1e-10, 2.5e-7, 5.0, 3e-7, 1.5e-7, 0.01, vx=2e-5, vy=-1e-5),
    ),
    Case(
        f"{ANALYTICAL}:fcs_circular_scanning",
        lambda f: f(_tau(), 2.0, 1e-3, 2.5e-7, 5.0, 1e-3, 2e-7, 0.01, vx=2e-5, vy=-1e-5),
    ),
    Case(f"{ANALYTICAL}:fcs_circular_scanning_c", lambda f: f(_tau(), 0.5, 1e-10, 2.5e-7, 5.0, 1e-3, 2e-7, 0.01)),
    Case(
        f"{ANALYTICAL}:fcs_2c_2d_analytical",
        lambda f: f(_tau(), 2.0, 1e-3, 3e-2, 0.6, alpha=1.5, T=0.1, tautrip=5e-6, offset=0.01, A=1e-8, B=-1.0),
    ),
    Case(
        f"{ANALYTICAL}:nanosecond_fcs_analytical",
        lambda f: f(_tau(), 0.5, 0.3, 2e-6, 0.2, 2e-5, 0.1, 1e-6, 0.15, 1e-5, 1e-3, 5.0),
    ),
    Case(f"{ANALYTICAL}:uncoupled_reaction_diffusion", lambda f: f(_tau(), 0.5, 1e-3, 5.0, 0.6, 100.0)),
    Case(f"{ANALYTICAL}:fcs_finitelength", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 1e4, 10.0, 1e-6)),
    Case(f"{ANALYTICAL}:fcs_1c_2dgl", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 0.01)),
    Case(f"{ANALYTICAL}:compute_gamma", lambda f: f(_tau(), 2.0, 1e-3), "default parameters"),
    Case(f"{ANALYTICAL}:compute_gamma", lambda f: f(_tau(), 2.0, 1e-3, 20.0, 1e-6, 0.8, 5.0, 1e4), "given parameters"),
]

def _single_dualfocus_residual(function):
    tau, curve, s = _fit_data(function, "fitfun_dualfocus")
    return function(s.start[s.fit == 1], s.start[s.fit == 0], s.fit, None, tau, curve, _weights())


def _global_residual(function):
    tau, curves, _, fit, _, _ = _global_dualfocus(function)
    fitted = np.array([1.0, 50.0, 0.01])
    fixed = np.array([250.0, 5.0, 300.0, 450.0, 0.0, 0.0, 0.0, 0.0])
    return function(fitted, fixed, fit, GLOBAL_PARAM, tau, curves, _weights()[:, None] * np.ones(2))


CASES += [
    Case(f"{FIT}:compute_gamma", lambda f: f(_tau(), 2.0, 1e-3), "default parameters"),
    Case(f"{FIT}:compute_gamma", lambda f: f(_tau(), 2.0, 1e-3, 20.0, 1e-6, 0.8, 5.0, 1e4), "given parameters"),
] + [
    Case(f"{FIT}:{kind}", lambda f, k=kind: _residual(f, k, _weights()), "weights")
    for kind in _FITS
    if kind != "fitfun_dualfocus"
] + [
    Case(f"{FIT}:fitfun", lambda f: _residual(f, "fitfun"), "no weights"),
    Case(f"{FIT}:fitfun_dualfocus", _single_dualfocus_residual, "single curve"),
    Case(f"{FIT}:fitfun_dualfocus", _global_residual, "global fit of two curves"),
]


def _fit_global(function, weights=1, plot=-1):
    tau, curves, start, fit, lower, upper = _global_dualfocus(function)
    return function(
        curves,
        tau,
        _sibling(function, "fcs_fit").fcs_fit_dualfocus,
        fit,
        start,
        lower,
        upper,
        plot,
        weights=weights,
        global_param=GLOBAL_PARAM,
    )


def _fit_dualfocus_direct(function, global_fit):
    if global_fit:
        tau, curves, start, fit, lower, upper = _global_dualfocus(function)
        return function(curves, tau, fit, start, global_param=GLOBAL_PARAM, lBounds=lower, uBounds=upper)
    tau, curve, s = _fit_data(function, "fitfun_dualfocus")
    return function(curve, tau, s.fit, s.start, lBounds=s.lower, uBounds=s.upper)


def _fit_corr(function, curves, **kwargs):
    model, tau, data = _fit_corr_inputs(function, curves)
    start = np.array([1.0, 5e-4, 5.0, 0.01])
    if curves > 1:
        start = np.column_stack([start * (1 + 0.1 * i) for i in range(curves)])
        data_arg = data
    else:
        data_arg = data[:, 0]
    return function(
        data_arg,
        tau,
        np.array([1, 1, 0, 1] if curves > 1 else [1, 1, 0, 0]),
        start,
        lower_bounds=[0, 1e-6, 1, -1],
        upper_bounds=[1e3, 1, 50, 1],
        fitmodel=model,
        **kwargs,
    )


def _fitfun_general(function, curves):
    model, tau, data = _fit_corr_inputs(function, curves)
    fit = np.array([1, 1, 0, 1])
    if curves == 1:
        return function(
            np.array([1.0, 5e-4, 0.01]), np.array([5.0]), fit, tau, data[:, 0], model, [False] * 4, _weights()
        )
    return function(
        np.array([1.0, 1.1, 0.5, 0.01, 0.02]),
        np.array([5.0]),
        fit,
        tau,
        data,
        model,
        [False, True, True, False],
        np.ones((50, 2)),
        np.array([1.0, 1e-3, 1.0, 1.0]),
    )


def _plot_fit(function):
    tau, curve, s = _fit_data(function, "fitfun_an")
    param = s.start.copy()
    fit = np.ones(5, dtype=int)
    result = _fit_result(5)
    return _drawn(lambda: (function(tau, curve, param, fit, result, "central"), param))


def _fcs_fit_plotted(function):
    tau, curve, s = _fit_data(function, "fitfun_an")
    fit = np.ones(5, dtype=int)
    return _drawn(
        lambda: function(curve, tau, "fitfun_an", fit, s.start, s.lower, s.upper, "central")
    )


def _mem_fit(function, fitfun):
    tau, curve, s = _fit_data(function, "fitfun")
    param = [20, 30, 1e-5, 2e-4, 1.0, 20.0, 3.0]
    return function(curve, tau, fitfun(_sibling(function, "mem_fit")), s.fit, param, s.lower, s.upper, -1)


CASES += [
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun_2c", lambda m: "fitfun_2c"), "fitfun_2c by name"),
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun_an", lambda m: "fitfun_an", _weights()), "fitfun_an by name, weights"),
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun_dualfocus", lambda m: "fitfun_dualfocus"), "fitfun_dualfocus by name"),
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun_circfcs", lambda m: "fitfun_circfcs"), "fitfun_circfcs by name"),
    Case(
        f"{FIT}:fcs_fit",
        lambda f: _fit(f, "fitfun_free_diffusion_2d", lambda m: "fitfun_free_diffusion_2d"),
        "fitfun_free_diffusion_2d by name",
    ),
    Case(f"{FIT}:fcs_fit", lambda f: _mem_fit(f, lambda m: "mem_fit_free_diffusion"), "mem_fit_free_diffusion by name"),
    Case(f"{FIT}:fcs_fit", lambda f: _mem_fit(f, lambda m: m.mem_fit_free_diffusion), "mem_fit_free_diffusion function"),
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun", lambda m: m.fitfun, _weights()), "function object, weights"),
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun_an_2c", lambda m: m.fitfun_an_2c), "anomalous two components function"),
    Case(
        f"{FIT}:fcs_fit",
        lambda f: _fit(f, "fitfun_finitelength", lambda m: m.fitfun_finitelength),
        "finite length function",
    ),
    Case(
        f"{FIT}:fcs_fit",
        lambda f: _fit(f, "fitfun_nanosecond_fcs", lambda m: m.fitfun_nanosecond_fcs),
        "nanosecond function",
    ),
    Case(
        f"{FIT}:fcs_fit",
        lambda f: _fit(f, "fitfun_uncoupled_reaction_diffusion", lambda m: m.fitfun_uncoupled_reaction_diffusion),
        "reaction diffusion function",
    ),
    Case(
        f"{FIT}:fcs_fit",
        lambda f: _fit(f, "fitfun_dualfocus", lambda m: m.fitfun_dualfocus),
        "fitfun_dualfocus function",
    ),
    Case(f"{FIT}:fcs_fit", lambda f: _fit(f, "fitfun_dualfocus", lambda m: m.fcs_fit_dualfocus), "fcs_fit_dualfocus function"),
    Case(f"{FIT}:fcs_fit", _fit_global, "global dual focus"),
    Case(f"{FIT}:fcs_fit", _fcs_fit_plotted, "plot"),
    Case(f"{FIT}:fcs_fit_dualfocus", lambda f: _fit_dualfocus_direct(f, False), "single curve"),
    Case(f"{FIT}:fcs_fit_dualfocus", lambda f: _fit_dualfocus_direct(f, True), "global fit"),
    Case(f"{FIT}:fit_corr", lambda f: _fit_corr(f, 1), "single curve"),
    Case(f"{FIT}:fit_corr", lambda f: _fit_corr(f, 1, x_scale="jac"), "single curve with least_squares options"),
    Case(
        f"{FIT}:fit_corr",
        lambda f: _fit_corr(f, 2, global_param=[False, True, True, False], weights=np.ones((50, 2))),
        "global fit",
    ),
    Case(f"{FIT}:fitfun_general", lambda f: _fitfun_general(f, 1), "single curve"),
    Case(f"{FIT}:fitfun_general", lambda f: _fitfun_general(f, 2), "global fit with scaled parameters"),
    Case(f"{FIT}:plot_fit", _plot_fit),
    Case(
        f"{FIT}:make_fit_info_global_fit",
        lambda f: f(*_fit_info_inputs()),
    ),
    Case(
        f"{FIT}:read_global_fit_result",
        lambda f: f(
            SimpleNamespace(x=np.arange(1.0, 10.0)),
            ["a", "b"],
            np.array([1, 1, 1, 0, 1, 0, 1, 0, 1]),
            np.linspace(1, 7, 7),
            [10.0, 20.0],
            [1.0, 2.0],
        ),
    ),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_spatial_correlations(), ["spatialCorr"], 2, 3), "spatialCorr"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_spatial_correlations(), ["crossCenterAv"], 2), "crossCenterAv"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_spatial_correlations(), ["crossCenterAvN"], 1, 3), "crossCenterAvN"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_spatial_correlations(), ["twofocus5"], 2), "twofocus5"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_spatial_correlations(), ["twofocus5Av"], 2), "twofocus5Av"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_spatial_correlations(), "", 0, 3), "all fields"),
    Case(
        f"{FIT}:chi_2_fcs_fit",
        lambda f: f(_tau(), correlation_curve()[1], np.array([1, 1, 0, 1]), _fit_result(4)),
    ),
    Case(f"{FIT}:stddev_2_weights", lambda f: f(np.append(_rng().uniform(0.5, 2, 20), 0.0))),
    Case(f"{FIT}:stddev_2_weights", lambda f: f(_rng().uniform(0.5, 2, 20), clipmin=0.2, clipmax=0.8), "clipped"),
]


def _fcs2corr(function, list_of_g, data=None, **kwargs):
    trace = time_trace() if data is None else data
    return _printed(function, trace, 1.0, list_of_g, accuracy=8, print_info=False, **kwargs)


def _correlations_methods(cls):
    G = _correlations(cls, ["a", "b"], n_filters=1)
    unfiltered = _correlations(cls, ["a", "b"], n_filters=0)
    G.good_chunks = [0, 2]
    return (
        G.get_av_corrs(["a", "b"]),
        G.get_av_corrs(["b", "a"], good_chunks_only=False),
        unfiltered.get_av_corrs(["a"], good_chunks_only=False),
        G.get_corrs("b"),
        G.num_chunks,
        G.n_filters,
        unfiltered.n_filters,
        G,
    )


def _chunks_methods(cls):
    chunks = cls(_rng().uniform(0.1, 1, (4, 20, 3)))
    return chunks.num_chunks, chunks.average(), chunks.average([0, 3]), chunks


def _crosscenter(function, return_field, return_obj):
    names = [f"det12x{j}" for j in range(25)]
    G = _correlations(function, names, n_chunks=3)
    G.good_chunks = [0, 2]
    return function(G, return_field, return_obj), G


def _spatialcorrav(function, return_type):
    G = _all_pair_averages()
    return function(G, 5, return_type), G


CASES += [
    Case(f"{FCS_CORR}:Correlations", _correlations_methods),
    Case(f"{FCS_CORR}:Correlations_chunks", _chunks_methods),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["central", "sum3", "sum5"]), "keywords"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["central"], list_of_g_out=["c"]), "keyword with output name"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, [12, 3]), "channels"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["x0112", "x1201"]), "channel pairs"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["C6+7+8", "C6+7xC11+12"]), "custom sums"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["A1+2+3"]), "average of autocorrelations"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["crossCenter"]), "crossCenter"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["singleElement"]), "singleElement"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["2MPD"]), "2MPD"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["av"]), "av"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["autoSpatial"], time_trace()[:12]), "autoSpatial"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["crossAll"], time_trace()[:2000]), "crossAll"),
    Case(
        f"{FCS_CORR}:fcs2corr",
        lambda f: _fcs2corr(f, ["crossAll"], algorithm="sparse_matrices"),
        "crossAll sparse matrices",
    ),
    Case(
        f"{FCS_CORR}:fcs2corr",
        lambda f: _fcs2corr(f, ["crossAll"], algorithm="sparse_matrices", averaging="default"),
        "crossAll sparse matrices default averaging",
    ),
    Case(
        f"{FCS_CORR}:fcs2corr",
        lambda f: _fcs2corr(
            f, ["crossAll"], algorithm="sparse_matrices", averaging=["0x0+1x1", "0x1"], list_of_g_out=["auto", "cross"]
        ),
        "crossAll sparse matrices custom averaging",
    ),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["central"], algorithm="wiener-khinchin"), "wiener-khinchin"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["central"], algorithm="tt2corr"), "arrival times algorithm"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, ["central"], binsize=2, algorithm="pch"), "pch"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, [], time_trace()[:, 12]), "single detector vector"),
    Case(f"{FCS_CORR}:fcs2corr", lambda f: _fcs2corr(f, [], time_trace()[:, 12], list_of_g_out=["only"]), "vector with output name"),
    Case(
        f"{FCS_CORR}:correlate",
        lambda f: f(time_trace()[:, 12], time_trace()[:, 7], m=8, deltat=1e-6),
        "multipletau",
    ),
    Case(
        f"{FCS_CORR}:correlate",
        lambda f: f(time_trace()[:, 12], time_trace()[:, 7], m=8, deltat=1e-6, normalize=False),
        "multipletau not normalised",
    ),
    Case(
        f"{FCS_CORR}:correlate",
        lambda f: f(time_trace()[:, 12], time_trace()[:, 7], m=8, deltat=1e-6, algorithm="wiener-khinchin"),
        "wiener-khinchin",
    ),
    Case(
        f"{FCS_CORR}:correlate",
        lambda f: f(time_trace()[:, 12], time_trace()[:, 7], m=8, binsize=2, algorithm="pch"),
        "pch",
    ),
    Case(
        f"{FCS_CORR}:correlate",
        lambda f: f(time_trace()[:, 12], time_trace()[:, 7], m=8, deltat=1e-6, algorithm="tt2corr"),
        "arrival times algorithm",
    ),
    Case(f"{FCS_CORR}:correlate_parallel", lambda f: f(time_trace()[:, :3].T, m=8, deltat=1e-6), "average compression"),
    Case(f"{FCS_CORR}:correlate_parallel", lambda f: f(time_trace()[:, :3].T, m=8, compress="first", ret_sum=True), "first compression"),
    Case(f"{FCS_CORR}:correlate_parallel", lambda f: f(time_trace()[:, :3].T, m=8, compress="second"), "second compression"),
    Case(f"{FCS_CORR}:fcs2corrsplit", lambda f: _printed(f, time_trace(), 1.0, ["central", "sum3"], 8, 1), "no split"),
    Case(
        f"{FCS_CORR}:fcs_sparse",
        lambda f: f(time_trace()[:2000], 1e-6, m=8),
        "normalised",
    ),
    Case(f"{FCS_CORR}:fcs_sparse", lambda f: f(time_trace()[:2000], 1e-6, m=9, normalize=False), "not normalised, odd accuracy"),
    Case(f"{FCS_CORR}:fcs_spatialcorrav", lambda f: _spatialcorrav(f, "3Darray"), "3D array"),
    Case(f"{FCS_CORR}:fcs_spatialcorrav", lambda f: _spatialcorrav(f, "fields"), "fields"),
    Case(f"{FCS_CORR}:fcs_av_chunks", lambda f: _in_place(f, _chunk_correlations(["central", "sum3"]), [0, 2]), "named chunks"),
    Case(f"{FCS_CORR}:fcs_crosscenter_av", lambda f: _crosscenter(f, "_averageX", True), "good chunks, object"),
    Case(f"{FCS_CORR}:fcs_crosscenter_av", lambda f: _crosscenter(f, "_average", False), "all chunks, array"),
    Case(f"{FCS_CORR}:corrs2average_for_stics", lambda f: f(), "default"),
    Case(f"{FCS_CORR}:corrs2average_for_stics", lambda f: f(3), "3x3 detector"),
    Case(
        f"{FCS_CORR}:list_of_g_out_v1p1",
        lambda f: f(_chunk_correlations(["central", "sum3"])),
    ),
]

SKIPPED |= {
    f"{FCS_CORR}:fcs_bin_to_csv_all": SKIP_FILE_IO,
    f"{FCS_CORR}:fcs_load_and_corr_split": SKIP_FILE_IO,
    f"{FCS_CORR}:fcs_sparse_matrices": SKIP_FILE_IO,
    f"{FCS_CORR}:plot_cross_center_scheme": SKIP_PLOT_ONLY,
    f"{FCS_CORR}:plot_fcs_correlations": SKIP_PLOT_ONLY,
    f"{FCS_CORR}:plot_g_surf": SKIP_PLOT_ONLY,
}
