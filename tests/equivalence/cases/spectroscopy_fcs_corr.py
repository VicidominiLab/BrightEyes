import importlib
from types import SimpleNamespace

import numpy as np

from .._inputs import (
    correlation_curve,
    decay_histogram,
    photon_arrival_times,
    time_trace,
)
from .._registry import SKIP_FILE_IO, SKIP_PLOT_ONLY, Case

A2C = "brighteyes_ffs.fcs.atimes2corr"
A2CP = "brighteyes_ffs.fcs.atimes2corrparallel"
F2C = "brighteyes_ffs.fcs.fcs2corr"
T2C = "brighteyes_ffs.fcs.timetrace2corr"
AN = "brighteyes_ffs.fcs.fcs_analytical"
FIT = "brighteyes_ffs.fcs.fcs_fit"
MEM = "brighteyes_ffs.fcs.mem_fit"
FAP = "brighteyes_ffs.fcs.filter_ap"
FG = "brighteyes_ffs.fcs.filter_g"
SG = "brighteyes_ffs.fcs.select_g"
POL = "brighteyes_ffs.fcs.fcs_polar"
DT = "brighteyes_ffs.fcs.fcs2difftime"
IMSD = "brighteyes_ffs.fcs.imsd"
TF = "brighteyes_ffs.fcs.two_focus_fcs"
W0 = "brighteyes_ffs.fcs.w0_from_fingerprint"
DIST = "brighteyes_ffs.fcs.distance2detelements"
DET = "brighteyes_ffs.fcs.detectors"
SHIFTS = "brighteyes_ffs.fcs.spad_shifts"
GDE = "brighteyes_ffs.fcs.get_det_elem_from_array"


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


def _atimes(f, macrotime: float = 1e-5, weights: bool = False):
    """Arrival-time object with 25 channels spanning about 10 s."""
    obj = _sib(f, "atimes_data", "ATimesData")()
    obj.macrotime = macrotime
    rng = np.random.default_rng(1)
    for name, array in photon_arrival_times().items():
        if weights:
            array = np.column_stack((array, rng.random(len(array))))
        setattr(obj, name, array)
    return obj


def _macro(channel: int) -> np.ndarray:
    return photon_arrival_times()[f"det{channel}"][:, 0]


def _channel_array(f, n_groups: int = 4) -> np.ndarray:
    """Time x (1 + n_groups) array of photon counts as used by the cross-correlation code."""
    groups = [[i] for i in range(n_groups)]
    return _atimes(f).to_time_channel_array(0, select_channels=groups)


def _trace(n: int = 1000) -> np.ndarray:
    return time_trace()[:n]


def _tau() -> np.ndarray:
    return correlation_curve()[0]


def _noisy(y: np.ndarray, seed: int = 3) -> np.ndarray:
    return y * (1 + 0.01 * np.random.default_rng(seed).standard_normal(y.shape))


def _chunk_array(n_chunks=6, n_tau=20, n_filters=0, seed=5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    tau = np.logspace(-6, -1, n_tau)
    base = 1.0 / (1 + tau / 1e-3)
    out = np.zeros((n_chunks, n_tau, 2 + n_filters))
    out[:, :, 0] = tau
    for k in range(1 + n_filters):
        noise = 1 + 0.05 * rng.standard_normal((n_chunks, n_tau))
        out[:, :, 1 + k] = base * (1 + 0.05 * k) * noise
    return out


def _g_obj(f, names=("central", "sum3", "sum5"), n_filters=0, good=(0, 1, 2, 4, 5)):
    G = _sib(f, "fcs2corr", "Correlations")()
    for i, name in enumerate(names):
        G.add_corr_chunks(name, _chunk_array(seed=5 + i, n_filters=n_filters))
        G.list_of_g_out.append(name)
    G.good_chunks = list(good)
    G.dwell_time = 1e-6
    return G


def _chunk_attrs_obj(f, module: str, names=("central", "sum3"), n_chunks=4):
    """Object with `<name>_chunk<k>` attributes holding (n, 2) arrays."""
    G = _sib(f, module, "Correlations")()
    for i, name in enumerate(names):
        chunks = _chunk_array(n_chunks=n_chunks, seed=7 + i)
        for k in range(n_chunks):
            setattr(G, f"{name}_chunk{k}", chunks[k])
    return G


def _cross_center_obj(f):
    G = _g_obj(f, names=())
    for j in range(25):
        G.add_corr_chunks(f"det12x{j}", _chunk_array(n_chunks=4, n_tau=15, seed=20 + j)[:, :, :2])
    G.good_chunks = [0, 1, 3]
    return G


def _spatial_obj(f):
    G = _g_obj(f, names=())
    rng = np.random.default_rng(9)
    tau = np.logspace(-6, -2, 10)
    for i in range(25):
        for j in range(25):
            setattr(G, f"det{i}x{j}_average", np.column_stack((tau, rng.random(10), rng.random(10))))
    return G


def _call_atimes_parallel(f, list_of_g, weights=False, **kwargs):
    options = {
        "accuracy": 4,
        "split": 2,
        "min_step": 0,
        "parallel_prefer": "threads",
        "print_info": False,
    }
    options.update(kwargs)
    return f(_atimes(f, weights=weights), list_of_g, **options)


def _call_atimes_corrs(f, list_of_corr, **kwargs):
    options = {"accuracy": 4, "split": 5, "taumax": 0.01}
    options.update(kwargs)
    return f(_atimes(f), list_of_corr, **options)


def _fcs2corr(f, list_of_g, n=1000, **kwargs):
    options = {"accuracy": 16, "print_info": False}
    options.update(kwargs)
    return f(_trace(n), 10.0, list_of_g, **options)


def _correlations_class(cls):
    G = cls()
    names = ["central", "sum3", "sum5"]
    for i, name in enumerate(names):
        G.add_corr_chunks(name, _chunk_array(seed=5 + i))
        G.list_of_g_out.append(name)
    G.good_chunks = [0, 1, 2, 4, 5]
    return (
        G.get_av_corrs(names),
        G.get_av_corrs(names, good_chunks_only=False),
        G.get_corrs("sum3"),
        G.num_chunks,
        G.n_filters,
        G,
    )


def _correlations_chunks_class(cls):
    chunks = cls(_chunk_array(n_filters=1))
    return chunks.num_chunks, chunks.average(), chunks.average([0, 2, 3]), chunks


def _empty_correlations(cls):
    G = cls()
    G.det0_chunk0 = np.arange(6.0).reshape(3, 2)
    G.dwellTime = 1e-6
    return G


def _gauss_stack(n_tau=6) -> np.ndarray:
    rng = np.random.default_rng(2)
    y, x = np.mgrid[0:9, 0:9]
    r2 = (x - 4) ** 2 + (y - 4) ** 2
    stack = np.empty((n_tau, 9, 9))
    for t in range(n_tau):
        sigma2 = 1.5 + 0.8 * t
        stack[t] = 0.5 * np.exp(-r2 / (2 * sigma2)) + 0.002 * rng.standard_normal((9, 9))
    return stack


def _hist_obj(f):
    obj = _atimes(f)
    histograms = decay_histogram()
    for channel in range(25):
        bins = np.arange(256)
        setattr(obj, f"hist{channel}", np.column_stack((bins, histograms[channel])))
    return obj


def _g3d(tau, N, tauD, SF):
    return 1 / N / (1 + tau / tauD) / np.sqrt(1 + tau / (SF**2 * tauD))


def _mem_curve(shape=5.0):
    tau = _tau()
    return tau, _noisy(_g3d(tau, 2.0, 1e-3, shape))


def _mem_param(shape=5.0) -> list:
    return [20, 30, 1e-5, 2e-3, 1, 20, shape]


# Fit problems: (fit_info, true parameters, start parameters, lower bounds, upper bounds, curve)


def _fit_fitfun(f):
    tau = _tau()
    true = np.array([2.0, 1e-3, 5.0, 0.0, 0.05, 1.05])
    y = _noisy(
        _sib(f, "fcs_analytical", "fcs_analytical")(tau, 2.0, 1e-3, 5.0, 0.0, 5e-8, -1.05)
    )
    start = np.array([1.0, 3e-3, 5.0, 0.01, 0.05, 1.05])
    info = np.array([1, 1, 0, 1, 0, 0])
    lower = np.array([0.01, 1e-6, 1.0, -1.0, 0.0, 0.0])
    upper = np.array([100.0, 1.0, 50.0, 1.0, 100.0, 3.0])
    return tau, y, true, start, info, lower, upper


def _fit_an(f):
    tau = _tau()
    true = np.array([2.0, 1e-3, 5.0, 0.0, 0.85])
    y = _noisy(_sib(f, "fcs_analytical", "fcs_analytical")(tau, 2.0, 1e-3, 5.0, 0.0, 0, 0, 0.85))
    start = np.array([1.0, 2e-3, 5.0, 0.0, 1.0])
    info = np.array([1, 1, 0, 0, 1])
    lower = np.array([0.01, 1e-6, 1.0, -1.0, 0.1])
    upper = np.array([100.0, 1.0, 50.0, 1.0, 1.5])
    return tau, y, true, start, info, lower, upper


def _fit_an_2c(f):
    tau = _tau()
    an = _sib(f, "fcs_analytical", "fcs_analytical_2c_anomalous")
    true = np.array([2.0, 1e-3, 2e-2, 1.0, 0.9, 0.6, 1.0, 0.1, 5e-6, 5.0, 0.0])
    y = _noisy(an(tau, 2.0, 1e-3, 2e-2, 1.0, 0.9, 0.6, 0.1, 5e-6, 5.0, 0.0, 1.0))
    start = np.array([1.5, 2e-3, 3e-2, 1.0, 0.9, 0.5, 1.0, 0.1, 5e-6, 5.0, 0.0])
    info = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
    lower = np.array([0.01, 1e-6, 1e-6, 0.1, 0.1, 0.0, 0.1, 0.0, 1e-7, 1.0, -1.0])
    upper = np.array([100.0, 1.0, 1.0, 1.5, 1.5, 1.0, 10.0, 0.9, 1e-3, 50.0, 1.0])
    return tau, y, true, start, info, lower, upper


def _fit_2c(f):
    tau = _tau()
    an = _sib(f, "fcs_analytical", "fcs_2c_analytical")
    true = np.array([2.0, 1.0, 20.0, 0.6, 1.0, 0.1, 5.0, 5.0, 0.0, 0.05, 1.05])
    y = _noisy(an(tau, 2.0, 1e-3, 2e-2, 0.6, 1.0, 0.1, 5e-6, 5.0, 0.0, 5e-8, -1.05))
    start = np.array([1.5, 2.0, 30.0, 0.5, 1.0, 0.1, 5.0, 5.0, 0.0, 0.05, 1.05])
    info = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    lower = np.array([0.01, 1e-3, 1e-3, 0.0, 0.1, 0.0, 0.1, 1.0, -1.0, 0.0, 0.0])
    upper = np.array([100.0, 1e3, 1e4, 1.0, 10.0, 0.9, 100.0, 50.0, 1.0, 100.0, 3.0])
    return tau, y, true, start, info, lower, upper


def _fit_dualfocus_params(rhox):
    n = len(rhox)
    param = np.zeros((9, n))
    param[0] = 2.0
    param[1] = 10.0
    param[2] = [240.0, 250.0, 260.0][:n] if n > 1 else 250.0
    param[3] = 5.0
    param[4] = rhox
    return param


def _fit_dualfocus_curves(f, rhox=(0.0, 200.0, 400.0)):
    tau = np.logspace(-5, 0, 40)
    dual = _sib(f, "fcs_analytical", "fcs_dualfocus")
    param = _fit_dualfocus_params(np.array(rhox))
    curves = np.column_stack(
        [
            _noisy(
                dual(
                    tau,
                    param[0, i],
                    1e-12 * param[1, i],
                    1e-9 * param[2, i],
                    param[3, i],
                    1e-9 * param[4, i],
                    1e-9 * param[5, i],
                    param[8, i],
                    1e-6 * param[6, i],
                    1e-6 * param[7, i],
                ),
                seed=3 + i,
            )
            for i in range(param.shape[1])
        ]
    )
    return tau, curves, param


DUAL_LOWER = np.array([0.01, 0.1, 50.0, 1.0, -1e4, -1e4, -100.0, -100.0, -1.0])
DUAL_UPPER = np.array([100.0, 1e3, 1e3, 50.0, 1e4, 1e4, 100.0, 100.0, 1.0])


def _fit_circ(f):
    tau = _tau()
    circ = _sib(f, "fcs_analytical", "fcs_circular_scanning")
    true = np.array([2.0, 1.5, 2.5e-7, 5.0, 2e-7, 1e-3, 0.0])
    y = _noisy(circ(tau, 2.0, 1.5e-3, 2.5e-7, 5.0, 1e-3, 2e-7, 0.0))
    start = np.array([1.0, 2.5, 2.5e-7, 5.0, 2e-7, 1e-3, 0.0])
    info = np.array([1, 1, 0, 0, 0, 0, 1])
    lower = np.array([0.01, 1e-3, 1e-9, 1.0, 0.0, 1e-5, -1.0])
    upper = np.array([100.0, 1e3, 1e-5, 50.0, 1e-5, 1.0, 1.0])
    return tau, y, true, start, info, lower, upper


def _fit_finitelength(f):
    tau = _tau()
    fl = _sib(f, "fcs_analytical", "fcs_finitelength")
    true = np.array([2.0, 1.0, 5.0, 1.5, 0.01, 1e4])
    y = _noisy(fl(tau, 2.0, 1e-3, 5.0, 1e4, 1.5, 1e-5))
    start = np.array([1.5, 1.5, 5.0, 1.5, 0.01, 1e4])
    info = np.array([1, 1, 0, 0, 0, 0])
    lower = np.array([0.01, 1e-3, 1.0, 0.1, 1e-4, 1.0])
    upper = np.array([100.0, 1e3, 50.0, 100.0, 1.0, 1e7])
    return tau, y, true, start, info, lower, upper


def _fit_free_diffusion_2d(f):
    tau = _tau()
    an = _sib(f, "fcs_analytical", "fcs_2c_2d_analytical")
    true = np.array([2.0, 1.0, 20.0, 0.6, 1.0, 0.1, 5.0, 0.0, 1e-8, -1.0])
    y = _noisy(an(tau, 2.0, 1e-3, 2e-2, 0.6, 1.0, 0.1, 5e-6, 0.0, 1e-8, -1.0))
    start = np.array([1.5, 2.0, 30.0, 0.5, 1.0, 0.1, 5.0, 0.0, 1e-8, -1.0])
    info = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    lower = np.array([0.01, 1e-3, 1e-3, 0.0, 0.1, 0.0, 0.1, -1.0, 0.0, -3.0])
    upper = np.array([100.0, 1e3, 1e4, 1.0, 10.0, 0.9, 100.0, 1.0, 1.0, 0.0])
    return tau, y, true, start, info, lower, upper


def _fit_nanosecond(f):
    tau = np.logspace(-9, -3, 60)
    an = _sib(f, "fcs_analytical", "nanosecond_fcs_analytical")
    true = np.array([0.5, 1.0, 2e-9, 0.2, 1e-7, 0.1, 5e-9, 0.3, 3e-6, 5e-4, 5.0])
    y = _noisy(an(tau, *true))
    start = np.array([0.4, 0.8, 3e-9, 0.2, 1e-7, 0.1, 5e-9, 0.3, 3e-6, 4e-4, 5.0])
    info = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    lower = np.array([0.0, 0.0, 1e-10, 0.0, 1e-9, 0.0, 1e-10, 0.0, 1e-7, 1e-6, 1.0])
    upper = np.array([10.0, 1.0, 1e-6, 5.0, 1e-3, 5.0, 1e-6, 5.0, 1e-3, 1.0, 50.0])
    return tau, y, true, start, info, lower, upper


def _fit_reaction(f):
    tau = _tau()
    an = _sib(f, "fcs_analytical", "uncoupled_reaction_diffusion")
    true = np.array([0.5, 1e-3, 5.0, 0.7, 50.0])
    y = _noisy(an(tau, 0.5, 1e-3, 5.0, 0.7, 50.0))
    start = np.array([0.4, 2e-3, 5.0, 0.6, 30.0])
    info = np.array([1, 1, 0, 1, 1])
    lower = np.array([0.0, 1e-6, 1.0, 0.0, 0.0])
    upper = np.array([10.0, 1.0, 50.0, 1.0, 1e4])
    return tau, y, true, start, info, lower, upper


def _residual(problem):
    """Call a fit function once at the start values of a problem."""

    def call(f):
        tau, y, _, start, info, _, _ = problem(f)
        return f(start[info == 1], start[info == 0], info, tau, y)

    return call


def _fit(problem, name, **kwargs):
    def call(f):
        tau, y, _, start, info, lower, upper = problem(f)
        return f(y, tau, _sib(f, "fcs_fit", name), info, start, lower, upper, -1, **kwargs)

    return call


def _fit_by_string(problem, name, **kwargs):
    def call(f):
        tau, y, _, start, info, lower, upper = problem(f)
        return f(y, tau, name, info, start, lower, upper, -1, **kwargs)

    return call


def _fit_general_data(f):
    return _fit_an_model(f)


def _fit_an_model(f):
    """Seven-parameter problem for `fcs_analytical` used as a general fit model."""
    tau = _tau()
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    y = _noisy(model(tau, 2.0, 1e-3, 5.0, 0.0, 0.0, 0.0, 1.0))
    start = np.array([1.0, 2e-3, 5.0, 0.01, 0.0, 0.0, 1.0])
    info = np.array([1, 1, 0, 1, 0, 0, 0])
    lower = np.array([0.01, 1e-6, 1.0, -1.0, -1.0, -3.0, 0.1])
    upper = np.array([100.0, 1.0, 50.0, 1.0, 1.0, 3.0, 1.5])
    return tau, y, start, info, lower, upper


def _general_global_curves(f):
    tau = _tau()
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    curves = np.column_stack(
        [_noisy(model(tau, n, 1e-3, 5.0, 0.0, 0.0, 0.0, 1.0), seed=3 + i) for i, n in enumerate([1.0, 2.0, 3.0])]
    )
    param = np.zeros((7, 3))
    param[0] = [1.5, 2.5, 3.5]
    param[1] = 2e-3
    param[2] = 5.0
    param[6] = 1.0
    info = np.array([1, 1, 0, 1, 0, 0, 0])
    global_param = [False, True, False, False, False, False, False]
    lower = np.array([0.01, 1e-6, 1.0, -1.0, -1.0, -3.0, 0.1])
    upper = np.array([100.0, 1.0, 50.0, 1.0, 1.0, 3.0, 1.5])
    return tau, curves, param, info, global_param, lower, upper


def _general_global_vectors(param, info, global_param):
    start, fixed = [], []
    for i in range(len(info)):
        values = [param[i, 0]] if global_param[i] else list(param[i])
        (start if info[i] else fixed).extend(values)
    return np.array(start), np.array(fixed)


def _fitfun_general_single(f):
    tau, y, start, info, _, _ = _fit_general_data(f)
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    return f(start[info == 1], start[info == 0], info, tau, y, model, [False] * 7)


def _fitfun_general_global(f):
    tau, curves, param, info, global_param, _, _ = _general_global_curves(f)
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    start, fixed = _general_global_vectors(param, info, global_param)
    weights = np.random.default_rng(4).random(curves.shape) + 0.5
    return (
        f(start, fixed, info, tau, curves, model, global_param),
        f(start, fixed, info, tau, curves, model, global_param, weights, np.array([1, 1e-3, 1, 1, 1, 1, 1])),
    )


def _fit_corr_single(f):
    tau, y, start, info, lower, upper = _fit_general_data(f)
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    return f(y, tau, info, start, lower_bounds=lower, upper_bounds=upper, fitmodel=model)


def _fit_corr_default_bounds(f):
    tau, y, start, info, _, _ = _fit_general_data(f)
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    return f(y, tau, info, start, fitmodel=model, ftol=1e-10)


def _fit_corr_global(f):
    tau, curves, param, info, global_param, lower, upper = _general_global_curves(f)
    model = _sib(f, "fcs_analytical", "fcs_analytical")
    return f(
        curves,
        tau,
        info,
        param,
        global_param=global_param,
        lower_bounds=lower,
        upper_bounds=upper,
        fitmodel=model,
        param_factors10=np.array([1, 1e-3, 1, 1, 1, 1, 1]),
    )


def _dual_start(f, global_n=False):
    tau, curves, param = _fit_dualfocus_curves(f)
    info = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    global_param = [global_n, True, False, False, False, False, False, False, False]
    start_param = param.copy()
    start_param[0] = 1.5
    start_param[1] = 8.0
    return tau, curves, start_param, info, global_param


def _dual_vectors(param, info, global_param):
    return _general_global_vectors(param, info, global_param)


def _fitfun_dualfocus_single(f):
    tau, curves, param = _fit_dualfocus_curves(f)
    p = param[:, 1]
    info = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    return f(p[info == 1], p[info == 0], info, None, tau, curves[:, 1])


def _fitfun_dualfocus_global(f, global_n):
    tau, curves, param, info, global_param = _dual_start(f, global_n)
    start, fixed = _dual_vectors(param, info, global_param)
    weights = np.random.default_rng(4).random(curves.shape) + 0.5
    return (
        f(start, fixed, info, global_param, tau, curves),
        f(start, fixed, info, global_param, tau, curves, weights),
    )


def _fcs_fit_dualfocus_global(f, global_n):
    tau, curves, param, info, global_param = _dual_start(f, global_n)
    return f(curves, tau, info, param, global_param=global_param, lBounds=DUAL_LOWER, uBounds=DUAL_UPPER)


def _fcs_fit_dualfocus_single(f):
    tau, curves, param = _fit_dualfocus_curves(f)
    info = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    start = param[:, 1].copy()
    start[1] = 8.0
    return f(curves[:, 1], tau, info, start, lBounds=DUAL_LOWER, uBounds=DUAL_UPPER)


def _fcs_fit_dualfocus_no_bounds(f):
    tau, curves, param = _fit_dualfocus_curves(f)
    info = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    return _error(f, curves[:, 1], tau, info, param[:, 1].copy())


def _fcs_fit_global_dual(f, name):
    tau, curves, param, info, global_param = _dual_start(f, False)
    function = _sib(f, "fcs_fit", name)
    return f(curves, tau, function, info, param, DUAL_LOWER, DUAL_UPPER, -1, global_param=global_param)


def _fcs_fit_weighted(f):
    tau, y, _, start, info, lower, upper = _fit_fitfun(f)
    weights = np.random.default_rng(6).random(len(tau)) + 0.5
    return f(y, tau, _sib(f, "fcs_fit", "fitfun"), info, start, lower, upper, -1, weights=weights)


def _fcs_fit_mem(f, as_string):
    tau, y = _mem_curve(3.0)
    fitfun = "mem_fit_free_diffusion" if as_string else _sib(f, "fcs_fit", "mem_fit_free_diffusion")
    info = np.zeros(7)
    return f(y, tau, fitfun, info, _mem_param(3.0), np.zeros(7), np.ones(7), -1)


def _fcs_fit_2c_plot(f):
    tau, y, _, start, info, lower, upper = _fit_2c(f)
    return f(y, tau, "fitfun_2c", info, start, lower, upper, "central", 0, True)


def _fcs_fit_all_fitted_plot(f):
    tau, y, _, start, _, lower, upper = _fit_an(f)
    info = np.ones(5, dtype=int)
    result = f(y, tau, "fitfun_an", info, start, lower, upper, "central", 0, True)
    return result, start


def _fit_dualfocus_function(f):
    tau, curves, param = _fit_dualfocus_curves(f)
    info = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    start = param[:, 1].copy()
    start[1] = 8.0
    function = _sib(f, "fcs_fit", "fitfun_dualfocus")
    return f(curves[:, 1], tau, function, info, start, DUAL_LOWER, DUAL_UPPER, -1)


def _plot_fit(f):
    tau, y, _, start, info, _, _ = _fit_2c(f)
    result = SimpleNamespace(x=start[info == 1] * 1.1, fun=0.01 * np.sin(np.arange(len(tau))))
    param = start.copy()
    return f(tau, y, param, info, result, "sum3", 0, True), param


def _plot_fit_single(f):
    tau, y, _, start, info, _, _ = _fit_fitfun(f)
    result = SimpleNamespace(x=start[info == 1] * 1.1, fun=0.01 * np.sin(np.arange(len(tau))))
    param = start.copy()
    return f(tau, y, param, info, result, "central", 0, False), param


def _struct_obj(f):
    G = _sib(f, "fcs2corr", "Correlations")()
    rng = np.random.default_rng(11)
    G.dwellTime = 1e-5
    G.spatialCorr = rng.random((9, 9, 12))
    G.crossCenterAv = rng.random((12, 6))
    for name in ["TB", "BT", "LR", "RL", "LT", "TL", "BR", "RB", "LB", "BL", "TR", "RT"]:
        setattr(G, f"twofocus{name}_average", rng.random((12, 2)))
    return G


def _global_fit_info_obj(f):
    G = _sib(f, "fcs2corr", "Correlations")()
    rng = np.random.default_rng(12)
    for name in ["a", "b", "c"]:
        setattr(G, f"{name}_averageX", np.column_stack((np.logspace(-6, 0, 10), rng.random(10))))
    return G


def _fingerprint_param():
    return np.array([0.0, 0.0, 0.2, 500.0, 1.0, 50.0, 20.0])


def _fingerprint(f, shift=(0.0, 0.0)):
    param = _fingerprint_param()
    param[:2] = shift
    y = _sib(f, "w0_from_fingerprint", "calc_fingerprint")(param, 40)
    y = y * (1 + 0.01 * np.random.default_rng(3).standard_normal(y.shape))
    return param, y


def _fingerprint_fit(f, fit_info, start_overrides, shift=(0.0, 0.0)):
    param, y = _fingerprint(f, shift)
    start = param.copy()
    start[:2] = 0.0
    for index, value in start_overrides.items():
        start[index] = value
    return f(y, fit_info, start, N=40)


def _fingerprint_residual(f):
    param, y = _fingerprint(f)
    info = np.array([0, 0, 1, 0, 0, 0, 0])
    return f(np.array([0.25]), param[info == 0], info, y, 40)


def _two_focus_args():
    return (np.logspace(-5, 0, 40), 1e-7, 2e-7, 0.0, 6e17, 1e-11, 2.5e-7, 2.5e-7, 1e-6, 1e-6)


def _psf_arrays():
    rng = np.random.default_rng(13)
    return 2.5e-7 * (1 + 0.1 * rng.random(25)), 1e-6 * (1 + 0.1 * rng.random(25))


def _simulate_cross_center(f):
    w, z = _psf_arrays()
    return f(np.logspace(-5, 0, 20), 6e17, 1e-11, w, z, 1e-7)


def _simulate_spatialcorr(f, tau):
    w, z = _psf_arrays()
    return f(tau, 1e-7, 6e17, 1e-11, w, z)


def _polar_g(n_dirs, n_lags=30):
    rng = np.random.default_rng(14)
    lag = np.arange(n_lags)[:, None]
    return (1.0 + 0.2 * rng.random((n_lags, n_dirs))) / (1 + lag / 5.0)


def _curve_2d():
    tau, G = correlation_curve()
    return np.column_stack((tau, G[:, 0]))


def _sort_curves(n, lags=12):
    rng = np.random.default_rng(15)
    base = 1.0 / (1 + np.arange(lags) / 3.0)
    return base[:, None] * (1 + 0.05 * rng.standard_normal((lags, n)))


def _good_chunks_obj(f):
    G = _sib(f, "fcs2corr", "Correlations")()
    chunks = _chunk_array(n_chunks=10, n_tau=25, seed=21)
    chunks[3, :, 1] *= 3
    G.add_corr_chunks("sum5", chunks)
    G.list_of_g_out.append("sum5")
    return G


def _spatial_g_chunks(f):
    G = _sib(f, "select_g", "Correlations")()
    G.central_chunk0 = np.ones((3, 2))
    G.central_average = np.full((3, 2), 2.0)
    G.sum3_average = np.full((3, 2), 3.0)
    G.dwellTime = 1e-6
    return G


def _mixed_length_obj(f):
    G = _sib(f, "select_g", "Correlations")()
    G.a_chunk0 = np.ones((5, 2))
    G.a_chunk1 = 2.0 * np.ones((5, 2))
    G.a_chunk2 = 3.0 * np.ones((4, 2))
    return G


def _add_cross_all(f, averaging, list_of_g_out, c0_all, c1_all):
    G = _sib(f, "fcs2corr", "Correlations")()
    g_cross_all = np.random.default_rng(17).random((2, 3, 10, 25, 25))
    f(G, g_cross_all, np.logspace(-6, -2, 10), averaging, list_of_g_out, c0_all, c1_all, channels_order=list(range(25)))
    return G


def _no_good_chunks(f):
    G = _g_obj(f)
    G.good_chunks = None
    return G


def _auto_wk(f):
    x = _trace(1000)[:, 12].astype(float)
    return f(x, x, 1e-5)


def _auto_tt(f):
    x = _trace(1000)[:, 12].astype(float)
    return f(x, x, 1e-5, 20)


def _padded_trace(kind: int) -> np.ndarray:
    x = _trace(1000)[:, 12].astype(float)
    if kind == 0:
        x[:200] = 0
    else:
        x[-200:] = 0
    return x


def _fcs_fit_dualfocus_string(f):
    tau, curves, param = _fit_dualfocus_curves(f)
    info = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0])
    start = param[:, 1].copy()
    start[1] = 8.0
    return f(curves[:, 1], tau, "fitfun_dualfocus", info, start, DUAL_LOWER, DUAL_UPPER, -1)


def _chi2(f):
    tau = _tau()
    y = _noisy(_g3d(tau, 2.0, 1e-3, 5.0))
    result = SimpleNamespace(fun=0.01 * np.cos(np.arange(len(tau))) * y)
    return f(tau, y, np.array([1, 1, 0, 1, 0, 0]), result)


def _std() -> np.ndarray:
    return np.array([0.1, 0.2, 0.0, 0.4, 0.15, 0.3])


def _make_fit_info(f):
    return f(
        _global_fit_info_obj(f),
        ["a", "b", "c"],
        np.array([0.0, 0.0, 50.0, 1.0, -1e3, -1e3, -10.0, -10.0, -1.0]),
        np.array([1e2, 1e2, 1e3, 50.0, 1e3, 1e3, 10.0, 10.0, 1.0]),
        np.array([1, 1, 1, 1, 0, 0, 0, 0, 1]),
        np.array([2.0, 10.0, 250.0, 5.0, 0.0, 0.0, 0.0]),
        [0.0, 200.0, 400.0],
        [0.0, 0.0, 0.0],
    )


def _read_global_result(f, fitarray=(1,) * 9, n_x=19):
    result = SimpleNamespace(x=np.random.default_rng(16).random(n_x) + 0.5)
    return f(result, ["a", "b", "c"], fitarray, np.arange(1.0, 8.0), [0.0, 200.0, 400.0], [0.0, 0.0, 0.0])


def _fitresult_class(cls):
    result = cls()
    result.x = np.arange(3.0)
    result.fun = np.ones(3)
    return result


CASES: list[Case] = [
    # atimes2corr
    Case(f"{A2C}:Correlations", _empty_correlations),
    Case(f"{A2C}:time_coarsening", lambda f: f(_macro(12), np.ones(400)), "ones"),
    Case(f"{A2C}:time_coarsening", lambda f: f(_macro(12), np.random.default_rng(1).random(400)), "weights"),
    Case(f"{A2C}:time_coarsening_all_ch", lambda f: f(_channel_array(f))),
    Case(f"{A2C}:generate_list_of_tau_values", lambda f: f(1e5, 8, 0, True), "log"),
    Case(f"{A2C}:generate_list_of_tau_values", lambda f: f(1e5, 8, 4, True), "log_min_step"),
    Case(f"{A2C}:generate_list_of_tau_values", lambda f: f(200, 8, 0, False), "linear"),
    Case(f"{A2C}:atimes_2_corr_single", lambda f: f(_macro(12) // 1024, _macro(13) // 1024, np.ones(400), np.ones(400), 3, 1)),
    Case(f"{A2C}:atimes_2_corr_single", lambda f: f(_macro(12), _macro(12), np.ones(400), np.ones(400), 2 * 10**6, 1), "overlap_empty"),
    Case(f"{A2C}:atimes_2_corr_single", lambda f: f(_macro(12), _macro(12), np.zeros(400), np.zeros(400), 5, 1), "zero_weights"),
    Case(f"{A2C}:atimes_2_corr", lambda f: f(_macro(12), _macro(12), [1], [1], 1e-9, 8, "auto", True, True), "auto_default_step"),
    Case(f"{A2C}:atimes_2_corr", lambda f: f(_macro(12), _macro(13), [1], [1], 1e-9, 8, 1e-4, True, True, 0), "cross_min_step_0"),
    Case(f"{A2C}:atimes_2_corr", lambda f: f(_macro(12), _macro(12), np.random.default_rng(1).random(400), np.random.default_rng(2).random(400), 1e-9, 8, "auto", True, True, 4), "weighted"),
    Case(f"{A2C}:atimes_2_corr", lambda f: f(_macro(12), _macro(12), [1], [1], 1e-9, 8, "auto", False, True, 0), "no_coarsening"),
    Case(f"{A2C}:atimes_2_corr", lambda f: f(_macro(12) // 8000, _macro(12) // 8000, [1], [1], 1e-9, 8, 5e-5, False, False, 0), "linear_tau"),
    Case(f"{A2C}:atimes_2_corrs", lambda f: _call_atimes_corrs(f, [12, "sum3"]), "single_and_keyword"),
    Case(f"{A2C}:atimes_2_corrs", lambda f: _call_atimes_corrs(f, [12], taumax="auto"), "auto_taumax"),
    Case(f"{A2C}:atimes_2_corrs", lambda f: f(_atimes(f, weights=True), [12, "C3+4"], accuracy=4, split=5, taumax=0.01), "filtered"),
    Case(f"{A2C}:atimes_2_corrs", lambda f: _call_atimes_corrs(f, ["av"]), "average"),
    Case(f"{A2C}:all_crosscorr_at_tau", lambda f: f(_channel_array(f), 3), "small_tau"),
    Case(f"{A2C}:all_crosscorr_at_tau", lambda f: f(_channel_array(f), 2 * 10**6), "overlap_empty"),
    Case(f"{A2C}:all_crosscorr_at_tau", lambda f: f(np.column_stack((_channel_array(f)[:, :3], np.zeros(len(_channel_array(f))))), 3), "empty_channel"),
    Case(f"{A2C}:atimes_2_all_crosscorr", lambda f: f(_channel_array(f), 1e-5, 4, "auto", True, True, 0), "auto"),
    Case(f"{A2C}:atimes_2_all_crosscorr", lambda f: f(_channel_array(f), 1e-5, 4, 0.05, True, True, 2), "min_step"),
    Case(f"{A2C}:atimes_2_all_crosscorr", lambda f: f(_channel_array(f), 1e-5, 4, 0.05, False, True, 0), "no_coarsening"),
    Case(f"{A2C}:atimes_2_all_crosscorr", lambda f: f(_channel_array(f), 1e-5, 4, 0.0005, True, False, 0), "linear_tau"),
    # atimes2corrparallel
    Case(f"{A2CP}:convert_crossall_to_list_of_g", lambda f: f(12), "default"),
    Case(f"{A2CP}:convert_crossall_to_list_of_g", lambda f: f(12, 3), "start_channel"),
    Case(f"{A2CP}:extract_atimes_for_corr", lambda f: f(_atimes(f), 12, 2), "channel"),
    Case(f"{A2CP}:extract_atimes_for_corr", lambda f: f(_atimes(f), "x1213", 2), "cross"),
    Case(f"{A2CP}:extract_atimes_for_corr", lambda f: f(_atimes(f), "C3+4x5+6", 2), "custom_cross"),
    Case(f"{A2CP}:extract_atimes_for_corr", lambda f: f(_atimes(f), "C3+4", 2), "custom_auto"),
    Case(f"{A2CP}:extract_atimes_for_corr", lambda f: f(_atimes(f), "sum3", 2, print_info=False), "keyword"),
    Case(f"{A2CP}:parallel_g", lambda f: f(_macro(12), [1], 1e-5, 0, 2, 4, 0.02, True, True, 0, 1), "unfiltered"),
    Case(f"{A2CP}:parallel_g", lambda f: f(_macro(12), np.random.default_rng(1).random(400), 1e-5, 1, 2, 4, 0.02, True, True, 0, 1), "filtered"),
    Case(f"{A2CP}:parallel_gx", lambda f: f(_macro(12), [1], _macro(13), [1], 1e-5, 0, 2, 4, 0.02, True, True, 0, 2), "unfiltered"),
    Case(f"{A2CP}:parallel_gx", lambda f: f(_macro(12), np.random.default_rng(1).random(400), _macro(13), np.random.default_rng(2).random(400), 1e-5, 1, 2, 4, 0.02, True, True, 0, 2), "filtered"),
    Case(f"{A2CP}:calc_average_correlation", lambda f: f(_chunk_attrs_obj(f, "atimes2corr")), "equal_length"),
    Case(f"{A2CP}:calc_average_correlation", lambda f: f(_mixed_length_obj(f)), "mixed_length"),
    Case(f"{A2CP}:add_cross_all_to_G", lambda f: _add_cross_all(f, ["12x7+11x6", "7x12"], ["up", "left"], None, None), "averaging_names"),
    Case(f"{A2CP}:add_cross_all_to_G", lambda f: _add_cross_all(f, ["12x7+11x6", "7x12"], None, None, None), "averaging_default_names"),
    Case(f"{A2CP}:add_cross_all_to_G", lambda f: _add_cross_all(f, None, ["a", "b"], ["12", "7"], ["07", "12"]), "pairs"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, [12, "sum3"], list_of_g_out=["c", "s3"]), "auto_correlations"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, [12, "C3+4"], weights=True, list_of_g_out=["c", "cs"]), "filtered"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, [12], weights=True, filters=[0], list_of_g_out=["c"]), "unfiltered_only"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, ["x1213", "C3+4x5+6"], list_of_g_out=["x", "cs"], taumax=0.02), "cross_correlations"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _error(_call_atimes_parallel, f, [12]), "missing_output_names"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _error(_call_atimes_parallel, f, ["av"], list_of_g_out=[f"det{i}" for i in range(25)]), "average"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, ["crossAll"]), "cross_all"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, ["crossAll"], averaging=["12x7+11x6", "7x12"], list_of_g_out=["up", "left"]), "cross_all_averaging"),
    Case(f"{A2CP}:atimes_2_corrs_parallel", lambda f: _call_atimes_parallel(f, ["crossAll", [[0], [1], [12]]], weights=True), "cross_all_channel_groups"),
    # fcs2corr: classes
    Case(f"{F2C}:Correlations", _correlations_class),
    Case(f"{F2C}:Correlations.get_av_corrs", lambda f: _error(f, _no_good_chunks(f), ["central"]), "no_good_chunks"),
    Case(f"{F2C}:Correlations_chunks", _correlations_chunks_class),
    # fcs2corr: correlate
    Case(f"{F2C}:correlate", lambda f: f(_trace(1000)[:, 12].astype(float), _trace(1000)[:, 13].astype(float), m=16, deltat=1e-5), "multipletau"),
    Case(f"{F2C}:correlate", lambda f: f(_trace(1000)[:, 12].astype(float), _trace(1000)[:, 12].astype(float), m=16, deltat=1e-5, normalize=False), "multipletau_unnormalized"),
    Case(f"{F2C}:correlate", lambda f: f(_trace(1000)[:, 12].astype(float), _trace(1000)[:, 13].astype(float), m=16, deltat=1e-5, algorithm="wiener-khinchin"), "wiener_khinchin"),
    Case(f"{F2C}:correlate", lambda f: f(_trace(1000)[:, 12].astype(float), None, m=10, binsize=4, algorithm="pch"), "pch"),
    Case(f"{F2C}:correlate", lambda f: f(_trace(1000)[:, 12].astype(float), _trace(1000)[:, 13].astype(float), m=8, deltat=1e-5, algorithm="tt2corr"), "time_tags"),
    # fcs2corr: fcs2corr modes
    Case(f"{F2C}:fcs2corr", lambda f: f(_trace(1000)[:, 12], 10.0, [0], 16, 1, "multipletau", None, None, False), "vector"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, [12, 3]), "channel_numbers"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, [12, 3], list_of_g_out=["a", "b"]), "output_names"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["x1213", "x0112"]), "cross"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["C3+4+7", "C1+2x3+4"]), "custom_sum"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["A1+3+5"]), "average_of_channels"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["crossCenter"]), "cross_center"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["singleElement"]), "single_element"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["2MPD"]), "two_elements"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["crossAll"], n=600), "cross_all"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["crossAll"], n=600, algorithm="sparse_matrices"), "cross_all_sparse"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["crossAll"], n=600, algorithm="sparse_matrices", averaging="default"), "cross_all_sparse_default_averaging"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["crossAll"], n=600, algorithm="sparse_matrices", averaging=["12x7+11x6", "7x12"], list_of_g_out=["up", "left"]), "cross_all_sparse_averaging"),
    Case(f"{F2C}:fcs2corr", lambda f: f(_trace(1000)[:20], 10.0, ["autoSpatial"], 16, 1, "multipletau", None, None, False), "auto_spatial"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["av"], n=600), "average_all"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["central", "sum3", "chess0"]), "keywords"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["central", 12, "x1213"], algorithm="wiener-khinchin"), "wiener_khinchin"),
    Case(f"{F2C}:fcs2corr", lambda f: _fcs2corr(f, ["central"], binsize=4, algorithm="pch"), "pch"),
    Case(f"{F2C}:fcs2corrsplit", lambda f: f(_trace(1000), 10.0, ["central", "sum3"], 16, 1), "no_split"),
    Case(f"{F2C}:fcs2corrsplit", lambda f: _error(f, _trace(1000), 10.0, ["central", "sum3"], 16, 2), "split_keywords"),
    Case(f"{F2C}:fcs2corrsplit", lambda f: f(_trace(1000), 10.0, ["2MPD"], 16, 2), "split_two_elements"),
    Case(f"{F2C}:fcs_sparse", lambda f: f(_trace(1000), 1e-5, 16), "normalized"),
    Case(f"{F2C}:fcs_sparse", lambda f: f(_trace(1000), 1e-5, 16, False), "unnormalized"),
    Case(f"{F2C}:fcs_sparse", lambda f: f(_trace(999), 1e-5, 15), "odd_length_and_m"),
    Case(f"{F2C}:fcs_spatialcorrav", lambda f: f(_spatial_obj(f)), "array"),
    Case(f"{F2C}:fcs_spatialcorrav", lambda f: f(_spatial_obj(f), 5, "object"), "attributes"),
    Case(f"{F2C}:correlate_parallel", lambda f: f(_trace(512)[:, :3].T.astype(float), 16, 1e-5, False, True, float), "average"),
    Case(f"{F2C}:correlate_parallel", lambda f: f(_trace(512)[:, :3].T.astype(float), 16, 1e-5, False, False, float, "first"), "first"),
    Case(f"{F2C}:correlate_parallel", lambda f: f(_trace(512)[:, :3].T.astype(float), 16, 1e-5, False, True, float, "second"), "second"),
    Case(f"{F2C}:correlate_parallel", lambda f: f(_trace(512)[:, :3].T.astype(float), 16, 1e-5, False, True, float, "average", True), "sum"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(512)[:, :3].T.astype(float), 16, 1e-5, True, True, float), "normalized"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(512)[:, :3].T.astype(float), 16, 1e-5, 1, True, float), "normalize_not_bool"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(512)[:, :3].T.astype(float), 16, 1e-5, False, 1, float), "copy_not_bool"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(512)[:, :3].T.astype(float), 16, 1e-5, False, True, float, "average", 1), "ret_sum_not_bool"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(512)[:, :3].T.astype(float), 16, 1e-5, True, True, float, "average", True), "normalize_and_sum"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(512)[:, :3].T.astype(float), 16, 1e-5, False, True, float, "last"), "compress_unknown"),
    Case(f"{F2C}:correlate_parallel", lambda f: _error(f, _trace(20)[:, :3].T.astype(float), 16, 1e-5, False, True, float), "too_short"),
    Case(f"{F2C}:fcs_av_chunks", lambda f: f(_chunk_attrs_obj(f, "select_g"), [0, 2, 3]), "chunk_subset"),
    Case(f"{F2C}:fcs_crosscenter_av", lambda f: f(_cross_center_obj(f)), "good_chunks_object"),
    Case(f"{F2C}:fcs_crosscenter_av", lambda f: f(_cross_center_obj(f), "_average", False), "all_chunks_array"),
    Case(f"{F2C}:corrs2average_for_stics", lambda f: f(), "default"),
    Case(f"{F2C}:corrs2average_for_stics", lambda f: _error(f, 5, [0, 1, 3, 4, 5, 9, 15, 19, 20, 21, 23, 24]), "pixels_off"),
    Case(f"{F2C}:list_of_g_out_v1p1", lambda f: f(_chunk_attrs_obj(f, "fcs2corr")), "chunks"),
    # timetrace2corr
    Case(f"{T2C}:autocorrelation_wiener_khinchin", lambda f: f(_trace(1000)[:, 12].astype(float), _trace(1000)[:, 13].astype(float), 1e-5), "cross"),
    Case(f"{T2C}:autocorrelation_wiener_khinchin", lambda f: _auto_wk(f), "auto_same_array"),
    Case(f"{T2C}:autocorrelation_wiener_khinchin", lambda f: f(_padded_trace(0), _padded_trace(1), 1e-5), "zero_mean_lags"),
    Case(f"{T2C}:tt2corr", lambda f: f(_trace(1000)[:, 12].astype(float), _trace(1000)[:, 13].astype(float), 1e-5, 20), "cross"),
    Case(f"{T2C}:tt2corr", lambda f: _auto_tt(f), "auto_same_array"),
    Case(f"{T2C}:tt2corr", lambda f: f(_padded_trace(0), _padded_trace(1), 1e-5, 20), "zero_mean_lags"),
    # fcs_analytical
    Case(f"{AN}:fcs_analytical", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 0.01), "free_diffusion"),
    Case(f"{AN}:fcs_analytical", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 0.01, 5e-8, -1.05, 0.8), "afterpulsing_anomalous"),
    Case(f"{AN}:fcs_analytical", lambda f: f(np.float64(1e-3), 2.0, 1e-3, 5.0, 0.0), "scalar_tau"),
    Case(f"{AN}:fcs_2c_analytical", lambda f: f(_tau(), 2.0, 1e-3, 2e-2, 0.6), "defaults"),
    Case(f"{AN}:fcs_2c_analytical", lambda f: f(_tau(), 2.0, 1e-3, 2e-2, 0.6, 1.5, 0.1, 5e-6, 5.0, 0.01, 5e-8, -1.05), "triplet_afterpulsing"),
    Case(f"{AN}:fcs_analytical_2c_anomalous", lambda f: f(_tau(), 2.0, 1e-3, 2e-2, 0.9, 0.8, 0.6, 0.1, 5e-6, 5.0, 0.01, 1.5)),
    Case(f"{AN}:fcs_analytical_2c_anomalous_c", lambda f: f(_tau(), 1.0, 10.0, 1.0, 0.9, 0.8, 0.6, 0.1, 5e-6, 0.25, 5.0, 0.01, 1.5)),
    Case(f"{AN}:fcs_dualfocus", lambda f: f(_tau(), 2.0, 1e-11, 2.5e-7, 5.0, 2e-7, 1e-7, 0.01), "static"),
    Case(f"{AN}:fcs_dualfocus", lambda f: f(_tau(), 2.0, 1e-11, 2.5e-7, 5.0, 2e-7, 1e-7, 0.01, 1e-4, -5e-5), "flow"),
    Case(f"{AN}:fcs_dualfocus_c", lambda f: f(_tau(), 1.0, 10.0, 0.25, 5.0, 0.2, 0.1, 0.01), "static"),
    Case(f"{AN}:fcs_dualfocus_c", lambda f: f(_tau(), 1.0, 10.0, 0.25, 5.0, 0.2, 0.1, 0.01, 100.0, -50.0), "flow"),
    Case(f"{AN}:fcs_circular_scanning", lambda f: f(_tau(), 2.0, 1.5e-3, 2.5e-7, 5.0, 1e-3, 2e-7, 0.01), "static"),
    Case(f"{AN}:fcs_circular_scanning", lambda f: f(_tau(), 2.0, 1.5e-3, 2.5e-7, 5.0, 1e-3, 2e-7, 0.01, 1e-4, 5e-5), "flow"),
    Case(f"{AN}:fcs_circular_scanning_c", lambda f: f(_tau(), 1.0, 10.0, 0.25, 5.0, 1e-3, 0.2, 0.01)),
    Case(f"{AN}:fcs_2c_2d_analytical", lambda f: f(_tau(), 2.0, 1e-3, 2e-2, 0.6), "defaults"),
    Case(f"{AN}:fcs_2c_2d_analytical", lambda f: f(_tau(), 2.0, 1e-3, 2e-2, 0.6, 1.5, 0.1, 5e-6, 0.01, 1e-8, -1.0), "triplet_afterpulsing"),
    Case(f"{AN}:nanosecond_fcs_analytical", lambda f: f(np.logspace(-9, -3, 60), 0.5, 1.0, 2e-9, 0.2, 1e-7, 0.1, 5e-9, 0.3, 3e-6, 5e-4, 5.0)),
    Case(f"{AN}:uncoupled_reaction_diffusion", lambda f: f(_tau(), 0.5, 1e-3, 5.0, 0.7, 50.0)),
    Case(f"{AN}:fcs_finitelength", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 1e4, 1.5, 1e-5)),
    Case(f"{AN}:fcs_1c_2dgl", lambda f: f(_tau(), 2.0, 1e-3, 5.0, 0.01)),
    Case(f"{AN}:compute_gamma", lambda f: f(np.logspace(-6, np.log10(8), 40), 2.0, 1e-3), "defaults_across_half_period"),
    Case(f"{AN}:compute_gamma", lambda f: f(_tau(), 2.0, 1e-3, 1.5, 1e-5, 1, 5.0, 1e4), "short_period"),
    # fcs_fit: residual functions
    Case(f"{FIT}:fitfun", _residual(_fit_fitfun)),
    Case(f"{FIT}:fitfun_an", _residual(_fit_an)),
    Case(f"{FIT}:fitfun_an_2c", _residual(_fit_an_2c)),
    Case(f"{FIT}:fitfun_2c", _residual(_fit_2c)),
    Case(f"{FIT}:fitfun_general", _fitfun_general_single, "single_curve"),
    Case(f"{FIT}:fitfun_general", _fitfun_general_global, "global_curves"),
    Case(f"{FIT}:fitfun_dualfocus", _fitfun_dualfocus_single, "single_curve"),
    Case(f"{FIT}:fitfun_dualfocus", lambda f: _fitfun_dualfocus_global(f, False), "global_diffusion"),
    Case(f"{FIT}:fitfun_dualfocus", lambda f: _fitfun_dualfocus_global(f, True), "global_concentration"),
    Case(f"{FIT}:fitfun_circfcs", _residual(_fit_circ)),
    Case(f"{FIT}:fitfun_finitelength", _residual(_fit_finitelength)),
    Case(f"{FIT}:fitfun_free_diffusion_2d", _residual(_fit_free_diffusion_2d)),
    Case(f"{FIT}:fitfun_nanosecond_fcs", _residual(_fit_nanosecond)),
    Case(f"{FIT}:fitfun_uncoupled_reaction_diffusion", _residual(_fit_reaction)),
    Case(f"{FIT}:compute_gamma", lambda f: f(np.logspace(-6, np.log10(8), 40), 2.0, 1e-3), "defaults_across_half_period"),
    Case(f"{FIT}:compute_gamma", lambda f: f(_tau(), 2.0, 1e-3, 1.5, 1e-5, 1, 5.0, 1e4), "short_period"),
    # fcs_fit: fits
    Case(f"{FIT}:fcs_fit", _fit_by_string(_fit_2c, "fitfun_2c"), "string_two_components"),
    Case(f"{FIT}:fcs_fit", _fit_by_string(_fit_an, "fitfun_an"), "string_anomalous"),
    Case(f"{FIT}:fcs_fit", _fit_by_string(_fit_circ, "fitfun_circfcs"), "string_circular"),
    Case(f"{FIT}:fcs_fit", _fit_by_string(_fit_free_diffusion_2d, "fitfun_free_diffusion_2d"), "string_2d"),
    Case(f"{FIT}:fcs_fit", lambda f: _error(_fit_by_string(_fit_an_2c, "fcs_analytical_2c_anomalous"), f), "string_anomalous_model_wrong_signature"),
    Case(f"{FIT}:fcs_fit", lambda f: _fcs_fit_mem(f, True), "string_maximum_entropy"),
    Case(f"{FIT}:fcs_fit", lambda f: _fcs_fit_mem(f, False), "function_maximum_entropy"),
    Case(f"{FIT}:fcs_fit", _fit(_fit_fitfun, "fitfun"), "function_free_diffusion"),
    Case(f"{FIT}:fcs_fit", _fit(_fit_nanosecond, "fitfun_nanosecond_fcs"), "function_nanosecond"),
    Case(f"{FIT}:fcs_fit", _fit(_fit_reaction, "fitfun_uncoupled_reaction_diffusion"), "function_reaction"),
    Case(f"{FIT}:fcs_fit", _fit(_fit_an_2c, "fitfun_an_2c"), "function_anomalous_two_components"),
    Case(f"{FIT}:fcs_fit", _fit(_fit_finitelength, "fitfun_finitelength"), "function_finite_length"),
    Case(f"{FIT}:fcs_fit", _fcs_fit_weighted, "weights"),
    Case(f"{FIT}:fcs_fit", lambda f: _error(_fcs_fit_global_dual, f, "fitfun_dualfocus"), "function_dualfocus_2d_start_values"),
    Case(f"{FIT}:fcs_fit", _fit_dualfocus_function, "function_dualfocus"),
    Case(f"{FIT}:fcs_fit", lambda f: _fcs_fit_global_dual(f, "fcs_fit_dualfocus"), "function_fit_dualfocus_global"),
    Case(f"{FIT}:fcs_fit", _fcs_fit_dualfocus_string, "string_dualfocus"),
    Case(f"{FIT}:fcs_fit", lambda f: _error(_fcs_fit_2c_plot, f), "plot_single_curve"),
    Case(f"{FIT}:fcs_fit", _fcs_fit_all_fitted_plot, "plot_all_fitted"),
    Case(f"{FIT}:fcs_fit_dualfocus", lambda f: _fcs_fit_dualfocus_global(f, False), "global_diffusion"),
    Case(f"{FIT}:fcs_fit_dualfocus", lambda f: _fcs_fit_dualfocus_global(f, True), "global_concentration"),
    Case(f"{FIT}:fcs_fit_dualfocus", _fcs_fit_dualfocus_single, "single_curve"),
    Case(f"{FIT}:fcs_fit_dualfocus", _fcs_fit_dualfocus_no_bounds, "default_bounds"),
    Case(f"{FIT}:fit_corr", _fit_corr_default_bounds, "default_bounds"),
    Case(f"{FIT}:fit_corr", _fit_corr_single, "single_curve"),
    Case(f"{FIT}:fit_corr", _fit_corr_global, "global_curves"),
    Case(f"{FIT}:plot_fit", _plot_fit, "two_components"),
    Case(f"{FIT}:plot_fit", _plot_fit_single, "one_component"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ["spatialCorr"], 0, 3), "spatial_3"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ["spatialCorr"], 2, 5), "spatial_5_start"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ["crossCenterAv"], 1), "cross_center"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ["crossCenterAvN"], 0, 3), "cross_center_n"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ["twofocus5"], 1), "two_focus"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ["twofocus5Av"], 1), "two_focus_average"),
    Case(f"{FIT}:g2global_fit_struct", lambda f: f(_struct_obj(f), ""), "all_fields"),
    Case(f"{FIT}:chi_2_fcs_fit", _chi2),
    Case(f"{FIT}:stddev_2_weights", lambda f: f(_std()), "defaults"),
    Case(f"{FIT}:stddev_2_weights", lambda f: f(_std(), 0.2, 0.9), "clipped"),
    Case(f"{FIT}:make_fit_info_global_fit", _make_fit_info),
    Case(f"{FIT}:read_global_fit_result", _read_global_result, "all_fitted"),
    Case(f"{FIT}:read_global_fit_result", lambda f: _read_global_result(f, fitarray=(1, 0, 1, 0, 0, 0, 1, 0, 0), n_x=1 + 3 + 1), "some_fitted"),
    # mem_fit
    Case(f"{MEM}:Fitresultclass", _fitresult_class),
    Case(f"{MEM}:mem_fit", lambda f: f(_mem_param(), *_mem_curve()), "defaults"),
    Case(f"{MEM}:mem_fit", lambda f: f(_mem_param(), *_mem_curve(), weights=np.full(60, 0.1), startdistr=np.linspace(1, 2, 20)), "weights_and_start_distribution"),
    Case(f"{MEM}:mem_fit", lambda f: f(_mem_param(), *_mem_curve(), multiprocess=True), "threads"),
    Case(f"{MEM}:mem_fit", lambda f: f([20, 30, 1e-5, 2e-3, 0.8, 20, 5.0], *_mem_curve(), fitfun=_sib(f, "fcs_analytical", "fcs_analytical")), "anomalous"),
    Case(f"{MEM}:mem_fit_free_diffusion", lambda f: _error(f, _mem_param()[:5], *_mem_curve(3.0)), "default_shape"),
    Case(f"{MEM}:mem_fit_free_diffusion", lambda f: f(_mem_param(), *_mem_curve(), weights=np.full(60, 0.1)), "weights"),
    # filter_ap
    Case(f"{FAP}:filter_ap", lambda f: f(0, 3.0, 64, 0.028, False, False), "no_sted"),
    Case(f"{FAP}:filter_ap", lambda f: f(0.5, 3.0, 64, 0.028, True, False), "sted_normalized"),
    Case(f"{FAP}:filter_ap", lambda f: f(0, 3.0, 64, 0.028), "plot"),
    Case(f"{FAP}:filter_2c", lambda f: f(np.exp(-np.arange(64) / 3.0), np.ones(64), 0.1, False)),
    Case(f"{FAP}:filter_2c", lambda f: f(np.exp(-np.arange(64) / 3.0), np.exp(-np.arange(64) / 10.0), 0.5), "plot"),
    Case(f"{FAP}:filter_range", lambda f: f(_hist_obj(f)), "default"),
    # filter_g
    Case(f"{FG}:sort_g_array", lambda f: f(_sort_curves(2)), "two_curves"),
    Case(f"{FG}:sort_g_array", lambda f: f(_sort_curves(6)), "six_curves"),
    Case(f"{FG}:sort_g_array", lambda f: f(_sort_curves(6), False), "six_curves_not_optimized"),
    Case(f"{FG}:sort_g_array", lambda f: f(_sort_curves(253)), "optimized_over_250_curves"),
    Case(f"{FG}:sort_g_obj", lambda f: f(_good_chunks_obj(f), "sum5")),
    Case(f"{FG}:check_chunks_from_g_obj", lambda f: f(_good_chunks_obj(f)), "default"),
    Case(f"{FG}:check_chunks_from_g_obj", lambda f: f(_good_chunks_obj(f), "sum5", 1.0), "accept_all"),
    Case(f"{FG}:good_chunks_from_g_obj", lambda f: f(_good_chunks_obj(f)), "default"),
    Case(f"{FG}:good_chunks_from_g_obj", lambda f: f(_good_chunks_obj(f), "sum5", 0.5), "half"),
    # select_g
    Case(f"{SG}:Correlations", _empty_correlations),
    Case(f"{SG}:select_g", lambda f: f(_spatial_g_chunks(f)), "average"),
    Case(f"{SG}:select_g", lambda f: f(_spatial_g_chunks(f), "all"), "all"),
    # fcs_polar
    Case(f"{POL}:sig", lambda f: f(np.linspace(0, 1, 11), 0.5, 10.0)),
    Case(f"{POL}:g2polar", lambda f: f(_polar_g(4), 64), "four_directions"),
    Case(f"{POL}:g2polar", lambda f: f(_polar_g(6), 64), "six_directions"),
    Case(f"{POL}:g2polar", lambda f: f(_polar_g(5), 64), "unsupported_directions"),
    Case(f"{POL}:g2polar", lambda f: _error(f, np.ones(10), 64), "not_2d"),
    Case(f"{POL}:g2flow", lambda f: f(_polar_g(4), 64), "square"),
    Case(f"{POL}:g2flow", lambda f: f(_polar_g(4), 64, "airy"), "airy"),
    Case(f"{POL}:g2flow", lambda f: f(_polar_g(6), 64, "airy6"), "airy6"),
    Case(f"{POL}:g2flow", lambda f: _error(f, np.ones(10), 64), "not_2d"),
    Case(f"{POL}:g2polar_old", lambda f: f(correlation_curve()[1][:, 0]), "defaults"),
    Case(f"{POL}:g2polar_old", lambda f: f(correlation_curve()[1][:, 1], 5), "smoothing"),
    # fcs2difftime
    Case(f"{DT}:g2difftime", lambda f: f(_curve_2d()), "defaults"),
    Case(f"{DT}:g2difftime", lambda f: f(_curve_2d(), 3), "smoothing"),
    # imsd
    Case(f"{IMSD}:fcs2imsd", lambda f: f(_gauss_stack(), np.arange(6) * 1e-5, np.array([1, 1, 0, 0], bool), np.array([1.0, 1.0, 2.0, 50.0]), np.array([-1e9] * 4), np.array([1e9] * 4)), "arrays"),
    Case(f"{IMSD}:fcs2imsd", lambda f: f(_gauss_stack(), np.arange(6) * 1e-5, np.array([1, 1, 0, 0], bool), np.array([1.0, 1.0, 2.0, 50.0]), np.array([-1e9] * 4), np.array([1e9] * 4), True, False), "outliers_kept_afterpulsing"),
    Case(f"{IMSD}:fcs2imsd", lambda f: _error(f, _gauss_stack(), np.arange(6) * 1e-5, [1, 1, 0, 0], [1.0, 1.0, 2.0, 50.0], [-1e9] * 4, [1e9] * 4), "lists"),
    # two_focus_fcs
    Case(f"{TF}:two_focus_fcs", lambda f: f(*_two_focus_args()), "array_tau"),
    Case(f"{TF}:two_focus_fcs", lambda f: f(np.float64(1e-3), *_two_focus_args()[1:]), "scalar_tau"),
    Case(f"{TF}:simulate_two_focus_cross_center", _simulate_cross_center),
    Case(f"{TF}:simulate_spatialcorr", lambda f: _simulate_spatialcorr(f, [1e-4, 1e-3]), "list_tau"),
    Case(f"{TF}:simulate_spatialcorr", lambda f: _simulate_spatialcorr(f, 1e-3), "float_tau"),
    # w0_from_fingerprint
    Case(f"{W0}:calc_fingerprint", lambda f: f(_fingerprint_param(), 40), "centered"),
    Case(f"{W0}:calc_fingerprint", lambda f: f(np.array([5.0, -3.0, 0.2, 500.0, 1.2, 50.0, 20.0]), 40), "shifted"),
    Case(f"{W0}:fitfun_w0_from_fingerprint", _fingerprint_residual),
    Case(f"{W0}:get_w0_from_fingerprint", lambda f: _fingerprint_fit(f, np.array([0, 0, 1, 0, 0, 0, 0]), {2: 0.22}), "waist"),
    Case(f"{W0}:get_w0_from_fingerprint", lambda f: _fingerprint_fit(f, np.array([1, 1, 1, 0, 0, 0, 0]), {2: 0.22}, shift=(8.0, -5.0)), "waist_and_position"),
    # distance2detelements
    Case(f"{DIST}:distance2detelements", lambda f: f(12, 0), "diagonal"),
    Case(f"{DIST}:distance2detelements", lambda f: f(7, 12), "neighbour"),
    Case(f"{DIST}:spad_coord_from_det_numb", lambda f: f(17)),
    Case(f"{DIST}:spad_shift_vector_crosscorr", lambda f: f(), "default"),
    Case(f"{DIST}:spad_shift_vector_crosscorr", lambda f: f([1, 0]), "down"),
    Case(f"{DIST}:spad_shift_vector_crosscorr", lambda f: f([-1, 2], 3), "central_3x3"),
    # detectors
    Case(f"{DET}:detector_element_coordinates", lambda f: f("airyscan"), "airyscan"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("Luminosa", 20), "luminosa_element"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("pda23"), "pda23"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("nsparc"), "nsparc"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("5x5", 7), "5x5_element"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("gi"), "gi"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("prism", 3), "prism_element"),
    Case(f"{DET}:detector_element_coordinates", lambda f: f("unknown"), "unknown"),
    Case(f"{DET}:distance_between_elements", lambda f: f("airyscan", 0, 7), "airyscan"),
    Case(f"{DET}:distance_between_elements", lambda f: f("luminosa", 9, 20), "luminosa"),
    Case(f"{DET}:distance_between_elements", lambda f: f("nsparc", 0, 24), "nsparc"),
    Case(f"{DET}:distance_between_elements", lambda f: f("gi", 0, 48), "gi"),
    Case(f"{DET}:distance_between_elements", lambda f: _error(f, "unknown", 0, 1), "unknown"),
    # spad_shifts and get_det_elem_from_array
    Case(f"{SHIFTS}:spad_shifts", lambda f: f()),
    Case(f"{GDE}:get_el_ring_array", lambda f: f(5, 0), "ring_0"),
    Case(f"{GDE}:get_el_ring_array", lambda f: f(5, 1), "ring_1"),
    Case(f"{GDE}:get_el_ring_array", lambda f: f(5, 2), "ring_2"),
    Case(f"{GDE}:get_el_ring_array", lambda f: f(7, 3), "7x7"),
    Case(f"{GDE}:get_elsum", lambda f: f(5, 1), "3x3"),
    Case(f"{GDE}:get_elsum", lambda f: f(5, 2), "5x5"),
    Case(f"{GDE}:get_elsum", lambda f: f(7, 0), "centre_only"),
]


SKIPPED: dict[str, str] = {
    f"{A2CP}:atimes_file_2_corr": SKIP_FILE_IO,
    f"{F2C}:fcs_load_and_corr_split": SKIP_FILE_IO,
    f"{F2C}:fcs_sparse_matrices": SKIP_FILE_IO,
    f"{F2C}:fcs_bin_to_csv_all": SKIP_FILE_IO,
    f"{F2C}:plot_cross_center_scheme": SKIP_PLOT_ONLY,
    f"{F2C}:plot_fcs_correlations": SKIP_PLOT_ONLY,
    f"{F2C}:plot_g_surf": SKIP_PLOT_ONLY,
}
