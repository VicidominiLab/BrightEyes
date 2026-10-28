import contextlib
import importlib
import io
from unittest import mock

import joblib
import numpy as np

from .._compare import assert_same
from .._inputs import decay_histogram, photon_arrival_times, time_trace
from .._registry import SKIP_FILE_IO, SKIP_GUI, SKIP_PLOT_ONLY, Case

PCH = "brighteyes_ffs.pch"
A2P = f"{PCH}.atimes2pch"
D2P = f"{PCH}.data2pch"
GPS = f"{PCH}.generate_psf"
PFIT = f"{PCH}.pch_fit"
SPC = f"{PCH}.simulate_pch"

T = "brighteyes_ffs.tools"
AFR = f"{T}.analyze_fit_result"
A2T = f"{T}.array2tiff"
A2V = f"{T}.array2video"
BIN = f"{T}.bindata"
CCO = f"{T}.calcconc"
CDC = f"{T}.calcdist_from_coord"
CAD = f"{T}.cast_data"
CHK = f"{T}.checkfname"
CLF = f"{T}.closefile"
C2D = f"{T}.cmap2d"
CMS = f"{T}.cmaps"
CFM = f"{T}.color_from_map"
CON = f"{T}.constants"
C2A = f"{T}.csv2array"
DEL = f"{T}.draw_ellipsoid"
EXP = f"{T}.exp_psf"
FNE = f"{T}.find_nearest"
FCU = f"{T}.fit_curve"
FDA = f"{T}.fit_data"
FDO = f"{T}.fit_doughnut2d"
FG2 = f"{T}.fit_gauss_2d"
FP2 = f"{T}.fit_polynomial_2d"
FPS = f"{T}.fit_psf"
GTO = f"{T}.goto"
H5D = f"{T}.h5data"
LSF = f"{T}.list_files"
MAV = f"{T}.moving_average"
N2H = f"{T}.np2hdf5"
P2F = f"{T}.path2fname"
PHA = f"{T}.phasor"
PLC = f"{T}.plot_colors"
PRT = f"{T}.print_tools"
SVF = f"{T}.savefigure"
SVV = f"{T}.savevar"
STE = f"{T}.stokes_einstein"
T2S = f"{T}.time2string"

G = "brighteyes_ffs.fcs_gui"
ANS = f"{G}.analysis_settings"
CFC = f"{G}.correlation_functions_class"
FMC = f"{G}.fitmodel_class"
LFM = f"{G}.load_ffs_metadata"
RFS = f"{G}.read_ffs"
RES = f"{G}.restore_session"
S2E = f"{G}.session_to_excel"
TTE = f"{G}.timetrace_end"


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


def _fcs(obj, module: str, name: str):
    """Return `name` from the `fcs` module `module` of the package that `obj` lives in."""
    root = obj.__module__.rsplit(".", 2)[0]
    return getattr(importlib.import_module(f"{root}.fcs.{module}"), name)


def _printed(function, *args, **kwargs) -> tuple[object, str]:
    """Return the result of a call together with the text it printed."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def _serial():
    """Run joblib workers in the calling process so that the seeded generators are used."""
    return joblib.parallel_config(backend="sequential")


def _psf(n: int = 8, w: float = 2.5, wz: float = 3.5) -> np.ndarray:
    """Return a Gaussian focal volume on an `n`-cube with maximum 1."""
    z, y, x = np.indices((n, n, n)) - n // 2
    return np.exp(-2 * (x**2 + y**2) / w**2 - 2 * z**2 / wz**2)


def _psf_stack() -> np.ndarray:
    return np.stack([_psf(8, 2.5, 3.5), _psf(8, 3.5, 4.5)])


def _pch(f, c=(0.3,), q=(3.0,), k_max=20, bg=0.0, psf=None) -> np.ndarray:
    """Return a histogram made with the simulate_pch_nc of the package under test."""
    simulate = _sib(f, "simulate_pch", "simulate_pch_nc")
    return simulate(
        _psf() if psf is None else psf, k_max=k_max, c=list(c), q=list(q), bg=bg
    )


def _compressed_psf(f, n_bins: int = 1000):
    generate = _sib(f, "pch_fit", "generate_psfs")
    return generate(_psf(), 1, n_bins)


def _global_problem(f, n_comp: int = 1):
    """Return histograms, parameters and PSFs for a global fit of two histograms."""
    psfs = _psf_stack()
    simulate = _sib(f, "simulate_pch", "simulate_pch_nc")
    hist = np.stack(
        [
            simulate(
                psfs[0], k_max=20, c=[0.3] * n_comp, q=[3.0, 1.5][:n_comp], bg=0.0
            ),
            simulate(
                psfs[1], k_max=20, c=[0.3] * n_comp, q=[3.0, 1.5][:n_comp], bg=0.0
            ),
        ],
        axis=1,
    )
    return hist, psfs


def _global_params(n_comp: int = 1):
    n_param = 2 * n_comp + 3
    param = np.ones((n_param, 2))
    param[:n_comp] = 0.5
    param[n_comp : 2 * n_comp] = (
        [[2.0, 2.5], [1.0, 1.2]][:n_comp] if n_comp == 2 else [[2.0, 2.5]]
    )
    param[2 * n_comp] = 0.0
    return param


def _atimes_pch(f, macrotime: float = 1e-5):
    """Arrival-time object with float macrotimes (about 10 s) and 25 channels."""
    obj = _fcs(f, "atimes_data", "ATimesData")()
    obj.macrotime = macrotime
    for name, array in photon_arrival_times().items():
        setattr(obj, name, array.astype(float))
    return obj


def _t0() -> np.ndarray:
    return photon_arrival_times()["det3"][:, 0].astype(float) * 1e-5


_NC_INFO = np.array([1, 1, 0, 0, 0])
_NC_LOWER = [1e-10, 1e-10, 0, 0, 0]
_NC_UPPER = [1e10, 1e10, 1e10, 1e10, 1e10]


def _fit_single(f, minimization, fitfun="fitfun_pch", n_draws=1, **kwargs):
    fit = _sib(f, "pch_fit", "fit_pch")
    hist = _pch(f)
    if fitfun == "fitfun_pch":
        options = {"fit_info": [1, 1, 0, 0], "param": [0.5, 2.0, 1.0, 1.0]}
    else:
        options = {
            "fit_info": _NC_INFO,
            "param": [0.5, 2.0, 0.0, 1.0, 1.0],
            "lBounds": _NC_LOWER,
            "uBounds": _NC_UPPER,
        }
    options.update(kwargs)
    return fit(
        hist,
        psf=_psf(),
        n_bins=1000,
        fitfun=fitfun,
        minimization=minimization,
        n_draws=n_draws,
        **options,
    )


def _fit_mle_bounds():
    return {"lBounds": [1e-6, 1e-6, 0, 0, 0], "uBounds": [1e3, 1e3, 1e3, 1e3, 1e3]}


def _fit_two(f, minimization):
    fit = _sib(f, "pch_fit", "fit_pch")
    hist = _pch(f, c=(0.2, 0.2), q=(3.0, 1.0))
    options = {}
    if minimization == "mle":
        options = {"lBounds": [1e-6] * 7, "uBounds": [1e3] * 7}
    else:
        options = {"lBounds": [1e-10] * 7, "uBounds": [1e10] * 7}
    return fit(
        hist,
        fit_info=np.array([1, 0, 1, 0, 0, 0, 0]),
        param=[0.3, 0.2, 2.5, 1.0, 0.0, 1.0, 1.0],
        psf=_psf(),
        n_bins=1000,
        fitfun="fitfun_pch_nc",
        minimization=minimization,
        **options,
    )


def _fit_global(f, minimization, n_comp=1, global_param=None, fit_info=None):
    fit = _sib(f, "pch_fit", "fit_pch")
    hist, psfs = _global_problem(f, n_comp)
    n_param = 2 * n_comp + 3
    if fit_info is None:
        fit_info = np.zeros(n_param)
        fit_info[: 2 * n_comp] = 1
    if global_param is None:
        global_param = [True] * n_comp + [False] * n_comp + [True] * 3
    low, high = (1e-6, 1e3) if minimization == "mle" else (1e-10, 1e10)
    return fit(
        hist,
        fit_info=fit_info,
        param=_global_params(n_comp),
        psf=psfs,
        lBounds=[low] * n_param,
        uBounds=[high] * n_param,
        n_bins=1000,
        global_param=global_param,
        fitfun="fitfun_pch_nc_global",
        minimization=minimization,
    )


def _mc_psf() -> np.ndarray:
    return _psf(8, 2.5, 3.5) / _psf(8, 2.5, 3.5).sum()


def _same_minimize(old, new) -> None:
    """Compare optimizer results, taking `hess_inv` as the dense matrix it stands for."""
    old, new = dict(old), dict(new)
    old_hess, new_hess = old.pop("hess_inv", None), new.pop("hess_inv", None)
    assert_same(old, new)
    if old_hess is not None:
        assert_same(old_hess.todense(), new_hess.todense())


def _run_serial(f, *args):
    with _serial():
        return f(*args)


def _global_fun(f, minimization, return_residuals=False, global_param=True):
    hist, psfs = _global_problem(f)
    psf_compressed, psf_weights = _sib(f, "pch_fit", "generate_psfs")(psfs, 2, 500)
    info = np.array([1, 1, 0, 0, 0])
    log = minimization != "mle"
    if global_param:
        shared = [True, False, True, True, True]
        fitparam = np.array([0.0, 0.3, 0.4] if log else [0.3, 3.0, 3.5])
        fixed = np.array([0.0, 1.0, 1.0])
    else:
        shared = None
        fitparam = np.array([-0.5, -0.4, 0.3, 0.4] if log else [0.3, 0.4, 3.0, 3.5])
        fixed = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    args = (
        fitparam,
        fixed,
        info,
        hist,
        psf_compressed,
        shared,
        psf_weights,
        1,
        minimization,
    )
    if minimization == "mle":
        return f(*args, return_residuals)
    return f(*args)


def _xy(n: int = 40, seed: int = 11):
    x = np.linspace(0.5, 20.0, n)
    return x, np.random.default_rng(seed)


def _power_law():
    x, rng = _xy()
    return x, 3.0 * x**-0.7 + 0.1 + 0.01 * rng.standard_normal(x.size)


def _exp_decay(n_comp: int = 1):
    x, rng = _xy()
    y = 2.0 * np.exp(-0.4 * x) + 0.3 + 0.01 * rng.standard_normal(x.size)
    if n_comp == 2:
        y = y + 1.0 * np.exp(-2.0 * x)
    return x, y


def _yx(maker, *args):
    x, y = maker(*args)
    return y, x


def _g_object(n: int, with_dwell_time: bool = False):
    """Object holding `n` correlation curves `det<i>_average` (or `central`, `sum3`, ...)."""
    obj = type("G", (), {})()
    x = np.logspace(-6, -1, 30)
    rng = np.random.default_rng(12)
    for i in range(n):
        name = f"det{i}_average" if n > 10 else ["central", "sum3", "sum5"][i]
        y = (2.0 + 0.1 * i) * x**-0.5 * 1e-3 + 0.01 + 1e-4 * rng.standard_normal(x.size)
        setattr(obj, name, np.column_stack((x, y)))
    if with_dwell_time:
        obj.dwellTime = 1e-6
    return obj


def _gauss_image(f):
    noise = 0.01 * np.random.default_rng(13).standard_normal((12, 10))
    return _sib(f, "fit_gauss_2d", "normal2d")(10, 12, 4.2, 5.1, 2.0, 1.8, 0.2) + noise


def _doughnut_image(f):
    noise = 0.01 * np.random.default_rng(13).standard_normal((12, 10))
    return (
        _sib(f, "fit_doughnut2d", "doughnut2d")(10, 12, 4.2, 5.1, 2.0, 1.8, 0.2) + noise
    )


def _bead_image(
    n: int = 40, centers=((20.0, 20.0), (22.0, 19.0)), planes: int = 0
) -> np.ndarray:
    y, x = np.mgrid[0:n, 0:n].astype(float)
    image = np.zeros((n, n))
    for cy, cx in centers:
        image += 200 * np.exp(-2 * ((x - cx) ** 2 + (y - cy) ** 2) / 3.0**2)
    if planes:
        z = np.arange(planes) - planes // 2
        image = image[:, :, None] * np.exp(-2 * z**2 / 2.0**2)
    return image + np.random.default_rng(14).random(image.shape)


def _beads_for_particles(channels: int = 0) -> np.ndarray:
    y, x = np.mgrid[0:64, 0:64].astype(float)
    image = np.zeros((64, 64))
    for cy, cx in ((20, 20), (20, 45), (45, 30), (4, 4)):
        image += 200 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 2.5**2))
    if channels:
        return np.stack([image * (1 + 0.1 * c) for c in range(channels)], axis=2)
    return image


def _patched_ginput(function, image, **kwargs):
    with mock.patch(
        "matplotlib.pyplot.ginput", return_value=[(20.0, 20.0), (22.0, 19.0)]
    ):
        return function(image, **kwargs)


def _lifetime_and_intensity():
    rng = np.random.default_rng(15)
    return rng.random((8, 8)) * 4 + 1, rng.random((8, 8)) * 100


def _same_without_date(old, new) -> None:
    """Compare the parameter objects without the recording time, which is read from the clock."""
    old_vars, new_vars = dict(vars(old)), dict(vars(new))
    assert len(old_vars.pop("RecordingDate")) == len(new_vars.pop("RecordingDate")) == 1
    assert type(old).__name__ == type(new).__name__
    assert_same(old_vars, new_vars)


def _ans(f, name: str):
    return _sib(f, "analysis_settings", name)


def _model(f, **override):
    """Fit model with six parameters of which parameters 0, 1 and 3 are used."""
    params = {
        "model": "Free diffusion",
        "shortlabel": "3D",
        "paramNames": ["N", "tauD", "offset"],
        "paramFittable": [1, 1, 1],
        "paramDefvalues": [1.0, 1e-3, 0.0],
        "allparamDefvalues": [1.0, 1e-3, 5.0, 0.0, 1.0, 1.0],
        "paramFactors10": [1.0, 1e-6, 1.0],
        "paramMinbound": [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
        "paramMaxbound": [1e3, 1e3, 50.0, 1.0, 10.0, 10.0],
        "fitfunctionName": "fcs_analytical",
        "fitfunctionParamUsed": [0, 1, 3],
    }
    params.update(override)
    model = _sib(f, "fitmodel_class", "FitModel")()
    model.set_params(params)
    return model


def _mem_model(f):
    """Fit model for the maximum entropy method: five parameters after five distribution points."""
    return _model(
        f,
        model="Maximum entropy method free diffusion",
        paramNames=["a", "b", "c", "d", "e"],
        allparamDefvalues=[0.5] * 10,
        paramFactors10=[1.0, 2.0, 1.0, 1.0, 1.0],
        paramMinbound=[0.0] * 10,
        paramMaxbound=[1e3] * 10,
        fitfunctionParamUsed=[0, 1, 2, 3, 4],
    )


def _corr_function(
    f, elements=("central", "sum3", "sum5"), list_of_g=None, mode="Pair-correlation fcs"
):
    function = _sib(f, "correlation_functions_class", "CorrelationFunction")()
    function.set_params(
        {
            "mode": mode,
            "elements": list(elements),
            "listOfG": list(elements) if list_of_g is None else list_of_g,
            "shortlabel": "PairCorr",
            "average": None,
        }
    )
    return function


def _tau_gui() -> np.ndarray:
    return np.logspace(-6, -1, 20)


def _corrs_attrs(f, elements=("central", "sum3", "sum5"), n_chunks=4):
    """Correlations object holding `<element>` and `<element>_chunk<k>` arrays."""
    corrs = _fcs(f, "fcs2corr", "Correlations")()
    rng = np.random.default_rng(21)
    tau = _tau_gui()
    for i, element in enumerate(elements):
        chunks = [
            np.column_stack(
                (
                    tau,
                    (1 + 0.1 * i)
                    / (1 + tau / 1e-3)
                    * (1 + 0.02 * rng.standard_normal(tau.size)),
                )
            )
            for _ in range(n_chunks)
        ]
        setattr(corrs, element, np.mean(chunks, 0))
        for index, chunk in enumerate(chunks):
            setattr(corrs, f"{element}_chunk{index}", chunk)
    corrs.list_of_g_out = list(elements)
    return corrs


def _corrs_chunks(f, names=("central", "sum3"), n_chunks=4, good=(0, 1, 3)):
    """Correlations object holding chunk arrays under each name."""
    corrs = _fcs(f, "fcs2corr", "Correlations")()
    rng = np.random.default_rng(22)
    tau = _tau_gui()
    for i, name in enumerate(names):
        chunks = np.zeros((n_chunks, tau.size, 2))
        chunks[:, :, 0] = tau
        chunks[:, :, 1] = (
            (1 + 0.1 * i)
            / (1 + tau / 1e-3)
            * (1 + 0.02 * rng.standard_normal((n_chunks, tau.size)))
        )
        corrs.add_corr_chunks(name, chunks)
        corrs.list_of_g_out.append(name)
    corrs.good_chunks = list(good)
    return corrs


def _start_values(n_curves: int) -> np.ndarray:
    return np.array(
        [[1.0 + 0.1 * i, 1e-3 * (i + 1), 0.2 + 0.01 * i] for i in range(n_curves)]
    ).T


def _corr_fit(
    f, names=("central", "sum3", "sum5"), model=None, fitresults=None, fitrange=(2, 12)
):
    """CorrFit with one fitted curve per name and the residuals of the fits filled in."""
    fit = _ans(f, "CorrFit")()
    fit.add_fit_analyses(
        "Pair-correlation fcs",
        _model(f) if model is None else model,
        [1, 1, 0, 1],
        _start_values(len(names)),
        list(fitrange),
        list(names),
    )
    for i, curve in enumerate(fit.fit_all_curves):
        result = (
            np.linspace(0, 0.05, 10) + 0.01 * i if fitresults is None else fitresults(i)
        )
        curve.update(fitresult=result, w0=250.0 + i, D=1.5 + i)
    return fit


def _direction_fit(f, names, length=20):
    return _corr_fit(
        f, names, fitresults=lambda i: np.linspace(1.0, 2.0, length) * (1 + 0.1 * i)
    )


def _mfda_fit(f, vertical_horizontal):
    names = [f"V{v}_H{h}" for v, h in vertical_horizontal]
    return _corr_fit(
        f, names, fitresults=lambda i: np.array([1e-3 * (i + 1), 0.5 + 0.01 * i])
    )


def _ffs_file(f, with_corrs=True, with_fit=True):
    """FFSfile with one analysis, its correlations and a fit of three curves."""
    metadata = _ans(f, "FFSmetadata")(
        num_pixels=8,
        num_lines=6,
        duration=20.0,
        dwelltime=1e-3,
        pxsize=0.05,
        range_x=10.0,
        coords=[3, 4],
    )
    ffs = _ans(f, "FFSfile")("a.h5", "label", metadata, timetrace=time_trace()[:200])
    ffs.add_analysis(_corr_function(f), 10, 5.0, "multipletau", "allon")
    analysis = ffs.get_analysis()
    if with_corrs:
        analysis.update(corrs=_corrs_attrs(f))
        analysis.settings.update(
            chunks_off=analysis.settings.chunks_off, analysis=analysis
        )
    if with_fit:
        analysis.add_fit_analysis(_model(f), [1, 1, 0, 1], _start_values(3), [2, 12])
        for i, curve in enumerate(analysis.fits[0].fit_all_curves):
            curve.update(fitresult=np.linspace(0, 0.05, 10) + 0.01 * i, w0=250.0, D=1.5)
    return ffs


def _lib_workflow(f):
    image = np.arange(48.0).reshape(6, 8)
    lib = f(date_created="created", date_modified="modified")
    lib.add_image(image, "one.tif")
    lib.add_image(image.T.copy(), "two.tif", mdata=_ans(f, "FFSmetadata")(range_x=10.0))
    lib.add_random_image()
    lib.active_image = 1
    seen = [
        lib.num_images,
        lib.get_image(),
        lib.get_image(10),
        lib.get_image(-1),
        lib.get_image_name(0),
    ]
    seen += [lib.return_image(1), lib.return_image(), lib.get_image("active")]
    lib.remove_image()
    seen += [lib.num_images, lib.active_image]
    lib.remove_image(5)
    lib.remove_image(0)
    lib.remove_image(0)
    seen += [
        lib.num_images,
        lib.active_image,
        lib.get_image(),
        lib.get_image_name(0),
        lib.get_image(0),
    ]
    return tuple(seen), lib


def _image_workflow(f):
    image = f()
    seen = [image.num_files, image.get_ffs_file(), image.print_image_metadata()]
    image.change_image(np.ones((5, 7)), "a.tif")
    image.change_image(
        np.ones((4, 6)),
        "b.tif",
        _ans(f, "FFSmetadata")(range_x=20.0, dwelltime=1e-3, pxsize=0.1),
    )
    seen += [image.print_image_metadata()]
    image.update("c.tif", np.zeros((2, 2)))
    image.update(ffs_list=[_ffs_file(f, False, False)])
    image.add_ffs_file(_ffs_file(f, False, False))
    seen += [
        image.num_files,
        image.get_ffs_file(),
        image.get_ffs_file(0),
        image.get_ffs_file(5),
    ]
    image.active_ffs = 1
    seen += [image.get_ffs_file("active")]
    image.remove_ffs_file(0)
    image.remove_ffs_file(9)
    seen += [image.num_files]
    return tuple(seen), image


def _file_properties(f):
    ffs = _ffs_file(f, False, False)
    seen = [ffs.num_analyses, ffs.coords, ffs.duration, ffs.number_of_elements]
    ffs.timetrace = time_trace()[:50, 0]
    seen += [ffs.number_of_elements]
    ffs.timetrace = None
    seen += [ffs.number_of_elements]
    ffs.update(
        label="new",
        coords=[1, 2],
        timetrace=time_trace()[:10],
        airy=np.ones(6),
        active_analysis="None",
    )
    ffs.update(active_analysis=0)
    return tuple(seen), ffs


def _file_analyses(f):
    ffs = _ffs_file(f, False, False)
    ffs.add_analysis(
        _corr_function(f, list_of_g=["crossAll"]), 5, 10.0, "multipletau", [1, 0], 0
    )
    ffs.add_analysis(_corr_function(f), chunksize=25.0, algorithm="tt2corr")
    seen = [
        ffs.num_analyses,
        ffs.active_analysis,
        ffs.get_analysis(),
        ffs.get_analysis(0),
        ffs.get_analysis(9),
    ]
    ffs.remove_analysis(1)
    ffs.remove_analysis(9)
    seen += [ffs.num_analyses, ffs.active_analysis]
    ffs.remove_analysis(0)
    ffs.remove_analysis(0)
    seen += [ffs.num_analyses, ffs.active_analysis]
    return tuple(seen), ffs


def _file_copy(f):
    ffs = _ffs_file(f)
    ffs.copy_correlation(0)
    ffs.copy_correlation(7)
    return (
        ffs.num_analyses,
        ffs.get_analysis(1).num_fits,
        ffs.get_analysis(0).num_fits,
        ffs,
    )


def _file_fit_as_data(f):
    ffs = _ffs_file(f)
    ffs.use_fit_as_data(0, 0)
    ffs.use_fit_as_data(7, 0)
    ffs.use_fit_as_data(0, 4)
    return (
        ffs.num_analyses,
        ffs.get_analysis(1).corrs,
        ffs.get_analysis(1).settings,
        ffs,
    )


def _file_fit_as_data_free(f):
    ffs = _ffs_file(f)
    for curve in ffs.get_analysis().fits[0].fit_all_curves:
        curve.fitfunction_label = "Mean squared displacement"
    ffs.use_fit_as_data(0, 0)
    return ffs.num_analyses, ffs.get_analysis(1).corrs, ffs


def _corr_workflow(f):
    function = _corr_function(f)
    analysis = f()
    seen = [analysis.calc_corr()]
    analysis.mode = "m"
    seen += [analysis.analysis_summary()]
    analysis.analysis(function, 10, 5.0, np.ones(4), "multipletau")
    seen += [
        analysis.mode,
        analysis.settings,
        analysis.n_curves_mode,
        analysis.active_fit,
        analysis.analysis_summary(),
    ]
    analysis.analysis(
        _corr_function(f, list_of_g=["crossAll"]), 10, 5.0, np.ones(4), "multipletau", 2
    )
    seen += [analysis.settings.algorithm, analysis.active_fit]
    analysis.analysis(
        _corr_function(f, list_of_g=["crossAll"]), 10, 5.0, np.ones(4), "tt2corr"
    )
    seen += [
        analysis.settings.algorithm,
        analysis.num_chunks(20.0),
        analysis.num_chunks(None),
        analysis.corr_param(),
    ]
    analysis.update(corrs=_corrs_attrs(f))
    seen += [analysis.calc_corr()]
    analysis.update("mode2", analysis.settings, None, [1, 2, 3], "None")
    seen += [analysis.mode, analysis.fits, analysis.active_fit]
    analysis.update(active_fit=2)
    seen += [
        analysis.active_fit,
        analysis.return_fit_obj(),
        analysis.return_fit_obj(0),
        analysis.return_fit_obj(9),
    ]
    analysis.remove_fit(1)
    seen += [analysis.fits, analysis.active_fit]
    analysis.remove_fit(9)
    analysis.remove_fit(0)
    analysis.remove_fit(0)
    seen += [analysis.fits, analysis.active_fit]
    return tuple(seen), analysis


def _corr_getters(f):
    analysis = f()
    seen = [analysis.get_corr(), analysis.get_corr3D()]
    analysis.update(corrs=_corrs_chunks(f))
    seen += [analysis.get_corr(), analysis.get_corr("sum3"), analysis.get_corr3D()]
    analysis.update(corrs=_corrs_attrs(f))
    seen += [analysis.get_corr("central"), analysis.get_corr("sum5_chunk2")]
    cross = _corrs_chunks(f, [f"det12x{j}" for j in range(25)], 4, (0, 1, 3))
    analysis.update(corrs=cross)
    seen += [
        analysis.get_corr("crossCenterAv"),
        analysis.get_corr("crossCenterAv_chunk1"),
    ]
    full = _corrs_chunks(f, [f"V{v}_H{h}" for v in range(-2, 3) for h in range(-2, 3)])
    analysis.update(corrs=full)
    seen += [analysis.get_corr3D()]
    analysis.update(
        corrs=_corrs_chunks(
            f, [f"V{v}_H{h}" for v in range(-1, 2) for h in range(-1, 2)]
        )
    )
    seen += [analysis.get_corr3D()]
    return tuple(seen), analysis


def _fit_management(f):
    analysis = f(mode="Pair-correlation fcs")
    analysis.settings.elements = ["central", "sum3", "sum5"]
    start = _start_values(3)
    analysis.add_fit_analysis(_model(f), [0, 0, 0, 0], start)
    seen = [analysis.num_fits, analysis.active_fit]
    analysis.add_fit_analysis(_model(f), [1, 0, 0, 0], start, [2, 12])
    seen += [analysis.num_fits, analysis.active_fit]
    analysis.add_fit_analysis(_model(f, model="Flow heat map"), [0, 0, 0, 0], start)
    seen += [analysis.num_fits, analysis.active_fit]
    analysis.update_fit_analysis(_model(f), [0, 0, 0, 0], start)
    seen += [analysis.fits[1].num_fitcurves]
    analysis.update_fit_analysis(_model(f), [1, 1, 0, 0], start * 2, [3, 8], 1)
    seen += [analysis.fits[1].fit_all_curves, analysis.fits[1].fitrange()]
    analysis.update_fit_analysis(
        _model(f, model="Asymmetry heat map"), [0, 0, 0, 0], start, [2, 9], -1
    )
    seen += [analysis.fits[1].fitrange()]
    return tuple(seen), analysis


def _settings_workflow(f):
    ffs = _ffs_file(f, True, False)
    analysis = ffs.get_analysis()
    settings = _ans(f, "CorrSettings")(
        ["central"], ["central"], 10, "multipletau", 5.0, np.ones(4), None
    )
    seen = [settings.elements, settings.average]
    settings.update(["a", "b"], 20, 2.5, None, None, "tt2corr", ["a", "b"])
    seen += [
        settings.elements,
        settings.resolution,
        settings.chunksize,
        settings.algorithm,
        settings.list_of_g,
    ]
    settings.update(chunks_off=np.array([1, 0, 1, 1]), analysis=analysis)
    seen += [settings.chunks_off, analysis.corrs.good_chunks]
    return tuple(seen), settings, _ans(f, "CorrSettings")()


def _corr_fit_workflow(f):
    empty = _ans(f, "CorrFit")()
    seen = [
        empty.num_fitcurves,
        empty.return_field("w0"),
        empty.return_field("D"),
        empty.return_field("fitAllCurves"),
    ]
    seen += [empty.return_all("fitrange"), empty.fitrange()]
    fit = _corr_fit(f)
    seen += [
        fit.num_fitcurves,
        fit.return_field("w0"),
        fit.return_field("D"),
        fit.return_field("fitAllCurves"),
    ]
    seen += [fit.return_all("data"), fit.return_all("fitrange"), fit.fitrange()]
    seen += [fit.fitresults(), fit.fitresults("array")]
    return tuple(seen), fit


def _corr_fit_maps(f):
    square = ["Right", "Up", "Left", "Down"]
    airy6 = ["Right", "UpRight", "UpLeft", "Left", "DownLeft", "DownRight"]
    seen = [_direction_fit(f, square).fitresults_asymmetrymap()]
    seen += [_direction_fit(f, airy6).fitresults_asymmetrymap()]
    seen += [
        _direction_fit(
            f, ["Right", "Up", "Down", "Left"][:3] + ["Other"]
        ).fitresults_asymmetrymap()
    ]
    seen += [_direction_fit(f, ["a", "b", "c"]).fitresults_asymmetrymap()]
    unfitted = _corr_fit(f, square, fitresults=lambda i: None)
    seen += [unfitted.fitresults_asymmetrymap(), unfitted.fitresults_flowmap()]
    seen += [_direction_fit(f, square).fitresults_flowmap()]
    seen += [
        _direction_fit(f, ["V0_H1", "Vn1_H0", "V0_Hn1", "V1_H0"]).fitresults_flowmap()
    ]
    seen += [
        _direction_fit(
            f, ["UpRight", "UpLeft", "DownLeft", "DownRight"]
        ).fitresults_flowmap()
    ]
    seen += [
        _direction_fit(f, [f"Angle{a}" for a in range(0, 360, 60)]).fitresults_flowmap()
    ]
    seen += [_direction_fit(f, ["a", "b", "c", "d"]).fitresults_flowmap()]
    return tuple(seen)


def _corr_fit_mfda(f):
    all_pairs = [(v, h) for v in range(-2, 3) for h in range(-2, 3)]
    some = [(v, h) for v, h in all_pairs if v == 0]
    names = [(0, 0), (1, 1)]
    seen = [
        _mfda_fit(f, all_pairs).fitresults_mfda(),
        _mfda_fit(f, some).fitresults_mfda(),
    ]
    seen += [_corr_fit(f, ["x", "y"]).fitresults_mfda()]
    single = _corr_fit(f, ["V0_H0"], fitresults=lambda i: np.array([2e-3]))
    seen += [single.fitresults_mfda(), _mfda_fit(f, names).fitresults_mfda()]
    return tuple(seen)


def _corr_fit_mem(f):
    tau = np.logspace(-6, -1, 30)
    fit = _ans(f, "CorrFit")()
    fit.add_fit_analyses(
        "mode",
        _mem_model(f),
        [1, 1, 0, 1, 1, 1],
        np.array([[0.1 * (i + 1) * (j + 1) for i in range(2)] for j in range(5)]),
        [2, 12],
        ["central", "sum3"],
    )
    seen = [fit.fitresults_mem(tau), fit.fitresults_mem(tau, 5)]
    msd = _corr_fit(
        f,
        ["central"],
        fitresults=lambda i: np.array(
            [np.linspace(1e-3, 1e-1, 8), np.linspace(0.1, 0.5, 8)]
        ),
    )
    seen += [msd.fitresults_msd()]
    return tuple(seen)


def _fit_single_workflow(f):
    model = _model(f)
    single = f("central", model, [1, 0, 1, 1], np.array([2.0, 3e-3, 0.1]), [2, 12])
    seen = [single.fitfunction_label, single.paramidx, single.minbound, single.fitarray]
    unused = f(
        "sum3",
        _model(f, fitfunctionParamUsed=None),
        [1, 0, 1, 1],
        np.array([2.0, 3e-3, 0.1]),
        [2, 12],
    )
    snapshot = [
        single.startvalues.copy(),
        single.param_factors10.copy(),
        unused.startvalues,
        unused.fitarray,
    ]
    single.update(np.arange(10.0), np.ones(6), 250.0, 1.2)
    single.update()
    unused.update(w0=100.0)
    return tuple(seen + snapshot), single, unused


def _corr_function_workflow(f):
    function = f()
    seen = [function]
    function.set_params(
        {
            "mode": "m",
            "elements": ["a"],
            "listOfG": ["b"],
            "shortlabel": "s",
            "average": ["x+y"],
        }
    )
    seen += [function, f()]
    function.set_params({})
    return tuple(seen), function


def _fit_model_workflow(f):
    model = f()
    seen = [model.num_param]
    model.set_params(
        {
            "model": "Free diffusion",
            "shortlabel": "3D",
            "paramNames": ["N"],
            "paramFittable": [1],
            "paramDefvalues": [1.0],
            "allparamDefvalues": [1.0, 2.0],
            "paramFactors10": [1.0],
            "paramMinbound": [0.0, 0.0, 0.0],
            "paramMaxbound": [1.0, 1.0, 1.0],
            "fitfunctionName": "fcs_analytical",
            "fitfunctionParamUsed": [0],
            "globalParam": [True],
        }
    )
    seen += [model.num_param, model.returnfitparam()]
    model.model = "Free diffusion pair-correlation (global fit)"
    seen += [model.num_param]
    bare = f()
    bare.set_params({"fitfunctionName": "fcs_analytical"})
    seen += [_error(lambda: bare.num_param), bare]
    return tuple(seen), model


def _timetrace_end_cases(f):
    plain = np.ones((100, 25))
    zeros_5 = plain.copy()
    zeros_5[-5:] = 0
    zeros_60 = plain.copy()
    zeros_60[-60:] = 0
    return [f(plain), f(zeros_5), f(zeros_60), f(time_trace()[:300])]


CASES: list[Case] = [
    Case(
        f"{A2P}:arrival_times_to_pch",
        lambda f: f(_t0(), 1e-2, 0, 5.0, True),
        "normalized",
    ),
    Case(
        f"{A2P}:arrival_times_to_pch", lambda f: f(_t0(), 1e-2, 1, 5.0, False), "counts"
    ),
    Case(
        f"{A2P}:arrival_times_to_pch",
        lambda f: f(_t0(), 1e-2, 0, 5.0, True, 3),
        "max_k cuts the histogram",
    ),
    Case(
        f"{A2P}:arrival_times_to_pch",
        lambda f: f(_t0(), 1e-1, 0, 5.0, True, 200),
        "max_k above the histogram",
    ),
    Case(
        f"{A2P}:arrival_times_to_pch",
        lambda f: _error(f, _t0(), 1e-2, 5, 5.0, True),
        "empty chunk",
    ),
    Case(
        f"{A2P}:atimes_2_pch_all",
        lambda f: f(_atimes_pch(f), [3, "x12"], 2.0, 1e-2, max_k=10),
        "int and x elements",
    ),
    Case(
        f"{A2P}:atimes_2_pch_all",
        lambda f: f(
            _atimes_pch(f),
            ["central", "sum3"],
            5.0,
            1e-2,
            normalize=False,
            list_of_pch_out=["a", "b"],
        ),
        "counts and renamed outputs",
    ),
    Case(
        f"{A2P}:atimes_2_pch_all",
        lambda f: _error(f, _atimes_pch(f), [3], 20.0, 1e-2),
        "duration shorter than split",
    ),
    Case(f"{D2P}:calc_pch", lambda f: f(time_trace(), 1, 30), "binsize 1"),
    Case(
        f"{D2P}:calc_pch",
        lambda f: f(time_trace(), 10, 30, False),
        "binned, not normalized",
    ),
    Case(
        f"{D2P}:calc_pch",
        lambda f: f(time_trace()[:, 0], 4, 8),
        "vector, maxcount cuts the histogram",
    ),
    Case(
        f"{D2P}:calc_pch",
        lambda f: f(np.zeros((100, 3), dtype=np.uint32), 2, 5),
        "no photons",
    ),
    Case(f"{GPS}:generate_3d_gaussian", lambda f: f((9, 11, 13), 4.0, 6.0, 1.0, 2.0)),
    Case(
        f"{GPS}:generate_3d_gaussian",
        lambda f: f((8, 8, 8), 3.0, 5.0),
        "default pixel sizes",
    ),
    Case(f"{SPC}:simulate_pch_1c", lambda f: f(_psf(), k_max=16, c=0.3, q=3.0)),
    Case(
        f"{SPC}:simulate_pch_1c",
        lambda f: f(_psf(), dV=2.0, k_max=30, c=0.1, q=5.0, T=0.5, dV0=2.0, n_draws=3),
        "all arguments",
    ),
    Case(
        f"{SPC}:simulate_pch_nc",
        lambda f: f(_psf(), k_max=30, c=[0.3, 0.2], q=[3.0, 1.0], bg=0.2),
        "two components, background, k_max not a power of two",
    ),
    Case(
        f"{SPC}:simulate_pch_nc",
        lambda f: f(_psf(), k_max=32, c=[0.3], q=[3.0], T=0.5, dV0=2.0, n_draws=2),
        "k_max a power of two",
    ),
    Case(
        f"{SPC}:simulate_pch_nc",
        lambda f: f(_psf(), dV=np.ones((8, 8, 8)), k_max=8, c=[0.3], q=[3.0], bg=0.1),
        "voxel weights",
    ),
    Case(
        f"{SPC}:simulate_photon_counts_1c_mc",
        lambda f: _run_serial(f, _mc_psf() * 512, 0.02, 5.0, 40, 0.5),
        "background",
    ),
    Case(
        f"{SPC}:simulate_photon_counts_1c_mc",
        lambda f: _run_serial(f, _mc_psf() * 512, 0.02, 5.0, 40),
        "no background",
    ),
    Case(
        f"{SPC}:simulate_pch_1c_mc",
        lambda f: _run_serial(f, _mc_psf() * 512, 0.02, 5.0, 40, 0.5, 21),
    ),
    Case(
        f"{SPC}:simulate_pch_nc_mc",
        lambda f: _run_serial(
            f, _mc_psf() * 512, [0.02, 0.01], [5.0, 2.0], 40, 0.5, 21
        ),
    ),
    Case(
        f"{SPC}:simulate_pch_1c_mc_ntimes",
        lambda f: _run_serial(f, _mc_psf() * 512, 0.02, 5.0, 30, 0, 6, 21, 1e-9),
        "runs to n_hist_max",
    ),
    Case(
        f"{SPC}:simulate_pch_1c_mc_ntimes",
        lambda f: _run_serial(f, _mc_psf() * 512, 0.02, 5.0, 30, 0, 8, 21, 1.0),
        "stops on the error criterion",
    ),
    Case(
        f"{PFIT}:fit_pch", lambda f: _fit_single(f, "relative"), "fitfun_pch relative"
    ),
    Case(
        f"{PFIT}:fit_pch", lambda f: _fit_single(f, "absolute"), "fitfun_pch absolute"
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_single(f, "relative", "fitfun_pch_nc"),
        "fitfun_pch_nc relative",
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_single(f, "absolute", "fitfun_pch_nc", 2),
        "fitfun_pch_nc absolute, n_draws",
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_single(f, "mle", "fitfun_pch_nc", **_fit_mle_bounds()),
        "fitfun_pch_nc mle",
        compare=_same_minimize,
    ),
    Case(
        f"{PFIT}:fit_pch", lambda f: _fit_two(f, "relative"), "two components relative"
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_two(f, "mle"),
        "two components mle",
        compare=_same_minimize,
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: f(
            _pch(f), [1, 1, 0, 0], [0.5, 2.0, 1.0, 1.0], [300.0, 2.0], n_bins=200
        ),
        "Gaussian PSF from [w0, z0]",
    ),
    Case(f"{PFIT}:fit_pch", lambda f: _fit_global(f, "relative"), "global relative"),
    Case(f"{PFIT}:fit_pch", lambda f: _fit_global(f, "absolute"), "global absolute"),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_global(f, "mle"),
        "global mle",
        compare=_same_minimize,
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_global(f, "relative", 1, [False] * 5),
        "global, nothing shared",
    ),
    Case(
        f"{PFIT}:fit_pch",
        lambda f: _fit_global(f, "relative", 1, [True] * 5, np.array([1, 0, 0, 0, 0])),
        "global, everything shared",
    ),
    Case(
        f"{PFIT}:fitfun_pch",
        lambda f: f(
            np.array([0.0, 0.3]),
            np.array([1.0, 1.0]),
            np.array([1, 1, 0, 0]),
            _pch(f),
            *_compressed_psf(f),
            1,
            1,
            "absolute",
        ),
    ),
    Case(
        f"{PFIT}:fitfun_pch",
        lambda f: f(
            np.array([0.0, 0.3]),
            np.array([1.0, 1.0]),
            np.array([1, 1, 0, 0]),
            _pch(f),
            *_compressed_psf(f),
            0.5,
            1,
            "relative",
        ),
        "relative residuals",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc",
        lambda f: f(
            np.array([-0.5, 0.3]),
            np.array([0.0, 1.0, 1.0]),
            _NC_INFO,
            _pch(f),
            *_compressed_psf(f),
            1,
            1,
            "absolute",
        ),
        "absolute",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc",
        lambda f: f(
            np.array([-0.5, 0.3]),
            np.array([0.0, 1.0, 1.0]),
            _NC_INFO,
            _pch(f),
            *_compressed_psf(f),
            1,
            2,
            "relative",
        ),
        "relative",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc",
        lambda f: f(
            np.array([0.3, 3.0]),
            np.array([0.0, 1.0, 1.0]),
            _NC_INFO,
            _pch(f),
            *_compressed_psf(f),
            1,
            1,
            "mle",
        ),
        "mle negative log likelihood",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc",
        lambda f: f(
            np.array([0.3, 3.0]),
            np.array([0.0, 1.0, 1.0]),
            _NC_INFO,
            _pch(f),
            *_compressed_psf(f),
            1,
            1,
            "mle",
            True,
        ),
        "mle residuals",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc_global", lambda f: _global_fun(f, "absolute"), "absolute"
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc_global", lambda f: _global_fun(f, "relative"), "relative"
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc_global",
        lambda f: _global_fun(f, "mle"),
        "mle negative log likelihood",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc_global",
        lambda f: _global_fun(f, "mle", return_residuals=True),
        "mle residuals",
    ),
    Case(
        f"{PFIT}:fitfun_pch_nc_global",
        lambda f: _global_fun(f, "absolute", global_param=False),
        "global_param None",
    ),
    Case(
        f"{PFIT}:fitfun_pch_mc",
        lambda f: _error(
            f,
            np.array([0.02, 5.0]),
            np.array([40.0, 4.0, 21.0, 1e-3]),
            np.array([1, 1, 0, 0, 0, 0]),
            _pch(f),
            _mc_psf() * 512,
        ),
        "float err used as an array size by simulate_pch_1c_mc_ntimes",
    ),
    Case(f"{PFIT}:generate_psfs", lambda f: f(_psf(), 1, 500), "array, one histogram"),
    Case(
        f"{PFIT}:generate_psfs",
        lambda f: f(_psf_stack(), 2, 500),
        "array, two histograms",
    ),
    Case(
        f"{PFIT}:generate_psfs",
        lambda f: f([300.0, 2.0], 1, 300),
        "list, one histogram",
    ),
    Case(
        f"{PFIT}:generate_psfs",
        lambda f: f([300.0, 2.0, 400.0, 3.0], 2, 300),
        "list, two histograms",
    ),
    Case(
        f"{PFIT}:global_fit_result_to_parameters",
        lambda f: f(
            np.ones((5, 2)),
            np.array([1, 1, 0, 0, 1]),
            2,
            [True, False, True, True, False],
            np.array([7.0, 8.0, 9.0, 10.0, 11.0]),
        ),
    ),
    Case(
        f"{PFIT}:global_fit_result_to_residuals",
        lambda f: f(
            np.arange(1.0, 13.0), np.arange(1.0, 13.0).reshape(6, 2), 2, 2.0, "relative"
        ),
        "relative",
    ),
    Case(
        f"{PFIT}:global_fit_result_to_residuals",
        lambda f: f(
            np.arange(1.0, 13.0), np.arange(1.0, 13.0).reshape(6, 2), 2, 2.0, "absolute"
        ),
        "absolute",
    ),
    Case(
        f"{PFIT}:hist_param",
        lambda f: (f(np.zeros(7)), f(np.zeros((7, 3)))),
        "vector and matrix",
    ),
    Case(
        f"{PFIT}:make_2D_fit_parameter_array_global_fit",
        lambda f: f(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0, 7.0, 8.0]),
            np.array([1, 0, 1, 0, 0]),
            [True, True, False, False, True],
            5,
            2,
        ),
    ),
    Case(
        f"{PFIT}:make_2D_fit_parameter_array_global_fit",
        lambda f: f(
            np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            np.array([6.0, 7.0, 8.0]),
            np.array([1, 1, 1, 0, 0]),
            [False, False, True, True, False],
            5,
            2,
            np.array([1, 10, 1, 1, 1]),
        ),
        "param_factors10",
    ),
    Case(
        f"{PFIT}:make_fit_parameters_global_fit",
        lambda f: f(
            np.arange(10.0).reshape(5, 2),
            np.array([1, 0, 1, 0, 1]),
            2,
            [True, False, False, True, False],
            np.arange(5.0),
            np.arange(5.0) + 10,
        ),
    ),
    Case(
        f"{PFIT}:parse_fitfun_and_fitinfo",
        lambda f: (
            f("fitfun_pch", [1, 1, 0, 0], np.zeros(5)),
            f("fitfun_pch_nc_global", [1] * 7, np.zeros((5, 3))),
            f("other", [1] * 7, np.zeros(5)),
        ),
        "all three functions",
    ),
    Case(
        f"{PFIT}:parse_fitfun_and_fitinfo",
        lambda f: f(f.__globals__["fitfun_pch_nc_global"], [1] * 5, np.zeros((5, 2))),
        "function object",
    ),
    Case(
        f"{AFR}:calc_bic",
        lambda f: f(np.linspace(0, 1, 20), np.linspace(0, 1, 20) + 0.1, 0.2, 3),
        "scalar sigma",
    ),
    Case(
        f"{AFR}:calc_bic",
        lambda f: f(
            [1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8], [0.1, 0.2, 0.1, 0.2], 2
        ),
        "sigma per point",
    ),
    Case(f"{BIN}:bindata", lambda f: f(time_trace()[:1000], 7), "2D"),
    Case(f"{BIN}:bindata", lambda f: f(time_trace()[:1000, 0], 7), "vector"),
    Case(f"{BIN}:bindata", lambda f: f(time_trace()[:1000], 4, True), "interleaved"),
    Case(f"{BIN}:bindata_chunks", lambda f: f(time_trace(), 100), "2D"),
    Case(
        f"{BIN}:bindata_chunks",
        lambda f: _printed(f, time_trace()[:, 3], 50, True),
        "vector, chunk numbers printed",
    ),
    Case(
        f"{BIN}:bindata_chunks",
        lambda f: f(time_trace()[:1000], 10, interleaved_binning=True),
        "interleaved",
    ),
    Case(f"{CCO}:calcconc", lambda f: _printed(f, 1e-3, 0.5), "default molar mass"),
    Case(f"{CCO}:calcconc", lambda f: _printed(f, 2e-3, 0.1, 500), "molar mass"),
    Case(f"{CDC}:calcdist_from_coord", lambda f: f(np.zeros((5, 7))), "center"),
    Case(
        f"{CDC}:calcdist_from_coord",
        lambda f: f(np.zeros((5, 7)), [1, 2]),
        "coordinate",
    ),
    Case(f"{CDC}:list_of_pixel_pairs_at_distance", lambda f: f([0, 1]), "default"),
    Case(
        f"{CDC}:list_of_pixel_pairs_at_distance",
        lambda f: f([1, -1], 9, [4]),
        "pixels off",
    ),
    Case(
        f"{CDC}:calc_shift_from_coord",
        lambda f: (f(3, 17), f(0, 8, 3)),
        "default and given N",
    ),
    Case(f"{CAD}:cast_data", lambda f: f(np.array([1.7, 2.2, -3.5])), "default"),
    Case(
        f"{CAD}:cast_data",
        lambda f: f(np.array([1.7, 2.2, -3.5]), "float32"),
        "float32",
    ),
    Case(
        f"{CHK}:checkfname",
        lambda f: (
            f("a", "tiff"),
            f("a.tiff", "tiff"),
            f(None, "x"),
            f("tiff", "tiff"),
            f("file.h5", "h5"),
        ),
        "missing, present, None, too short",
    ),
    Case(
        f"{C2D}:cmap2d",
        lambda f: f(
            *_lifetime_and_intensity(),
            ltimeBound=["auto", "auto"],
            intBound=["auto", "auto"],
        ),
        "automatic bounds",
    ),
    Case(
        f"{C2D}:cmap2d",
        lambda f: f(
            *_lifetime_and_intensity(),
            ltimeBound=[1.5, 4.0],
            intBound=[10.0, 80.0],
            cmap="viridis",
        ),
        "given bounds",
    ),
    Case(
        f"{C2D}:cmap2d",
        lambda f: f(
            *_lifetime_and_intensity(),
            ltimeBound=["auto", "auto"],
            intBound=["auto", "auto"],
            equalize_lum=False,
        ),
        "no luminance equalization",
    ),
    Case(
        f"{C2D}:cmap2d",
        lambda f: f(
            *_lifetime_and_intensity(),
            ltimeBound=["auto", "auto"],
            intBound=["auto", "auto"],
            cmap="seaborn-colorblind",
        ),
        "seaborn-colorblind",
    ),
    Case(
        f"{C2D}:cmap2d",
        lambda f: f(
            *_lifetime_and_intensity(),
            ltimeBound=["auto", "auto"],
            intBound=["auto", "auto"],
            plotFig=True,
        ),
        "plotted",
    ),
    Case(
        f"{CMS}:cmaps",
        lambda f: [
            f(name, 16)
            for name in (
                "blue2black",
                "orange2black",
                "green2black",
                "red2black",
                "purple2black",
                "other",
            )
        ],
        "every color and an invalid one",
    ),
    Case(f"{CMS}:color_gradient", lambda f: f((0, 0, 0, 1), (1.0, 0.5, 0.2, 1), 8)),
    Case(
        f"{CMS}:change_color_from_map",
        lambda f: f("viridis", [1.0, 0.0, 0.0], 3),
        "listed color map",
    ),
    Case(
        f"{CMS}:change_color_from_map",
        lambda f: f("seaborn-colorblind", [1.0, 0.0, 0.0]),
        "seaborn-colorblind",
    ),
    Case(
        f"{CMS}:change_color_from_map",
        lambda f: _error(f, "jet", [1.0, 0.0, 0.0]),
        "color map without a list of colors",
    ),
    Case(
        f"{CFM}:color_from_map",
        lambda f: [f(v, 0, 1, "viridis") for v in (-0.5, 0.0, 0.3, 1.0, 2.0)],
        "listed color map, inside and outside the range",
    ),
    Case(
        f"{CFM}:color_from_map",
        lambda f: [f(v, 0, 10, "jet") for v in (-1, 3, 10, 20)],
        "continuous color map",
    ),
    Case(
        f"{CFM}:color_from_map",
        lambda f: [f(v, 0, 1, "seaborn-colorblind") for v in (-0.5, 0.2, 0.9)],
        "seaborn-colorblind",
    ),
    Case(
        f"{CFM}:color_from_map",
        lambda f: _error(f, 1.0, 0, 1, "seaborn-colorblind"),
        "seaborn-colorblind, index past the palette",
    ),
    Case(
        f"{CON}:constants",
        lambda f: [f(name) for name in ("boltzmann", "Planck", "AVOGADRO", "unknown")],
        "every constant and an unknown one",
    ),
    Case(f"{DEL}:draw_ellipsoid", lambda f: f(1.0, 2.0, 3.0, False, 20), "data only"),
    Case(f"{DEL}:draw_ellipsoid", lambda f: f(1.0, 2.0, 3.0, True, 20), "plotted"),
    Case(
        f"{EXP}:find_particles",
        lambda f: _printed(f, _beads_for_particles(), 5, [20, 255]),
        "beads found",
    ),
    Case(
        f"{EXP}:find_particles",
        lambda f: _printed(f, _beads_for_particles(), 5, [20, 255], 10, True),
        "all results",
    ),
    Case(
        f"{EXP}:find_particles",
        lambda f: _printed(f, _beads_for_particles(), 5, [255, 255]),
        "nothing found",
    ),
    Case(
        f"{EXP}:exppsf",
        lambda f: _printed(f, _beads_for_particles(), 10, False, 5),
        "2D image",
    ),
    Case(
        f"{EXP}:exppsf",
        lambda f: _printed(f, _beads_for_particles(3), 10, True, 5),
        "3D image, plotted",
    ),
    Case(
        f"{FNE}:find_nearest",
        lambda f: (f([1.0, 4.0, 9.0], 5.0), f(np.array([3, 1, 2]), 0)),
    ),
    Case(
        f"{FCU}:fit_curve",
        lambda f: f(
            *_yx(_power_law),
            "powerlaw",
            [1, 1, 0],
            np.array([3.1, -1, 0]),
            np.array([0, -10, -10]),
            np.array([100, 0, 100]),
        ),
        "powerlaw",
    ),
    Case(
        f"{FCU}:fit_curve",
        lambda f: f(
            *_yx(_exp_decay),
            "exp",
            [1, 1, 0, 1],
            [1.0, 0.1, 0.0, 0.0],
            [0, 0, -1, -1],
            [10, 5, 1, 1],
        ),
        "exp",
    ),
    Case(
        f"{FCU}:fit_curve",
        lambda f: f(
            *_yx(_exp_decay, 2),
            "exp",
            [1, 1, 1, 1, 0, 1],
            [1.0, 0.1, 1.0, 1.0, 0.0, 0.0],
            [0, 0, 0, 0, -1, -1],
            [10, 5, 10, 10, 1, 1],
        ),
        "exp, two components",
    ),
    Case(
        f"{FCU}:fit_curve",
        lambda f: f(*_yx(_exp_decay), "linear", [1, 1], [-0.1, 1.0], [-5, -5], [5, 5]),
        "linear",
    ),
    Case(
        f"{FCU}:fit_curve",
        lambda f: _error(
            f, *_yx(_exp_decay), "other", [1, 1], [-0.1, 1.0], [-5, -5], [5, 5]
        ),
        "unknown function",
    ),
    Case(
        f"{FCU}:fit_power_law_g",
        lambda f: f(_g_object(3), 2, True),
        "few curves, plotted individually",
    ),
    Case(
        f"{FCU}:fit_power_law_g",
        lambda f: f(_g_object(25, True), 0, False),
        "25 curves in one figure",
    ),
    Case(
        f"{FCU}:fit_power_law_g",
        lambda f: f(_g_object(25, True), 3, True),
        "25 curves, plotted individually",
    ),
    Case(
        f"{FCU}:fitfun_powerlaw",
        lambda f: f(
            np.array([3.0, -0.5]), np.array([0.1]), np.array([1, 1, 0]), *_power_law()
        ),
    ),
    Case(
        f"{FCU}:fitfun_exp",
        lambda f: f(
            np.array([2.0, 0.4]),
            np.array([0.1, 0.3]),
            np.array([1, 1, 0, 0]),
            *_exp_decay(),
        ),
    ),
    Case(
        f"{FCU}:fitfun_exp_n",
        lambda f: f(
            np.array([2.0, 0.4, 0.3]),
            np.array([1.0, 2.0, 0.1]),
            np.array([1, 1, 0, 0, 0, 1]),
            *_exp_decay(2),
        ),
        "two components",
    ),
    Case(
        f"{FCU}:fun_exp_n",
        lambda f: f(_exp_decay()[0], [2.0, 0.4, 1.0, 2.0, 0.1, 0.3]),
        "two components",
    ),
    Case(
        f"{FCU}:fun_exp_n",
        lambda f: f(_exp_decay()[0], [2.0, 0.4, 0.1, 0.3]),
        "one component",
    ),
    Case(
        f"{FCU}:fitfun_linear",
        lambda f: f(np.array([0.5]), np.array([1.0]), np.array([1, 0]), *_exp_decay()),
    ),
    Case(
        f"{FDA}:fit_data",
        lambda f: f(
            *_yx(_exp_decay), "exp", [1, 1, 1], [1.0, 0.1, 0.0], [0, 0, -1], [10, 5, 1]
        ),
        "exp",
    ),
    Case(
        f"{FDA}:fit_data",
        lambda f: f(
            *_yx(_exp_decay, 2),
            "exp",
            [1, 1, 1, 1, 1],
            [1.0, 0.1, 1.0, 1.0, 0.0],
            [0, 0, 0, 0, -1],
            [10, 5, 10, 10, 1],
        ),
        "exp, two components",
    ),
    Case(
        f"{FDA}:fit_data",
        lambda f: _error(
            f,
            *_yx(_exp_decay),
            "other",
            [1, 1, 1],
            [1.0, 0.1, 0.0],
            [0, 0, -1],
            [10, 5, 1],
        ),
        "unknown function",
    ),
    Case(
        f"{FDA}:fit_and_plot",
        lambda f: _printed(f, *_yx(_exp_decay), "exp", [1, 1, 1], [1.0, 0.1, 0.0]),
        "default bounds, printed and plotted",
    ),
    Case(
        f"{FDA}:fit_and_plot",
        lambda f: _printed(
            f,
            *_yx(_exp_decay),
            "exp",
            [1, 1, 1],
            [1.0, 0.1, 0.0],
            printResults=0,
            plot_fit=0,
        ),
        "default bounds, silent",
    ),
    Case(
        f"{FDA}:fit_and_plot",
        lambda f: _error(
            f, *_yx(_exp_decay), "exp", [1, 1, 1], [1.0, 0.1, 0.0], 0, 100
        ),
        "scalar bounds",
    ),
    Case(
        f"{FDA}:fit_and_plot",
        lambda f: _error(
            f,
            *_yx(_exp_decay),
            "exp",
            [1, 1, 1],
            [1.0, 0.1, 0.0],
            np.zeros(3),
            np.ones(3),
        ),
        "array bounds",
    ),
    Case(
        f"{FDA}:fitfun_exp_n",
        lambda f: f(
            np.array([2.0, 0.4, 0.3]), np.array([]), np.array([1, 1, 1]), *_exp_decay()
        ),
    ),
    Case(f"{FDO}:doughnut2d", lambda f: f(10, 12, 4.2, 5.1, 2.0, 1.8, 0.2)),
    Case(
        f"{FDO}:fitfun_doughnut2d",
        lambda f: f(
            np.array([4.0, 5.0, 1.5, 1.9, 0.1]),
            np.array([]),
            np.ones(5),
            _doughnut_image(f),
        ),
        "all fitted",
    ),
    Case(
        f"{FDO}:fitfun_doughnut2d",
        lambda f: f(
            np.array([4.0, 1.5]),
            np.array([5.0, 1.9, 0.1]),
            np.array([1, 0, 1, 0, 0]),
            _doughnut_image(f),
            2.0,
        ),
        "some fixed, weighted",
    ),
    Case(
        f"{FDO}:fit_dougnut2D",
        lambda f: f(_doughnut_image(f), [1, 1, 1, 1, 1], [4.0, 5.0, 1.5, 1.9, 0.1]),
        "all fitted",
    ),
    Case(
        f"{FDO}:fit_dougnut2D",
        lambda f: f(
            _doughnut_image(f), [0, 0, 1, 1, 1], [4.2, 5.1, 1.5, 1.9, 0.1], 0.5
        ),
        "some fixed, weighted",
    ),
    Case(f"{FG2}:normal2d", lambda f: f(10, 12, 4.2, 5.1, 2.0, 1.8, 0.2)),
    Case(
        f"{FG2}:fitfun_gauss2d",
        lambda f: f(
            np.array([4.0, 5.0, 1.5, 1.9, 0.1]),
            np.array([]),
            np.ones(5),
            _gauss_image(f),
        ),
        "all fitted",
    ),
    Case(
        f"{FG2}:fitfun_gauss2d",
        lambda f: f(
            np.array([4.0, 1.5]),
            np.array([5.0, 1.9, 0.1]),
            np.array([1, 0, 1, 0, 0]),
            _gauss_image(f),
            2.0,
        ),
        "some fixed, weighted",
    ),
    Case(
        f"{FG2}:fit_gauss_2d",
        lambda f: f(_gauss_image(f), [1, 1, 1, 1, 1], [4.0, 5.0, 1.5, 1.9, 0.1]),
        "all fitted",
    ),
    Case(
        f"{FG2}:fit_gauss_2d",
        lambda f: f(_gauss_image(f), [0, 0, 1, 1, 1], [4.2, 5.1, 1.5, 1.9, 0.1], 0.5),
        "some fixed, weighted",
    ),
    Case(
        f"{FP2}:polynomial2d",
        lambda f: f(
            np.zeros((12, 10)),
            2,
            3,
            np.array([4.0, 5.0, 1.0, 0.1, 0.2, 0.02, 0.3, 0.01]),
        ),
    ),
    Case(
        f"{FP2}:fitfun_polynomial2d",
        lambda f: f(
            np.array([4.0, 5.0, 1.0, 0.1, 0.2, 0.02, 0.3, 0.01]), 2, 3, _gauss_image(f)
        ),
    ),
    Case(
        f"{FP2}:fitfun_polynomial2d",
        lambda f: f(
            np.array([4.0, 5.0, 1.0, 0.1, 0.2, 0.02, 0.3, 0.01]),
            2,
            3,
            _gauss_image(f),
            2.0,
        ),
        "weighted",
    ),
    Case(f"{FP2}:fit_polynomial2d", lambda f: f(_gauss_image(f), 2, 2), "order 2 x 2"),
    Case(
        f"{FP2}:fit_polynomial2d", lambda f: f(_gauss_image(f), 2, 3, 0.5), "weighted"
    ),
    Case(
        f"{FPS}:checkrange",
        lambda f: (f(5, 15, 0, 100), f(50, 15, 0, 100), f(95, 15, 0, 100)),
    ),
    Case(f"{FPS}:gaussian", lambda f: f(14, 12, 6.5, 5.0, 2.0, 3.0, 1.5, 0.1, 0.05)),
    Case(
        f"{FPS}:gaussian3d",
        lambda f: f(9, 8, 5, 4.0, 3.5, 2.0, 2.0, 2.5, 1.5, 0.05, 0.02, 1.5, 0.1),
    ),
    Case(
        f"{FPS}:gaussfitfun2d",
        lambda f: f(
            np.array([6.0, 5.0, 2.0, 3.0, 1.5, 0.1, 0.05]),
            np.array([]),
            np.ones(7),
            _bead_image(20, ((10.0, 9.0),)),
        ),
        "all fitted",
    ),
    Case(
        f"{FPS}:gaussfitfun2d",
        lambda f: f(
            np.array([6.0, 5.0]),
            np.array([2.0, 3.0, 1.5, 0.1, 0.05]),
            np.array([1, 1, 0, 0, 0, 0, 0]),
            _bead_image(20, ((10.0, 9.0),)),
        ),
        "some fixed",
    ),
    Case(
        f"{FPS}:gaussfitfun3d",
        lambda f: f(
            np.array([6.0, 5.0, 2.0, 3.0, 3.0, 2.0, 0.1, 0.1, 100.0, 0.5]),
            6.0,
            _bead_image(14, ((7.0, 7.0),), 5),
        ),
    ),
    Case(
        f"{FPS}:gauss_fitres",
        lambda f: f(
            _bead_image(20, ((10.0, 9.0),)),
            np.array([9.0, 10.0, 3.0, 3.0, 100.0, 0.5, 0.01]),
        ),
        "all fitted",
    ),
    Case(
        f"{FPS}:gauss_fitres",
        lambda f: f(
            _bead_image(20, ((10.0, 9.0),)),
            np.array([9.0, 10.0, 3.0, 3.0, 100.0, 0.5, 0.01]),
            np.array([1, 1, 0, 0, 1, 1, 0]),
        ),
        "some fixed",
    ),
    Case(
        f"{FPS}:gauss_fitres3d",
        lambda f: f(
            _bead_image(14, ((7.0, 7.0),), 5),
            [6.0, 7.0, 2, 3.0, 3.0, 3.0, 0.1, 0.1, 100.0, 0.0],
        ),
    ),
    Case(
        f"{FPS}:fit_psf",
        lambda f: _printed(_patched_ginput, f, _bead_image(), fitRange=10, Nbeads=2),
        "2D image",
    ),
    Case(
        f"{FPS}:fit_psf",
        lambda f: _printed(
            _patched_ginput, f, _bead_image(planes=5), fitRange=10, Nbeads=2
        ),
        "3D image",
    ),
    Case(
        f"{MAV}:moving_average",
        lambda f: (f(np.arange(10.0)), f(np.arange(12), 5)),
        "default and given window",
    ),
    Case(f"{N2H}:DataFileParam", lambda f: f(), "defaults", compare=_same_without_date),
    Case(
        f"{P2F}:path2fname",
        lambda f: (
            f("C:\\Users\\x\\file.ext"),
            f("C:/Users/x/file.ext"),
            f("file.ext"),
        ),
        "own splitting",
    ),
    Case(
        f"{P2F}:path2fname",
        lambda f: (f("C:\\Users\\x\\file.ext", True), f("file.ext", True)),
        "os.path.split",
    ),
    Case(
        f"{PHA}:phasor",
        lambda f: [
            f(h) for h in (decay_histogram()[0].astype(float), decay_histogram()[3])
        ],
        "normalized",
    ),
    Case(
        f"{PHA}:phasor",
        lambda f: f(decay_histogram()[0].astype(float), False),
        "not normalized",
    ),
    Case(
        f"{PLC}:plot_colors",
        lambda f: [f(v) for v in (0, 7, "central", "sum3_average", "av", "unknown")],
        "index, known and unknown names",
    ),
    Case(
        f"{PRT}:print_table",
        lambda f: _printed(f, np.arange(12.0).reshape(3, 4) / 7),
        "all columns",
    ),
    Case(
        f"{PRT}:print_table",
        lambda f: _printed(f, np.arange(36.0).reshape(3, 12) / 7, 6, 5),
        "truncated columns",
    ),
    Case(f"{SVF}:checkfname", lambda f: (f("figure"), f("figure.pickle"), f("short"))),
    Case(
        f"{STE}:stokes_einstein",
        lambda f: (f(1e-12), f(5e-12, 300, 8.9e-4)),
        "default and given conditions",
    ),
    Case(
        f"{T2S}:time2string",
        lambda f: [f(s) for s in (0, 0.5, 1.5, 59, 60, 61, 119, 125, 3600, 3601, 7300)],
        "every branch",
    ),
    Case(f"{ANS}:FFSlib", _lib_workflow, "images added, read and removed"),
    Case(
        f"{ANS}:FFSlib",
        lambda f: _error(f().add_image, np.zeros((3, 3)), ""),
        "image without a name",
    ),
    Case(f"{ANS}:FFSimage", _image_workflow, "files added, read and removed"),
    Case(f"{ANS}:FFSfile", _file_properties, "properties and update"),
    Case(f"{ANS}:FFSfile", _file_analyses, "analyses added and removed"),
    Case(
        f"{ANS}:FFSfile",
        lambda f: _error(
            _ffs_file(f, False, False).add_analysis,
            _corr_function(f),
            5,
            10.0,
            "multipletau",
            np.array([1, 0]),
        ),
        "chunks_off array compared with a string, numpy 2 makes it ambiguous",
    ),
    Case(f"{ANS}:FFSfile", _file_copy, "analysis copied without its fits"),
    Case(f"{ANS}:FFSfile", _file_fit_as_data, "fit used as data"),
    Case(f"{ANS}:FFSfile", _file_fit_as_data_free, "model-free fit skipped"),
    Case(
        f"{ANS}:FFSmetadata",
        lambda f: (
            f(),
            f(8, 6, 3, 10.0, 10.0, 2.0, 100, 5.0, 4.0, 0.5, 1e-3, 20.0, 0.05, [3, 4]),
        ),
        "default and all fields",
    ),
    Case(f"{ANS}:FFScorr", _corr_workflow, "settings, chunks and fits"),
    Case(f"{ANS}:FFScorr", _corr_getters, "correlations read"),
    Case(f"{ANS}:FFScorr", _fit_management, "fits added and updated"),
    Case(f"{ANS}:CorrSettings", lambda f: _settings_workflow(f), "updated"),
    Case(f"{ANS}:CorrFit", lambda f: _corr_fit_workflow(f), "fields and results"),
    Case(
        f"{ANS}:CorrFit",
        lambda f: _error(_corr_fit(f).return_field, "fitfunction_label"),
        "field of the first fit",
    ),
    Case(f"{ANS}:CorrFit", lambda f: _corr_fit_maps(f), "heat maps"),
    Case(
        f"{ANS}:CorrFit",
        lambda f: _corr_fit_mfda(f),
        "model-free displacement analysis",
    ),
    Case(
        f"{ANS}:CorrFit",
        lambda f: _corr_fit_mem(f),
        "maximum entropy method and mean squared displacement",
    ),
    Case(f"{ANS}:FitSingleObj", lambda f: _fit_single_workflow(f), "built and updated"),
    Case(
        f"{CFC}:CorrelationFunction", lambda f: _corr_function_workflow(f), "set_params"
    ),
    Case(
        f"{FMC}:FitModel", lambda f: _fit_model_workflow(f), "set_params and num_param"
    ),
    Case(
        f"{TTE}:timetrace_end",
        lambda f: _timetrace_end_cases(f),
        "trailing zeros, none, and more than fifty",
    ),
    Case(
        f"{RES}:check_none",
        lambda f: (
            [f(v) for v in (None, "None", "abc")]
            + [f(v, int) for v in ("5", 5, "None", None)]
            + [f(2.5, float)]
        ),
        "None and 'None' with and without a type",
    ),
    Case(f"{RES}:to_text", lambda f: [f(v) for v in (None, 3, "x", [1, 2])]),
    Case(f"{S2E}:next_letter", lambda f: [f("A"), f("Y")]),
    Case(f"{S2E}:green_fill", lambda f: f()),
]


SKIPPED: dict[str, str] = {
    f"{A2P}:atimes_file_2_pch": SKIP_FILE_IO,
    f"{D2P}:pch_load_and_hist_split": SKIP_FILE_IO,
    f"{A2T}:array2rgbtiff": SKIP_FILE_IO,
    f"{A2T}:array2tiff": SKIP_FILE_IO,
    f"{A2T}:tiff2array": SKIP_FILE_IO,
    f"{A2V}:array2video": SKIP_FILE_IO,
    f"{CLF}:closefile": SKIP_FILE_IO,
    f"{C2A}:array2csv": SKIP_FILE_IO,
    f"{C2A}:csv2array": SKIP_FILE_IO,
    f"{GTO}:goto": SKIP_FILE_IO,
    f"{H5D}:h5data": SKIP_FILE_IO,
    f"{H5D}:printmetadata": SKIP_FILE_IO,
    f"{LSF}:list_files": SKIP_FILE_IO,
    f"{N2H}:hdf5_read": SKIP_FILE_IO,
    f"{N2H}:np2hdf5": SKIP_FILE_IO,
    f"{SVF}:openfig": SKIP_FILE_IO,
    f"{SVF}:savefig": SKIP_FILE_IO,
    f"{SVV}:openvar": SKIP_FILE_IO,
    f"{SVV}:savevar": SKIP_FILE_IO,
    f"{LFM}:load_ffs_metadata": SKIP_FILE_IO,
    f"{RFS}:read_ffs_file": SKIP_FILE_IO,
    f"{RFS}:read_fitresults_from_ffs": SKIP_FILE_IO,
    f"{RFS}:read_g_from_ffs": SKIP_FILE_IO,
    f"{RES}:create_dir": SKIP_FILE_IO,
    f"{RES}:make_zip": SKIP_FILE_IO,
    f"{RES}:read_text": SKIP_FILE_IO,
    f"{RES}:restorelib": SKIP_FILE_IO,
    f"{RES}:restorelib_ffs": SKIP_FILE_IO,
    f"{RES}:restorelib_ffz": SKIP_FILE_IO,
    f"{RES}:save_image": SKIP_FILE_IO,
    f"{RES}:save_text": SKIP_FILE_IO,
    f"{RES}:savelib": SKIP_FILE_IO,
    f"{RES}:savelib_ffs": SKIP_FILE_IO,
    f"{RES}:savelib_ffz": SKIP_FILE_IO,
    f"{RES}:unzip": SKIP_FILE_IO,
    f"{S2E}:corr2excel": SKIP_FILE_IO,
    f"{S2E}:ffs2excel": SKIP_FILE_IO,
    f"{S2E}:lib2excel": SKIP_FILE_IO,
    f"{PHA}:plotphasor": SKIP_PLOT_ONLY,
    f"{RES}:save_ffs": SKIP_GUI,
}
