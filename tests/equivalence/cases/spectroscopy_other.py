"""Equivalence cases for `brighteyes_ffs.pch`, `brighteyes_ffs.tools` and `brighteyes_ffs.fcs_gui`."""

import contextlib
import io
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from _inputs import CHANNELS, bead_image, pch_psf, photon_arrival_times, time_trace
from _registry import SKIP_FILE_IO, SKIP_GUI, SKIP_PLOT_ONLY, Case
from joblib import parallel_config
from matplotlib import pyplot as plt

PCH = "brighteyes_ffs.pch"
TOOLS = "brighteyes_ffs.tools"
GUI = "brighteyes_ffs.fcs_gui"


def _sibling(obj, name):
    return import_module(obj.__module__.rpartition(".")[0] + "." + name)


def _printed(call, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = call(*args, **kwargs)
    return result, buffer.getvalue()


def _clicked_fit_psf(function, image):
    # bead_image is indexed (row, column), and ginput returns (column, row)
    clicks = [(16, 16), (44, 16), (30, 32), (14, 46), (48, 48)]
    with patch.object(plt, "ginput", return_value=clicks):
        return _drawn(lambda: _printed(function, image))


def _drawn(call):
    plt.close("all")
    result = call()
    plt.close("all")
    return result


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _seconds() -> np.ndarray:
    return photon_arrival_times().astype(np.float64) * 1e-7


def _root(obj):
    return ".".join(obj.__module__.split(".")[:-2])


def _pch_data(obj):
    data = import_module(_root(obj) + ".fcs.atimes_data").ATimesData()
    rng = _rng()
    for channel in range(CHANNELS):
        macro = np.sort(rng.integers(0, 4 * 10**11, size=2000)).astype(np.float64)
        micro = rng.integers(0, 25_000, size=2000).astype(np.float64)
        setattr(data, f"det{channel}", np.column_stack((macro, micro)))
    return data


def _pch_to_all(function, **kwargs):
    return _printed(function, _pch_data(function), split=0.1, bin_time=1e-3, **kwargs)


def _compressed(obj, n_hist=1, n_bins=200):
    psf = pch_psf() if n_hist == 1 else _two_psfs()
    return _sibling(obj, "pch_fit").generate_psfs(psf, n_hist, n_bins)


def _noisy_hist(hist):
    hist = hist * _rng().normal(1, 0.02, size=hist.shape)
    return hist / hist.sum(axis=0)


def _pch_hist(obj, concentration=(0.02,), brightness=(4.0,)):
    psf, weights = _compressed(obj)
    pch = _sibling(obj, "simulate_pch").simulate_pch_nc(
        psf,
        dV=weights,
        k_max=30,
        c=list(concentration),
        q=list(brightness),
        T=1.0,
        dV0=1.0,
        bg=0.05,
    )
    return _noisy_hist(pch)


def _global_pch_hist(obj):
    psf, weights = _compressed(obj, 2)
    simulate = _sibling(obj, "simulate_pch").simulate_pch_nc
    columns = [
        simulate(
            psf[i], dV=weights[i], k_max=30, c=[0.02], q=[q], T=1.0, dV0=1.0, bg=0.05
        )
        for i, q in enumerate((4.0, 2.5))
    ]
    return _noisy_hist(np.column_stack(columns))


_PCH_MODELS = {
    "single": SimpleNamespace(
        fit=[1, 1, 0, 0], start=[0.01, 2.0, 1.0, 1.0], n_comp=1, hist={}
    ),
    "one": SimpleNamespace(
        fit=[1, 1, 0, 0, 0], start=[0.01, 2.0, 0.0, 1.0, 1.0], n_comp=1, hist={}
    ),
    "background": SimpleNamespace(
        fit=[1, 1, 1, 0, 0], start=[0.01, 2.0, 0.01, 1.0, 1.0], n_comp=1, hist={}
    ),
    "likelihood": SimpleNamespace(
        fit=[1, 1, 0, 0, 0], start=[0.015, 3.0, 0.0, 1.0, 1.0], n_comp=1, hist={}
    ),
    "two": SimpleNamespace(
        fit=[1, 1, 1, 1, 0, 0, 0],
        start=[0.01, 0.01, 2.0, 5.0, 0.0, 1.0, 1.0],
        n_comp=2,
        hist={"concentration": (0.01, 0.02), "brightness": (1.5, 6.0)},
    ),
}


def _dense(result):
    # the L-BFGS-B inverse Hessian is a linear operator, which assert_same cannot compare
    if "hess_inv" in result:
        result.hess_inv = result.hess_inv.todense()
    return result


def _fit_pch(function, model, minimization, fitfun="fitfun_pch_nc", **kwargs):
    spec = _PCH_MODELS[model]
    n_param = len(spec.fit)
    return _dense(
        function(
            _pch_hist(function, **spec.hist),
            np.array(spec.fit),
            np.array(spec.start),
            pch_psf(),
            lBounds=[1e-10] * 2 * spec.n_comp + [0] * (n_param - 2 * spec.n_comp),
            uBounds=[1e10] * n_param,
            n_bins=200,
            fitfun=fitfun,
            minimization=minimization,
            **kwargs,
        )
    )


def _fit_pch_global(function, minimization):
    brightness = 3.0 if minimization == "mle" else 2.0
    start = np.array(
        [[0.015, 0.015], [brightness, brightness], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0]]
    )
    return _dense(
        function(
            _global_pch_hist(function),
            np.array([1, 1, 0, 0, 0]),
            start,
            _two_psfs(),
            lBounds=[1e-10, 1e-10, 0, 0, 0],
            uBounds=[1e10] * 5,
            n_bins=200,
            fitfun="fitfun_pch_nc_global",
            global_param=[True, False, False, False, False],
            minimization=minimization,
        )
    )


def _data_file_param(cls):
    # the constructor stores the current time
    parameters = cls()
    parameters.RecordingDate = ["fixed"]
    return parameters


def _serial(call, *args, **kwargs):
    # these callers fix n_jobs=-1 inside, which would draw from unseeded worker processes
    with parallel_config(backend="sequential"):
        return call(*args, **kwargs)


def _two_psfs() -> np.ndarray:
    return np.stack([pch_psf(), pch_psf() ** 0.5])


def _global_start() -> np.ndarray:
    return np.array([[0.01, 0.02], [2.0, 3.0], [0.0, 0.1], [1.0, 1.5], [1.0, 2.0]])


COLOR_MAPS = (
    "blue2black",
    "orange2black",
    "green2black",
    "red2black",
    "purple2black",
    "unknown",
)
GLOBAL_INFO = np.array([1, 1, 1, 0, 0])
GLOBAL_PARAM = [True, False, True, False, True]


def _residual_inputs(function):
    psf, weights = _compressed(function)
    return psf, weights, _pch_hist(function)


def _fitfun_pch(function, minimization, weights=1):
    psf, psf_weights, hist = _residual_inputs(function)
    return function(
        np.array([-1.7, 0.6]),
        np.array([1.0, 1.0]),
        np.array([1, 1, 0, 0]),
        hist,
        psf,
        psf_weights,
        weights,
        1,
        minimization,
    )


def _fitfun_nc(function, minimization, **kwargs):
    psf, psf_weights, hist = _residual_inputs(function)
    start = [0.02, 4.0] if minimization == "mle" else [-1.7, 0.6]
    return function(
        np.array(start),
        np.array([0.05, 1.0, 1.0]),
        np.array([1, 1, 0, 0, 0]),
        hist,
        psf,
        psf_weights,
        minimization=minimization,
        **kwargs,
    )


def _fitfun_nc_two(function):
    psf, psf_weights, hist = _residual_inputs(function)
    return function(
        np.array([-2.0, -1.7, 0.2, 0.8]),
        np.array([0.05, 1.0, 1.0]),
        np.array([1, 1, 1, 1, 0, 0, 0]),
        hist,
        psf,
        psf_weights,
    )


def _fitfun_global(function, minimization, weights=1, **kwargs):
    psf, psf_weights = _compressed(function, 2)
    start = [0.02, 4.0, 2.5] if minimization == "mle" else [-1.7, 0.6, 0.4]
    return function(
        np.array(start),
        np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([1, 1, 0, 0, 0]),
        _global_pch_hist(function),
        psf,
        [True, False, False, False, False],
        psf_weights,
        weights,
        minimization,
        **kwargs,
    )


def _curve(kind):
    x = np.arange(1.0, 41.0)
    noise = _rng().normal(0, 1e-3, size=x.shape)
    match kind:
        case "powerlaw":
            return x, 3.0 * x**-1.2 + 0.05 + noise
        case "exp":
            return x, 5.0 * np.exp(-0.3 * (x - 0.5)) + 0.2 + noise
        case "linear":
            return x, 2.0 * x + 1.0 + noise
    raise ValueError(kind)


def _fit_curve(function, kind, fit, start, lower, upper):
    x, y = _curve(kind)
    return function(
        y, x, kind, np.array(fit), np.array(start), np.array(lower), np.array(upper)
    )


def _fit_curve_g(function, n_curves, plot_individually):
    x = np.arange(1.0, 31.0)
    rng = _rng()
    names = (
        ["central", "sum3", "sum5"]
        if n_curves == 3
        else [f"det{i}_average" for i in range(n_curves)]
    )
    curves = {
        name: np.column_stack(
            (x, (2.0 + i % 3) * x**-1.1 + rng.normal(0, 1e-3, x.shape))
        )
        for i, name in enumerate(names)
    }
    return _drawn(
        lambda: function(
            SimpleNamespace(dwellTime=1e-6, **curves), 2, plot_individually
        )
    )


def _residuals(function, start, fixed, fit, *data):
    return function(np.array(start), np.array(fixed), np.array(fit), *data)


def _images():
    rng = _rng()
    return rng.uniform(1.0, 5.0, size=(16, 20)), rng.uniform(10.0, 100.0, size=(16, 20))


def _color_image(function, **kwargs):
    lifetime, intensity = _images()
    return function(lifetime, intensity, **kwargs)


def _blob_data(function, module, model, truth):
    generate = getattr(_sibling(function, module), model)
    return generate(20, 16, *truth) + _rng().normal(0, 0.01, size=(16, 20))


BLOB_TRUTH = (10.0, 7.5, 2.0, 3.0, 0.1)
BLOB_START = [9.0, 8.0, 1.5, 2.5, 0.0]
BLOB_WEIGHTS = np.linspace(0.5, 2.0, 320).reshape(16, 20)


def _blob_fit(function, module, model, fit, weights=1):
    return function(
        _blob_data(function, module, model, BLOB_TRUTH),
        np.array(fit),
        BLOB_START,
        weights,
    )


def _blob_residuals(function, module, model):
    y = _blob_data(function, module, model, BLOB_TRUTH)
    return function(
        np.array(BLOB_START[:4]),
        np.array([0.0]),
        np.array([1, 1, 1, 1, 0]),
        y,
        BLOB_WEIGHTS,
    )


POLY_COEFFS = np.array([5.0, 6.0, 1.0, 0.1, -0.05, 0.01])


def _poly_data(function):
    model = _sibling(function, "fit_polynomial_2d").polynomial2d
    return model(np.zeros((12, 10)), 2, 2, POLY_COEFFS) + _rng().normal(
        0, 0.01, size=(12, 10)
    )


def _psf_image(function, three_d=False):
    module = _sibling(function, "fit_psf")
    if three_d:
        image = module.gaussian3d(
            12, 14, 8, 5.5, 6.5, 3.5, 5.0, 6.0, 30.0, 0.05, 0.02, 50.0, 2.0
        )
        return image + _rng().normal(0, 0.1, size=image.shape)
    image = module.gaussian(21, 19, 10.5, 9.0, 6.0, 5.0, 100.0, 3.0, 0.05)
    return image + _rng().normal(0, 0.5, size=image.shape)


PSF_START = [10.0, 10.0, 5.0, 5.0, 80.0, 1.0, 0.0]


def _gui(obj, name="analysis_settings"):
    return import_module(f"{_root(obj)}.fcs_gui.{name}")


def _metadata(obj, duration=20.0):
    return _gui(obj).FFSmetadata(
        num_pixels=20,
        num_lines=16,
        num_frames=1,
        range_x=75.0,
        range_y=60.0,
        range_z=0.0,
        num_datapoints=1000,
        hold_off_x5=3,
        hold_off=5,
        time_resolution=1.0,
        dwelltime=1e-5,
        duration=duration,
        pxsize=0.0375,
        coords=[3, 4],
    )


def _corr_function(
    obj, mode="Spot-variation fcs", elements=("central", "sum3", "sum5")
):
    function = _gui(obj, "correlation_functions_class").CorrelationFunction()
    function.set_params(
        {
            "mode": mode,
            "elements": list(elements),
            "listOfG": list(elements),
            "shortlabel": "Spot",
            "average": None,
        }
    )
    return function


def _fit_model(obj):
    model = _gui(obj, "fitmodel_class").FitModel()
    model.set_params(
        {
            "model": "Free diffusion",
            "shortlabel": "FD",
            "paramNames": ["N", "tauD", "offset"],
            "paramFittable": [1, 1, 1],
            "paramDefvalues": [1.0, 1e-3, 0.0],
            "allparamDefvalues": [1.0, 1e-3, 5.0, 0.0, 0.0],
            "paramFactors10": [1, 1e3, 1],
            "paramMinbound": [0, 1e-6, 1, -1, 0],
            "paramMaxbound": [1e3, 1, 50, 1, 1e3],
            "fitfunctionName": import_module(
                _root(obj) + ".fcs.fcs_analytical"
            ).fcs_analytical,
            "fitfunctionParamUsed": [0, 1, 3],
            "globalParam": [False, False, False],
        }
    )
    return model


def _start_values(n_curves=3):
    return np.array(
        [
            [1.0 + i for i in range(n_curves)],
            [1e-3 * (1 + i) for i in range(n_curves)],
            [0.1 * (1 + i) for i in range(n_curves)],
        ]
    )


def _correlations(obj, n_chunks=2, names=None):
    names = names or [
        *("central", "sum3", "sum5"),
        *(f"det12x{n}" for n in range(25)),
        *(f"V{v}_H{h}" for v in range(-2, 3) for h in range(-2, 3)),
    ]
    rng = _rng()
    tau = np.logspace(-6, 0, 20)
    G = import_module(_root(obj) + ".fcs.fcs2corr").Correlations()
    for name in names:
        chunks = np.zeros((n_chunks, 20, 2))
        chunks[..., 0] = tau
        chunks[..., 1] = rng.uniform(0.1, 1, (n_chunks, 20))
        G.add_corr_chunks(name, chunks)
        G.list_of_g_out.append(name)
    return G


def _analysis(obj, n_fits=0, n_chunks=2):
    analysis = _gui(obj).FFScorr()
    analysis.analysis(_corr_function(obj), 10, 10, np.ones(n_chunks), "multipletau")
    analysis.update(corrs=_correlations(obj, n_chunks))
    analysis.settings.update(chunks_off=np.ones(n_chunks), analysis=analysis)
    for _ in range(n_fits):
        analysis.add_fit_analysis(
            _fit_model(obj), [1, 1, 0, 0], _start_values(), [1, 15]
        )
    return analysis


def _legacy_analysis(obj):
    rng = _rng()
    tau = np.logspace(-6, 0, 20)
    curve = lambda: np.column_stack((tau, rng.uniform(0.1, 1, 20)))
    corrs = SimpleNamespace(
        **{
            f"{name}{suffix}": curve()
            for name in ("central", "sum3", "sum5")
            for suffix in ("", "_chunk0", "_chunk1")
        }
    )
    analysis = _analysis(obj, n_fits=1)
    analysis.update(corrs=corrs)
    for curve_fit in analysis.fits[0].fit_all_curves:
        curve_fit.update(fitresult=rng.normal(0, 0.01, size=14))
    return analysis


def _corr_fit(obj, names=("central", "sum3", "sum5"), fit_range=(1, 15), result=None):
    fit = _gui(obj).CorrFit()
    fit.add_fit_analyses(
        "Spot-variation fcs",
        _fit_model(obj),
        [1, 1, 0, 0],
        _start_values(len(names)),
        list(fit_range),
        list(names),
    )
    for i, curve in enumerate(fit.fit_all_curves):
        curve.update(
            fitresult=_rng().normal(0, 0.01, size=14) if result is None else result(i),
            w0=250.0 + i,
            D=1.5 + i,
        )
    return fit


def _ffs_file(obj, label="spot", duration=20.0):
    return _gui(obj).FFSfile(
        fname="folder/file_y_3_x_4.bin",
        label=label,
        metadata=_metadata(obj, duration),
        timetrace=_rng().poisson(5, (100, 25)).astype(float),
    )


def _settings(cls):
    analysis = _analysis(cls)
    settings = cls()
    settings.update(
        elements=["central"],
        list_of_g=["central"],
        algorithm="tt2corr",
        resolution=12,
        chunksize=5.0,
    )
    settings.update(chunks_off=np.zeros(3), analysis=analysis)
    settings.update(chunks_off=np.array([1, 0, 1]), analysis=analysis)
    return analysis.corrs.good_chunks, settings, analysis


def _metadata_case(cls):
    return cls(), cls(num_pixels=3, range_x=1.5, coords=[1, 2])


def _correlation_function(cls):
    function = cls()
    function.set_params(
        {
            "mode": "Spot-variation fcs",
            "elements": ["central"],
            "listOfG": ["central", "sum3"],
            "average": ["a"],
        }
    )
    return cls(), function


def _fit_model_case(cls):
    model = _fit_model(cls)
    renamed = _fit_model(cls)
    renamed.model = "Free diffusion pair-correlation (global fit)"
    return (
        model.returnfitparam(),
        model.num_param,
        cls().num_param,
        renamed.num_param,
        model,
        renamed,
    )


def _fit_single(cls):
    single = cls(
        "central", _fit_model(cls), [1, 1, 0, 0], np.array([1.0, 2e-3, 0.1]), [1, 15]
    )
    single.update(fitresult=np.arange(14.0), w0=250.0, D=1.2)
    single.update(startvalues=np.arange(6.0))
    unused = _fit_model(cls)
    unused.fitfunction_param_used = None
    return single, cls("sum3", unused, [1, 1, 0, 0], np.array([1.0]), [1, 10])


def _corr_fit_results(cls):
    fit = _corr_fit(cls)
    results = [
        fit.num_fitcurves,
        fit.return_field("fitAllCurves"),
        fit.return_field("w0"),
        fit.return_field("D"),
        fit.return_all("data"),
        fit.fitrange(),
        fit.fitresults(),
        fit.fitresults("array"),
        cls().return_all("data"),
        cls().fitrange(),
        cls().return_field("anything"),
    ]
    return results, fit


def _corr_fit_mem(cls):
    fit = _corr_fit(cls, ("central", "sum3"), (2, 10))
    for i, curve in enumerate(fit.fit_all_curves):
        curve.update(
            startvalues=np.concatenate(
                (_rng().uniform(0.1, 1, 8) * (i + 1), [5.0, 1e-3, 5.0, 0.0, 0.0])
            )
        )
    return fit.fitresults_mem(np.logspace(-6, 0, 20), 5), fit


def _corr_fit_msd(cls):
    fit = _corr_fit(
        cls,
        ("central",),
        result=lambda i: np.vstack((np.logspace(-6, -1, 10), np.linspace(1, 2, 10))),
    )
    return fit.fitresults_msd(), fit


def _corr_fit_map(cls, names, method, result):
    fit = _corr_fit(cls, names, result=result)
    return getattr(fit, method)(), fit


def _directions(i):
    return np.linspace(1.0, 2.0, 12) * (1 + 0.3 * i) + 0.1 * i**2


def _corr_fit_mfda(cls):
    names = ("V0_H0", "V0_H1", "V1_H-1")
    return _corr_fit_map(
        cls, names, "fitresults_mfda", lambda i: np.array([1e-3 * (1 + i), 0.5 + i])
    )


def _ffs_corr(cls):
    analysis = _analysis(cls, n_fits=2)
    results = [
        analysis.n_curves_mode,
        analysis.num_fits,
        analysis.num_chunks(20.0),
        analysis.num_chunks(None),
        analysis.get_corr(),
        analysis.get_corr("sum3"),
        analysis.get_corr("crossCenterAv"),
        analysis.get_corr("crossCenterAv_chunk0"),
        analysis.get_corr3D(),
        analysis.analysis_summary(),
        analysis.calc_corr(),
        analysis.corr_param(),
        analysis.return_fit_obj(),
        analysis.return_fit_obj(0),
        analysis.return_fit_obj(5),
        cls().calc_corr(),
        cls().get_corr(),
        cls().get_corr3D(),
    ]
    analysis.update_fit_analysis(
        _fit_model(cls), [1, 1, 1, 1], _start_values() * 2, [2, 12]
    )
    analysis.update_fit_analysis(
        _fit_model(cls), [0, 1, 0, 0], _start_values() * 3, [3, 14], fit_num=0
    )
    analysis.add_fit_analysis(_fit_model(cls), [0, 0, 0, 0], _start_values(), [1, 15])
    analysis.remove_fit(0)
    analysis.remove_fit(7)
    analysis.update(mode="renamed", active_fit="None")
    analysis.update(active_fit=0)
    return results, analysis


def _ffs_corr_cross(cls):
    analysis = cls()
    analysis.analysis(
        _corr_function(cls, "Cross", ("crossAll",)),
        8,
        5,
        np.ones(3),
        "multipletau",
        active_fit=2,
    )
    other = cls()
    other.analysis(
        _corr_function(cls, "Cross", ("crossAll",)), 8, 5, np.ones(3), "tt2corr"
    )
    return analysis, other


def _ffs_file_session(cls):
    file = _ffs_file(cls)
    file.add_analysis(_corr_function(cls), resolution=8, chunksize=5)
    file.add_analysis(_corr_function(cls, "Second"), active_analysis=0)
    file.get_analysis(0).update(corrs=_correlations(cls, 4))
    file.get_analysis(0).add_fit_analysis(
        _fit_model(cls), [1, 1, 0, 0], _start_values(), [1, 15]
    )
    file.copy_correlation(0)
    file.copy_correlation(9)
    results = [
        file.num_analyses,
        file.coords,
        file.duration,
        file.number_of_elements,
        file.get_analysis(),
        file.get_analysis(1),
        file.get_analysis(9),
    ]
    file.update(
        label="renamed",
        coords=[7, 8],
        timetrace=_rng().poisson(5, 100).astype(float),
        airy=np.ones(6),
        active_analysis=1,
    )
    results.append(file.number_of_elements)
    file.remove_analysis(1)
    file.remove_analysis(0)
    file.remove_analysis(9)
    file.remove_analysis(0)
    file.update(active_analysis="None")
    results.append(cls().number_of_elements)
    return results, file


def _ffs_file_fit_as_data(cls):
    file = _ffs_file(cls)
    file.analysis_list.append(_legacy_analysis(cls))
    file.use_fit_as_data(0, 0)
    file.use_fit_as_data(3, 0)
    file.use_fit_as_data(0, 5)
    return file


def _ffs_image(cls):
    image = cls()
    image.change_image(np.arange(60.0).reshape(6, 10), "path/image.tif", _metadata(cls))
    image.update(image_name="renamed.tif", image=np.ones((6, 10)))
    first, second = _ffs_file(cls, "a"), _ffs_file(cls, "b", 40.0)
    image.add_ffs_file(first)
    image.add_ffs_file(second)
    results = [
        image.num_files,
        image.get_ffs_file(0),
        image.get_ffs_file(),
        image.get_ffs_file(9),
        image.print_image_metadata(),
    ]
    image.active_ffs = 1
    results.append(image.get_ffs_file())
    image.update(ffs_list=[second, first, second])
    image.remove_ffs_file(0)
    image.remove_ffs_file(4)
    return results, image


def _ffs_lib(cls):
    lib = cls(date_created="2026-01-01", date_modified="2026-01-02", notes="n")
    empty = cls(date_created="2026-01-01", date_modified="2026-01-02")
    lib.add_image(np.arange(60.0).reshape(6, 10), "image_a.tif")
    lib.add_image(np.ones((4, 8)), "image_b.tif", _metadata(cls))
    lib.add_random_image()
    lib.active_image = 1
    results = [
        lib.num_images,
        lib.get_image(),
        lib.get_image(0),
        lib.get_image(9),
        lib.get_image(-1),
        lib.get_image_name(1),
        lib.get_image_name(9),
        lib.return_image(0),
        lib.return_image(),
        empty.get_image(),
        empty.get_image_name(0),
    ]
    lib.remove_image()
    lib.remove_image(0)
    lib.remove_image(5)
    lib.remove_image()
    return results, lib, empty


def _timetrace(trailing_zeros, rows=300):
    trace = time_trace()[:rows].copy()
    trace[rows - trailing_zeros :] = 0
    return trace


def _green_fill(function):
    fill = function()
    return fill.fill_type, fill.start_color.rgb, fill.end_color.rgb


def _count_trace(value=None):
    trace = time_trace()
    return trace if value is None else np.full_like(trace, value)


CASES: list[Case] = [
    Case(
        f"{PCH}.atimes2pch:arrival_times_to_pch",
        lambda f: f(_seconds(), 1e-3, 0, 0.5, True),
        "normalized",
    ),
    Case(
        f"{PCH}.atimes2pch:arrival_times_to_pch",
        lambda f: f(_seconds(), 1e-3, 1, 0.5, False),
        "counts, second chunk",
    ),
    Case(
        f"{PCH}.atimes2pch:arrival_times_to_pch",
        lambda f: f(_seconds(), 1e-3, 0, 0.5, True, max_k=8),
        "truncated",
    ),
    Case(
        f"{PCH}.atimes2pch:atimes_2_pch_all",
        lambda f: _pch_to_all(f, list_of_pch=["central", "sum3", 12], max_k=60),
        "default names",
    ),
    Case(
        f"{PCH}.atimes2pch:atimes_2_pch_all",
        lambda f: _pch_to_all(
            f,
            list_of_pch=["central", "C3+4+7"],
            normalize=False,
            list_of_pch_out=["a", "b"],
            max_k=40,
        ),
        "counts, renamed",
    ),
    Case(f"{PCH}.data2pch:calc_pch", lambda f: f(_count_trace()), "normalized"),
    Case(
        f"{PCH}.data2pch:calc_pch",
        lambda f: f(_count_trace(), binsize=4, maxcount=60, norm=False),
        "binned counts",
    ),
    Case(
        f"{PCH}.data2pch:calc_pch",
        lambda f: f(_count_trace()[:, 12], binsize=2),
        "one channel",
    ),
    Case(
        f"{PCH}.data2pch:calc_pch",
        lambda f: f(_count_trace(50), maxcount=3),
        "nothing in range",
    ),
    Case(
        f"{PCH}.generate_psf:generate_3d_gaussian",
        lambda f: f((8, 12, 16), 120.0, 300.0, 25.0, 50.0),
    ),
]

SKIPPED: dict[str, str] = {
    f"{PCH}.atimes2pch:atimes_file_2_pch": SKIP_FILE_IO,
    f"{PCH}.data2pch:pch_load_and_hist_split": SKIP_FILE_IO,
    f"{PCH}.pch_fit:fitfun_pch_mc": "TypeError: 'numpy.float64' object cannot be interpreted as an integer",
}

CASES += [
    Case(
        f"{PCH}.simulate_pch:simulate_pch_1c",
        lambda f: f(pch_psf(), c=0.02, q=4.0),
        "defaults",
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_1c",
        lambda f: f(
            _compressed(f)[0],
            dV=_compressed(f)[1],
            k_max=40,
            c=0.02,
            q=4.0,
            T=0.5,
            dV0=2.0,
            n_draws=2,
        ),
        "compressed psf, 40 bins, two draws",
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_nc",
        lambda f: f(pch_psf(), c=[0.02], q=[4.0]),
        "one component",
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_nc",
        lambda f: f(
            pch_psf(),
            k_max=40,
            c=[0.01, 0.02],
            q=[1.5, 6.0],
            T=0.5,
            dV0=2.0,
            bg=0.05,
            n_draws=2,
        ),
        "two components with background",
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_photon_counts_1c_mc",
        lambda f: f(pch_psf(), 0.01, 4.0, 200, bg=0.05, n_jobs=1),
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_1c_mc",
        lambda f: _serial(f, pch_psf(), 0.01, 4.0, 200, 0.05, 31),
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_nc_mc",
        lambda f: _serial(f, pch_psf(), [0.01, 0.005], [4.0, 8.0], 200, 0.05, 31),
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_1c_mc_ntimes",
        lambda f: _serial(_printed, f, pch_psf(), 0.01, 4.0, 100, 0.05, 6, 31, 1.0),
        "stops on the error",
    ),
    Case(
        f"{PCH}.simulate_pch:simulate_pch_1c_mc_ntimes",
        lambda f: _serial(_printed, f, pch_psf(), 0.01, 4.0, 100, 0.05, 6, 31, 1e-12),
        "stops on the simulation count",
    ),
]

CASES += [
    Case(
        f"{PCH}.pch_fit:generate_psfs",
        lambda f: f([250.0, 3.0], 1, 200),
        "one Gaussian",
    ),
    Case(
        f"{PCH}.pch_fit:generate_psfs",
        lambda f: f([250.0, 3.0, 350.0, 2.0], 2, 200),
        "two Gaussians",
    ),
    Case(f"{PCH}.pch_fit:generate_psfs", lambda f: f(pch_psf(), 1, 200), "3D array"),
    Case(f"{PCH}.pch_fit:generate_psfs", lambda f: f(_two_psfs(), 2, 200), "4D array"),
    Case(f"{PCH}.pch_fit:hist_param", lambda f: f(np.arange(30.0)), "one histogram"),
    Case(
        f"{PCH}.pch_fit:hist_param", lambda f: f(np.ones((30, 3))), "three histograms"
    ),
    Case(
        f"{PCH}.pch_fit:parse_fitfun_and_fitinfo",
        lambda f: f("fitfun_pch", [1, 1, 0, 0], np.arange(30.0)),
        "single component",
    ),
    Case(
        f"{PCH}.pch_fit:parse_fitfun_and_fitinfo",
        lambda f: f(_sibling(f, "pch_fit").fitfun_pch, [1, 1, 0, 0], np.arange(30.0)),
        "function object",
    ),
    Case(
        f"{PCH}.pch_fit:parse_fitfun_and_fitinfo",
        lambda f: f("fitfun_pch_nc_global", [1, 1, 0, 0, 0, 0, 0], np.ones((30, 3))),
        "global",
    ),
    Case(
        f"{PCH}.pch_fit:parse_fitfun_and_fitinfo",
        lambda f: f("fitfun_pch_nc", [1, 1, 1, 1, 0, 0, 0], np.arange(30.0)),
        "several components",
    ),
    Case(
        f"{PCH}.pch_fit:make_fit_parameters_global_fit",
        lambda f: f(
            _global_start(),
            GLOBAL_INFO,
            2,
            GLOBAL_PARAM,
            np.arange(5.0),
            np.arange(5.0) + 10,
        ),
    ),
    Case(
        f"{PCH}.pch_fit:make_2D_fit_parameter_array_global_fit",
        lambda f: f(
            [0.1, 2.0, 3.0, 0.05], [1.0, 1.5, 2.0], GLOBAL_INFO, GLOBAL_PARAM, 5, 2
        ),
        "no factors",
    ),
    Case(
        f"{PCH}.pch_fit:make_2D_fit_parameter_array_global_fit",
        lambda f: f(
            [0.1, 2.0, 3.0, 0.05],
            [1.0, 1.5, 2.0],
            GLOBAL_INFO,
            GLOBAL_PARAM,
            5,
            2,
            np.array([1.0, 10.0, 1.0, 0.1, 1.0]),
        ),
        "factors",
    ),
    Case(
        f"{PCH}.pch_fit:global_fit_result_to_parameters",
        lambda f: f(
            _global_start(), GLOBAL_INFO, 2, GLOBAL_PARAM, [0.5, 2.5, 3.5, 0.07]
        ),
    ),
    Case(
        f"{PCH}.pch_fit:global_fit_result_to_residuals",
        lambda f: f(
            _rng().normal(size=60),
            _rng().uniform(0.1, 1, (30, 2)),
            2,
            np.linspace(0.5, 2, 60).reshape(30, 2),
            "relative",
        ),
        "relative",
    ),
    Case(
        f"{PCH}.pch_fit:global_fit_result_to_residuals",
        lambda f: f(
            _rng().normal(size=60), _rng().uniform(0.1, 1, (30, 2)), 2, 1, "absolute"
        ),
        "absolute",
    ),
]

CASES += [
    Case(f"{PCH}.pch_fit:fitfun_pch", lambda f: _fitfun_pch(f, "absolute"), "absolute"),
    Case(
        f"{PCH}.pch_fit:fitfun_pch",
        lambda f: _fitfun_pch(f, "relative", np.linspace(0.5, 2, 30)),
        "relative, weighted",
    ),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc",
        lambda f: _fitfun_nc(f, "absolute", n_draws=2),
        "absolute, two draws",
    ),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc",
        lambda f: _fitfun_nc(f, "relative", weights=np.linspace(0.5, 2, 30)),
        "relative",
    ),
    Case(f"{PCH}.pch_fit:fitfun_pch_nc", lambda f: _fitfun_nc(f, "mle"), "likelihood"),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc",
        lambda f: _fitfun_nc(f, "mle", return_residuals=True),
        "likelihood residuals",
    ),
    Case(f"{PCH}.pch_fit:fitfun_pch_nc", _fitfun_nc_two, "two components"),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc_global",
        lambda f: _fitfun_global(f, "absolute"),
        "absolute",
    ),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc_global",
        lambda f: _fitfun_global(f, "relative", np.linspace(0.5, 2, 60)),
        "relative, weighted",
    ),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc_global",
        lambda f: _fitfun_global(f, "mle"),
        "likelihood",
    ),
    Case(
        f"{PCH}.pch_fit:fitfun_pch_nc_global",
        lambda f: _fitfun_global(f, "mle", return_residuals=True),
        "likelihood residuals",
    ),
]

CASES += [
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch(f, "single", "relative", fitfun="fitfun_pch"),
        "single, relative",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch(f, "single", "absolute", fitfun="fitfun_pch"),
        "single, absolute",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch(f, "background", "relative", fitfun="fitfun_pch_nc"),
        "background",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch(
            f, "one", "absolute", weights=np.linspace(0.5, 2, 30), n_draws=2
        ),
        "weighted, two draws",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch(f, "two", "absolute"),
        "two components",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch(f, "likelihood", "mle"),
        "likelihood",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch_global(f, "relative"),
        "global, relative",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch_global(f, "absolute"),
        "global, absolute",
    ),
    Case(
        f"{PCH}.pch_fit:fit_pch",
        lambda f: _fit_pch_global(f, "mle"),
        "global, likelihood",
    ),
]

CASES += [
    Case(
        f"{TOOLS}.analyze_fit_result:calc_bic",
        lambda f: f(_rng().normal(size=30), _rng().normal(size=30), 0.9, 4),
        "scalar sigma",
    ),
    Case(
        f"{TOOLS}.analyze_fit_result:calc_bic",
        lambda f: f(
            _rng().normal(size=30),
            _rng().normal(size=30),
            _rng().uniform(0.5, 2, size=30),
            3,
        ),
        "sigma per point",
    ),
    Case(
        f"{TOOLS}.bindata:bindata",
        lambda f: f(time_trace()[:1000], 7),
        "consecutive, 2D",
    ),
    Case(
        f"{TOOLS}.bindata:bindata",
        lambda f: f(time_trace()[:1000, 3], 7),
        "consecutive, 1D",
    ),
    Case(
        f"{TOOLS}.bindata:bindata",
        lambda f: f(time_trace()[:1000], 4, interleaved_binning=True),
        "interleaved",
    ),
    Case(f"{TOOLS}.bindata:bindata_chunks", lambda f: f(time_trace(), 10), "2D"),
    Case(f"{TOOLS}.bindata:bindata_chunks", lambda f: f(time_trace()[:, 2], 25), "1D"),
    Case(
        f"{TOOLS}.bindata:bindata_chunks",
        lambda f: _printed(f, time_trace(), 8, True, True),
        "chunk numbers, interleaved",
    ),
    Case(
        f"{TOOLS}.calcconc:calcconc",
        lambda f: _printed(f, 2e-3, 0.05, 300),
        "molar mass",
    ),
    Case(
        f"{TOOLS}.calcconc:calcconc",
        lambda f: _printed(f, 2e-3, 0.05),
        "default molar mass",
    ),
    Case(
        f"{TOOLS}.calcdist_from_coord:calcdist_from_coord",
        lambda f: f(np.zeros((6, 9))),
        "center",
    ),
    Case(
        f"{TOOLS}.calcdist_from_coord:calcdist_from_coord",
        lambda f: f(np.zeros((6, 9)), [1, 7]),
        "given pixel",
    ),
    Case(
        f"{TOOLS}.calcdist_from_coord:calc_shift_from_coord",
        lambda f: [f(3, 12), f(12, 3, 5), f(1, 7, 3)],
    ),
    Case(
        f"{TOOLS}.calcdist_from_coord:list_of_pixel_pairs_at_distance",
        lambda f: f([0, 1]),
        "all pixels",
    ),
    Case(
        f"{TOOLS}.calcdist_from_coord:list_of_pixel_pairs_at_distance",
        lambda f: f([1, 2], 25, [3, 13]),
        "pixels off",
    ),
    Case(
        f"{TOOLS}.calcdist_from_coord:list_of_pixel_pairs_at_distance",
        lambda f: f([-1, 1], 9),
        "3 x 3 array",
    ),
    Case(f"{TOOLS}.cast_data:cast_data", lambda f: f(time_trace()[:100]), "default"),
    Case(
        f"{TOOLS}.cast_data:cast_data",
        lambda f: f(time_trace()[:100], "float32"),
        "float32",
    ),
    Case(
        f"{TOOLS}.checkfname:checkfname",
        lambda f: [
            f("data", "csv"),
            f("data.csv", "csv"),
            f("a", "tiff"),
            f(None, "csv"),
        ],
    ),
    Case(
        f"{TOOLS}.savefigure:checkfname",
        lambda f: [f("figure"), f("figure.pickle"), f("figure.png"), f("a.pickle")],
    ),
    Case(
        f"{TOOLS}.constants:constants",
        lambda f: [f("Boltzmann"), f("planck"), f("AVOGADRO"), f("unknown")],
    ),
    Case(
        f"{TOOLS}.find_nearest:find_nearest",
        lambda f: f([0.5, 1.5, 2.5, 4.0], 2.1),
        "list",
    ),
    Case(
        f"{TOOLS}.find_nearest:find_nearest",
        lambda f: f(np.linspace(-3, 3, 21), -0.34),
        "array",
    ),
    Case(
        f"{TOOLS}.moving_average:moving_average",
        lambda f: f(time_trace()[:200, 4]),
        "default window",
    ),
    Case(
        f"{TOOLS}.moving_average:moving_average",
        lambda f: f(time_trace()[:200, 4], 7),
        "window of 7",
    ),
    Case(
        f"{TOOLS}.path2fname:path2fname",
        lambda f: f(r"C:\data\runs\file.ext"),
        "backslashes",
    ),
    Case(
        f"{TOOLS}.path2fname:path2fname",
        lambda f: f("C:/Users/someone/data/file.ext"),
        "slashes",
    ),
    Case(
        f"{TOOLS}.path2fname:path2fname",
        lambda f: f("some/folder/file.ext", properWay=True),
        "split",
    ),
    Case(f"{TOOLS}.phasor:phasor", lambda f: f(np.arange(1.0, 33.0)), "normalized"),
    Case(
        f"{TOOLS}.phasor:phasor",
        lambda f: f(time_trace()[:64, 0], normalize=False),
        "not normalized",
    ),
    Case(
        f"{TOOLS}.plot_colors:plot_colors",
        lambda f: [f(0), f(7), f("sum3"), f("sum5_average"), f("unknown")],
    ),
    Case(
        f"{TOOLS}.print_tools:print_table",
        lambda f: _printed(f, _rng().normal(size=(3, 5)), 9),
        "all columns",
    ),
    Case(
        f"{TOOLS}.print_tools:print_table",
        lambda f: _printed(f, _rng().normal(size=(2, 12)), 7, 4),
        "truncated",
    ),
    Case(
        f"{TOOLS}.stokes_einstein:stokes_einstein",
        lambda f: [f(1e-10), f(4e-12, 310, 7e-4)],
    ),
    Case(
        f"{TOOLS}.time2string:time2string",
        lambda f: [f(x) for x in (0.5, 1.2, 30, 61, 130, 3601, 7300)],
    ),
    Case(f"{TOOLS}.np2hdf5:DataFileParam", _data_file_param),
]

CASES += [
    Case(
        f"{TOOLS}.fit_curve:fit_curve",
        lambda f: _fit_curve(
            f, "powerlaw", [1, 1, 0], [2.0, -1.0, 0.0], [0, -10, -10], [100, 0, 100]
        ),
        "power law",
    ),
    Case(
        f"{TOOLS}.fit_curve:fit_curve",
        lambda f: _fit_curve(
            f,
            "exp",
            [1, 1, 0, 1],
            [4.0, 0.2, 0.5, 0.1],
            [0, 0, -1, -1],
            [100, 10, 1, 10],
        ),
        "exponential",
    ),
    Case(
        f"{TOOLS}.fit_curve:fit_curve",
        lambda f: _fit_curve(f, "linear", [1, 1], [1.0, 0.0], [-10, -10], [10, 10]),
        "linear",
    ),
    Case(
        f"{TOOLS}.fit_curve:fit_power_law_g",
        lambda f: _fit_curve_g(f, 3, False),
        "three curves",
    ),
    Case(
        f"{TOOLS}.fit_curve:fit_power_law_g",
        lambda f: _fit_curve_g(f, 3, True),
        "three curves, plotted",
    ),
    Case(
        f"{TOOLS}.fit_curve:fit_power_law_g",
        lambda f: _fit_curve_g(f, 25, False),
        "25 curves",
    ),
    Case(
        f"{TOOLS}.fit_curve:fitfun_powerlaw",
        lambda f: _residuals(f, [3.0, -1.2], [0.05], [1, 1, 0], *_curve("powerlaw")),
    ),
    Case(
        f"{TOOLS}.fit_curve:fitfun_exp",
        lambda f: _residuals(f, [4.0, 0.2, 0.1], [0.5], [1, 1, 0, 1], *_curve("exp")),
    ),
    Case(
        f"{TOOLS}.fit_curve:fitfun_exp_n",
        lambda f: _residuals(
            f, [4.0, 0.2, 1.0, 0.05, 0.1], [0.5], [1, 1, 1, 1, 0, 1], *_curve("exp")
        ),
    ),
    Case(
        f"{TOOLS}.fit_curve:fitfun_linear",
        lambda f: _residuals(f, [1.5], [0.5], [1, 0], *_curve("linear")),
    ),
    Case(
        f"{TOOLS}.fit_curve:fun_exp_n",
        lambda f: [
            f(_curve("exp")[0], [4.0, 0.2, 0.5, 0.1]),
            f(_curve("exp")[0], [4.0, 0.2, 1.0, 0.05, 0.5, 0.1]),
        ],
        "one and two components",
    ),
]

CASES += [
    Case(
        f"{TOOLS}.fit_data:fit_data",
        lambda f: f(
            *_curve("exp")[::-1],
            "exp",
            [1, 1, 0],
            [4.0, 0.2, 0.1],
            [0, 0, -1],
            [100, 10, 10],
        ),
    ),
    Case(
        f"{TOOLS}.fit_data:fit_and_plot",
        lambda f: _drawn(
            lambda: _printed(f, *_curve("exp")[::-1], "exp", [1, 1, 1], [4.0, 0.2, 0.1])
        ),
        "printed and plotted",
    ),
    Case(
        f"{TOOLS}.fit_data:fit_and_plot",
        lambda f: f(
            *_curve("exp")[::-1],
            "exp",
            [1, 1, 0],
            [4.0, 0.2, 0.1],
            printResults=0,
            plot_fit=0,
        ),
        "quiet",
    ),
    Case(
        f"{TOOLS}.fit_data:fitfun_exp_n",
        lambda f: _residuals(
            f, [4.0, 0.2, 1.0, 0.05], [0.1], [1, 1, 1, 1, 0], *_curve("exp")
        ),
    ),
    Case(
        f"{TOOLS}.fit_doughnut2d:doughnut2d",
        lambda f: f(20, 16, *BLOB_TRUTH),
    ),
    Case(
        f"{TOOLS}.fit_doughnut2d:fitfun_doughnut2d",
        lambda f: _blob_residuals(f, "fit_doughnut2d", "doughnut2d"),
    ),
    Case(
        f"{TOOLS}.fit_doughnut2d:fit_dougnut2D",
        lambda f: _blob_fit(f, "fit_doughnut2d", "doughnut2d", [1, 1, 1, 1, 0]),
        "unweighted",
    ),
    Case(
        f"{TOOLS}.fit_doughnut2d:fit_dougnut2D",
        lambda f: _blob_fit(
            f, "fit_doughnut2d", "doughnut2d", [0, 0, 1, 1, 1], BLOB_WEIGHTS
        ),
        "weighted, fixed center",
    ),
    Case(f"{TOOLS}.fit_gauss_2d:normal2d", lambda f: f(20, 16, *BLOB_TRUTH)),
    Case(
        f"{TOOLS}.fit_gauss_2d:fitfun_gauss2d",
        lambda f: _blob_residuals(f, "fit_gauss_2d", "normal2d"),
    ),
    Case(
        f"{TOOLS}.fit_gauss_2d:fit_gauss_2d",
        lambda f: _blob_fit(f, "fit_gauss_2d", "normal2d", [1, 1, 1, 1, 0]),
        "unweighted",
    ),
    Case(
        f"{TOOLS}.fit_gauss_2d:fit_gauss_2d",
        lambda f: _blob_fit(
            f, "fit_gauss_2d", "normal2d", [0, 0, 1, 1, 1], BLOB_WEIGHTS
        ),
        "weighted, fixed center",
    ),
    Case(
        f"{TOOLS}.fit_polynomial_2d:polynomial2d",
        lambda f: f(np.zeros((12, 10)), 2, 2, POLY_COEFFS),
    ),
    Case(
        f"{TOOLS}.fit_polynomial_2d:fitfun_polynomial2d",
        lambda f: f(
            POLY_COEFFS + 0.1,
            2,
            2,
            _poly_data(f),
            np.linspace(0.5, 2, 120).reshape(12, 10),
        ),
    ),
    Case(
        f"{TOOLS}.fit_polynomial_2d:fit_polynomial2d",
        lambda f: f(_poly_data(f), 2, 2),
        "unweighted",
    ),
    Case(
        f"{TOOLS}.fit_polynomial_2d:fit_polynomial2d",
        lambda f: f(_poly_data(f), 2, 2, np.linspace(0.5, 2, 120).reshape(12, 10)),
        "weighted",
    ),
]

CASES += [
    Case(
        f"{TOOLS}.fit_psf:fit_psf",
        lambda f: _clicked_fit_psf(f, bead_image().sum(axis=2)),
        "2D",
    ),
    Case(f"{TOOLS}.fit_psf:fit_psf", lambda f: _clicked_fit_psf(f, bead_image()), "3D"),
    Case(
        f"{TOOLS}.fit_psf:checkrange",
        lambda f: [f(50, 15, 0, 100), f(10, 15, 0, 100), f(95, 15, 0, 100)],
    ),
    Case(
        f"{TOOLS}.fit_psf:gaussian",
        lambda f: f(21, 19, 10.5, 9.0, 6.0, 5.0, 100.0, 3.0, 0.05),
    ),
    Case(
        f"{TOOLS}.fit_psf:gaussian3d",
        lambda f: f(12, 14, 8, 5.5, 6.5, 3.5, 5.0, 6.0, 30.0, 0.05, 0.02, 50.0, 2.0),
    ),
    Case(
        f"{TOOLS}.fit_psf:gaussfitfun2d",
        lambda f: f(
            np.array(PSF_START[:6]),
            np.array([0.0]),
            np.array([1, 1, 1, 1, 1, 1, 0]),
            _psf_image(f),
        ),
    ),
    Case(
        f"{TOOLS}.fit_psf:gaussfitfun3d",
        lambda f: f(
            np.array([5.0, 6.0, 3.0, 5.0, 6.0, 30.0, 0.0, 0.0, 40.0, 1.0]),
            5.0,
            _psf_image(f, three_d=True),
        ),
    ),
    Case(
        f"{TOOLS}.fit_psf:gauss_fitres",
        lambda f: f(_psf_image(f), np.array(PSF_START)),
        "all parameters",
    ),
    Case(
        f"{TOOLS}.fit_psf:gauss_fitres",
        lambda f: f(
            _psf_image(f), np.array(PSF_START), np.array([1, 1, 1, 1, 1, 1, 0])
        ),
        "fixed rotation",
    ),
    Case(
        f"{TOOLS}.fit_psf:gauss_fitres3d",
        lambda f: f(
            _psf_image(f, three_d=True),
            [5.0, 6.0, 3.0, 5.0, 6.0, 30.0, 0.0, 0.0, 40.0, 1.0],
        ),
    ),
]

CASES += [
    Case(
        f"{TOOLS}.cmap2d:cmap2d",
        lambda f: _color_image(
            f, ltimeBound=["auto", "auto"], intBound=["auto", "auto"]
        ),
        "automatic bounds",
    ),
    Case(
        f"{TOOLS}.cmap2d:cmap2d",
        lambda f: _color_image(
            f, ltimeBound=[2.0, 4.0], intBound=[20.0, 80.0], cmap="viridis"
        ),
        "given bounds",
    ),
    Case(
        f"{TOOLS}.cmap2d:cmap2d",
        lambda f: _color_image(
            f,
            ltimeBound=["auto", 4.5],
            intBound=[10.0, "auto"],
            cmap="jet",
            equalize_lum=False,
        ),
        "half automatic, not equalized",
    ),
    # the old code leaves the image at zeros in this branch, so the case pins that output
    Case(
        f"{TOOLS}.cmap2d:cmap2d",
        lambda f: _color_image(
            f,
            ltimeBound=["auto", "auto"],
            intBound=["auto", "auto"],
            cmap="seaborn-colorblind",
        ),
        "colorblind",
    ),
    Case(
        f"{TOOLS}.cmap2d:cmap2d",
        lambda f: _drawn(
            lambda: _color_image(
                f, ltimeBound=["auto", "auto"], intBound=["auto", "auto"], plotFig=True
            )
        ),
        "plotted",
    ),
    Case(
        f"{TOOLS}.cmaps:color_gradient",
        lambda f: f((0, 0, 0, 1), (1.0, 0.5, 0.2, 1), 8),
    ),
    Case(
        f"{TOOLS}.cmaps:change_color_from_map",
        lambda f: f("viridis", [1.0, 0.0, 0.0]),
        "first color",
    ),
    Case(
        f"{TOOLS}.cmaps:change_color_from_map",
        lambda f: f("inferno", [0.0, 1.0, 0.0], idx=5),
        "other index",
    ),
    Case(
        f"{TOOLS}.cmaps:change_color_from_map",
        lambda f: f("seaborn-colorblind", [0.0, 1.0, 0.0]),
        "colorblind",
    ),
    Case(f"{TOOLS}.color_from_map:color_from_map", lambda f: f(0.3), "default"),
    Case(f"{TOOLS}.color_from_map:color_from_map", lambda f: f(1.0), "end of the map"),
    Case(
        f"{TOOLS}.color_from_map:color_from_map",
        lambda f: f(5.0, 2.0, 10.0, "jet"),
        "map without a color list",
    ),
    Case(
        f"{TOOLS}.color_from_map:color_from_map",
        lambda f: f(0.4, cmap="seaborn-colorblind"),
        "colorblind",
    ),
    Case(
        f"{TOOLS}.draw_ellipsoid:draw_ellipsoid",
        lambda f: f(1.0, 2.0, 3.0, False, 12),
        "data only",
    ),
    Case(
        f"{TOOLS}.draw_ellipsoid:draw_ellipsoid",
        lambda f: _drawn(lambda: f(1.0, 2.0, 3.0, res=12)),
        "plotted",
    ),
    Case(f"{TOOLS}.exp_psf:exppsf", lambda f: _printed(f, bead_image()), "channels"),
    Case(
        f"{TOOLS}.exp_psf:exppsf",
        lambda f: _printed(f, bead_image()[:, :, 0], 12, blur=7),
        "one plane",
    ),
    Case(
        f"{TOOLS}.exp_psf:exppsf",
        lambda f: _drawn(lambda: _printed(f, bead_image(), plots=True)),
        "plotted",
    ),
    Case(
        f"{TOOLS}.exp_psf:find_particles",
        lambda f: _printed(f, bead_image().sum(axis=2), 11, [20, 255]),
        "beads",
    ),
    Case(
        f"{TOOLS}.exp_psf:find_particles",
        lambda f: _printed(f, bead_image().sum(axis=2), 11, [20, 255], returnAll=True),
        "all results",
    ),
    Case(
        f"{TOOLS}.exp_psf:find_particles",
        lambda f: _printed(f, bead_image().sum(axis=2), 3, [254, 255]),
        "no particles",
    ),
]

CASES += [
    Case(f"{TOOLS}.cmaps:cmaps", lambda f, name=name: f(name, 16), name)
    for name in COLOR_MAPS
]

SKIPPED |= {
    f"{TOOLS}.array2tiff:array2tiff": SKIP_FILE_IO,
    f"{TOOLS}.array2tiff:array2rgbtiff": SKIP_FILE_IO,
    f"{TOOLS}.array2tiff:tiff2array": SKIP_FILE_IO,
    f"{TOOLS}.array2video:array2video": SKIP_FILE_IO,
    f"{TOOLS}.closefile:closefile": SKIP_FILE_IO,
    f"{TOOLS}.csv2array:array2csv": SKIP_FILE_IO,
    f"{TOOLS}.csv2array:csv2array": SKIP_FILE_IO,
    f"{TOOLS}.goto:goto": f"{SKIP_FILE_IO}: changes the working directory to a fixed lab path",
    f"{TOOLS}.h5data:h5data": SKIP_FILE_IO,
    f"{TOOLS}.h5data:printmetadata": SKIP_FILE_IO,
    f"{TOOLS}.list_files:list_files": SKIP_FILE_IO,
    f"{TOOLS}.np2hdf5:hdf5_read": SKIP_FILE_IO,
    f"{TOOLS}.np2hdf5:np2hdf5": SKIP_FILE_IO,
    f"{TOOLS}.phasor:plotphasor": SKIP_PLOT_ONLY,
    f"{TOOLS}.savefigure:openfig": SKIP_FILE_IO,
    f"{TOOLS}.savefigure:savefig": SKIP_FILE_IO,
    f"{TOOLS}.savevar:openvar": SKIP_FILE_IO,
    f"{TOOLS}.savevar:savevar": SKIP_FILE_IO,
}

SETTINGS = f"{GUI}.analysis_settings"
CASES += [
    Case(f"{SETTINGS}:FFSmetadata", _metadata_case),
    Case(
        f"{GUI}.correlation_functions_class:CorrelationFunction", _correlation_function
    ),
    Case(f"{GUI}.fitmodel_class:FitModel", _fit_model_case),
    Case(f"{SETTINGS}:CorrSettings", _settings),
    Case(f"{SETTINGS}:FitSingleObj", _fit_single),
    Case(f"{SETTINGS}:CorrFit", _corr_fit_results, "curves and fit results"),
    Case(f"{SETTINGS}:CorrFit", _corr_fit_mem, "maximum entropy"),
    Case(f"{SETTINGS}:CorrFit", _corr_fit_msd, "mean squared displacement"),
    Case(f"{SETTINGS}:CorrFit", _corr_fit_mfda, "model-free displacement"),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls, ("Right", "Up", "Left", "Down"), "fitresults_asymmetrymap", _directions
        ),
        "asymmetry map, square",
    ),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls, ("a", "b", "c", "d"), "fitresults_asymmetrymap", _directions
        ),
        "asymmetry map, curves not found",
    ),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls, ("Right", "Up", "Left", "Down"), "fitresults_flowmap", _directions
        ),
        "flow map, square",
    ),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls,
            ("V0_H1", "Vn1_H0", "V0_Hn1", "V1_H0"),
            "fitresults_flowmap",
            _directions,
        ),
        "flow map, square with pixel names",
    ),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls,
            ("UpRight", "UpLeft", "DownLeft", "DownRight"),
            "fitresults_flowmap",
            _directions,
        ),
        "flow map, airy",
    ),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls,
            ("Angle0", "Angle60", "Angle120", "Angle180", "Angle240", "Angle300"),
            "fitresults_flowmap",
            _directions,
        ),
        "flow map, six directions",
    ),
    Case(
        f"{SETTINGS}:CorrFit",
        lambda cls: _corr_fit_map(
            cls, ("a", "b", "c", "d"), "fitresults_flowmap", _directions
        ),
        "flow map, curves not found",
    ),
    Case(f"{SETTINGS}:FFScorr", _ffs_corr, "correlations and fits"),
    Case(f"{SETTINGS}:FFScorr", _ffs_corr_cross, "cross-correlation settings"),
    Case(f"{SETTINGS}:FFSfile", _ffs_file_session, "analyses"),
    Case(f"{SETTINGS}:FFSfile", _ffs_file_fit_as_data, "fit as data"),
    Case(f"{SETTINGS}:FFSimage", _ffs_image),
    Case(f"{SETTINGS}:FFSlib", _ffs_lib),
    Case(
        f"{GUI}.timetrace_end:timetrace_end",
        lambda f: [f(_timetrace(k)) for k in (0, 1, 7, 49, 60)],
    ),
    Case(
        f"{GUI}.restore_session:check_none",
        lambda f: [
            f("5", int),
            f(None),
            f("None"),
            f("a"),
            f(3.5, float),
            f("None", int),
            f(7),
        ],
    ),
    Case(
        f"{GUI}.restore_session:to_text", lambda f: [f(None), f(5), f("a"), f([1, 2])]
    ),
    Case(f"{GUI}.session_to_excel:next_letter", lambda f: [f("A"), f("Y")]),
    Case(f"{GUI}.session_to_excel:green_fill", _green_fill),
]

SKIPPED |= {
    f"{GUI}.load_ffs_metadata:load_ffs_metadata": SKIP_FILE_IO,
    f"{GUI}.read_ffs:read_ffs_file": SKIP_FILE_IO,
    f"{GUI}.read_ffs:read_fitresults_from_ffs": SKIP_FILE_IO,
    f"{GUI}.read_ffs:read_g_from_ffs": SKIP_FILE_IO,
    f"{GUI}.restore_session:create_dir": SKIP_FILE_IO,
    f"{GUI}.restore_session:make_zip": SKIP_FILE_IO,
    f"{GUI}.restore_session:read_text": SKIP_FILE_IO,
    f"{GUI}.restore_session:restorelib": SKIP_FILE_IO,
    f"{GUI}.restore_session:restorelib_ffs": SKIP_FILE_IO,
    f"{GUI}.restore_session:restorelib_ffz": SKIP_FILE_IO,
    f"{GUI}.restore_session:save_ffs": SKIP_GUI,
    f"{GUI}.restore_session:save_image": SKIP_FILE_IO,
    f"{GUI}.restore_session:save_text": SKIP_FILE_IO,
    f"{GUI}.restore_session:savelib": SKIP_FILE_IO,
    f"{GUI}.restore_session:savelib_ffs": SKIP_FILE_IO,
    f"{GUI}.restore_session:savelib_ffz": SKIP_FILE_IO,
    f"{GUI}.restore_session:unzip": SKIP_FILE_IO,
    f"{GUI}.session_to_excel:corr2excel": SKIP_FILE_IO,
    f"{GUI}.session_to_excel:ffs2excel": SKIP_FILE_IO,
    f"{GUI}.session_to_excel:lib2excel": SKIP_FILE_IO,
}
