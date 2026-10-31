"""Equivalence cases for `s2ism`, `flus2ISM_lib` and `psf_ism_distiller`."""

import tempfile
from functools import partial
from importlib import import_module
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch
from _capture import printed as _printed
from _inputs import (
    flus2ism_inputs,
    ism_dataset,
    ism_psf,
    mcs_h5_file,
    mcs_legacy_h5_file,
    shifted_spot_dataset,
)
from _registry import Case
from scipy.stats import qmc
from torch.fft import rfftn

S2ISM = "s2ism.s2ism"
ESTIMATOR = "s2ism.psf_estimator"
MAG = "s2ism.mag_finder"
MISALIGNMENT = "s2ism.misalignment"
SHIFTS = "s2ism.shift_vectors_minimizer"
DATAIO = "s2ism.dataio"
FLUS2ISM = "flus2ISM_lib"
DISTILLER = "psf_ism_distiller.distiller"


def _raised(call):
    try:
        call()
    except (RuntimeError, ValueError) as error:
        return type(error).__name__, str(error)
    raise AssertionError("no exception raised")


def _module(obj):
    return import_module(obj.__module__)


def _sim(obj):
    return import_module(obj.__module__.rpartition(".")[0] + ".psf_estimator").sim


def _settings(sim, **changes):
    return sim.simSettings(mask_sampl=32, **changes)


def _grid(sim, **changes):
    return sim.GridParameters(**{"Nx": 41, "Nz": 2, "N": 3, "pxsizex": 60, **changes})


def _theory_shifts(module, geometry, transform):
    coordinates = -np.swapaxes(module.det_coords(5, geometry), 0, 1)
    return module.transform_shift_vectors(
        transform, coordinates
    ) + np.random.default_rng(0).normal(0, 0.02, coordinates.shape)


def _scalar_psf_inputs():
    return np.linspace(-500.0, 500.0, 101), 640.0, 1.4


def _detector_psf(module):
    r, wl, na = _scalar_psf_inputs()
    return module.scalar_psf_det(r, wl, na, 50e3, 75e3, 450.0)


def _minimizer_inputs(module, tolerance=1e-6, **options):
    return _printed(
        lambda: module.loss_minimizer(
            60.0, 640.0, 660.0, 75e3, 50e3, 1.4, 500, tolerance, options
        )
    )


def _shift_loss_minimizer(module, mirror, **options):
    shift_theory = module.shift_matrix()
    shift_experiment = module.transform_shift_vectors([2.5, 0.4, mirror], shift_theory)
    return _printed(
        lambda: module.loss_minimizer(
            shift_experiment, shift_theory, 2, 0.5, 1e-6, options, mirror
        )
    )


def _crop_shift(module, geometry):
    count = {"rect": 25, "hex": 23}[geometry]
    rng = np.random.default_rng(0)
    return module.crop_shift(rng.uniform(-1, 1, (count, 2)), geometry)


def _find_parameters(module, geometry, transform):
    shifts = _theory_shifts(module, geometry, transform)
    return module.find_parameters(shifts, geometry)


def _fingerprint(dataset):
    return dataset.sum((0, 1)).reshape(3, 3)


def _mirrored_dataset():
    return shifted_spot_dataset().reshape(41, 41, 3, 3)[:, :, ::-1].reshape(41, 41, 9)


def _gaussian_inputs():
    y, x = np.mgrid[0:7, 0:9]
    return x, y


def _find_misalignment(function, dataset):
    return function(dataset, 75e3, 400.0, 1.4, 640.0)


def _grid_finder(finder):
    estimator = finder(_grid(_sim(finder)))
    estimator.estimate(shifted_spot_dataset(), 485.0, 600.0, 1.4)
    return estimator, _printed(estimator.Print), finder()


def _psf_from_data(function, **options):
    sim = _sim(function)
    excitation = _settings(sim)
    emission = _settings(sim, wl=600.0)
    grid = _grid(sim, pxsizez=300)
    result = _printed(
        lambda: function(shifted_spot_dataset(), excitation, emission, grid, **options)
    )
    return result, excitation, emission, grid


def _sted_settings(function, **options):
    sim = _sim(function)
    return _psf_from_data(function, stedPar=_settings(sim, wl=700.0), **options)


def _defocus_stack(module, mode):
    sim = module.sim
    grid = _grid(sim, Nx=15, Nz=60, pxsizex=100, pxsizez=25)
    stack = sim.SPAD_PSF_3D(
        grid, _settings(sim), _settings(sim, wl=600.0), stack="positive"
    )
    return module.find_out_of_focus(grid, list(stack), mode=mode, graph=mode == "KL")


def _from_param(module, **options):
    sim = module.sim
    return module.find_out_of_focus_from_param(
        exPar=_settings(sim), emPar=_settings(sim, wl=600.0), **options
    )


def _from_param_grid(module, **options):
    sim = module.sim
    grid = _grid(sim, Nx=15, Nz=20, pxsizex=100, pxsizez=40)
    return module.find_out_of_focus_from_param(
        pxsizex=100,
        exPar=_settings(sim),
        emPar=_settings(sim, wl=600.0),
        grid=grid,
        **options,
    )


def _from_param_sted(module):
    sim = module.sim
    return module.find_out_of_focus_from_param(
        100,
        _settings(sim),
        _settings(sim, wl=600.0),
        stedPar=_settings(sim, wl=700.0),
        mode="KL",
        stack="positive",
    )


def _max_discrepancy(module, mode, graph):
    sim = module.sim
    grid = _grid(sim, Nz=30, pxsizez=25)
    curve = np.cos(np.linspace(0, 3 * np.pi, 30)) + np.linspace(0, 1, 30)
    return module.find_max_discrepancy(curve, grid, mode, graph)


def _conditioning(module, mode, stack):
    sim = module.sim
    grid = _grid(sim, Nx=16, Nz=3, pxsizex=200, pxsizez=300)
    return module.conditioning(
        grid, _settings(sim), _settings(sim, wl=600.0), mode=mode, stack=stack
    ), grid


def _conditioning_sted(module):
    sim = module.sim
    grid = _grid(sim, Nx=16, Nz=3, pxsizex=200, pxsizez=300)
    return module.conditioning(
        grid,
        _settings(sim),
        _settings(sim, wl=600.0),
        _settings(sim, wl=700.0),
        mode="KL",
        stack="symmetrical",
    )


def _conditioning_input_psf(module, mode):
    rng = np.random.default_rng(0)
    grid = _grid(module.sim, Nz=3)
    stack = [
        rng.uniform(0.1, 1, (3, 9, 9, 4)),
        rng.uniform(0.1, 1, (3, 9, 9, 4)),
        rng.uniform(0.1, 1, (3, 9, 9)),
    ]
    return module.conditioning(grid, mode=mode, input_psf=stack), stack


def _combine_psf_irf(function):
    rng = np.random.default_rng(0)
    return function(rng.uniform(0, 1, (2, 5, 4, 9)), rng.uniform(0, 1, (6, 9)))


def _psf_stack():
    psf = ism_psf()
    return np.stack([psf, np.roll(psf, 2, axis=0)])[:, :31, :31]


def _reconstruction(function, dataset=None, psf=None, **options):
    dataset = ism_dataset() if dataset is None else dataset
    psf = _psf_stack() if psf is None else psf
    return _printed(
        lambda: function(dataset, psf, **{"max_iter": 3, "process": "cpu", **options})
    )


def _reconstruction_padded(function, **options):
    return _reconstruction(
        function, ism_dataset()[:31, :31], _psf_stack()[:, 5:26, 5:26], **options
    )


def _batch(function):
    return _printed(
        lambda: function(
            ism_dataset(),
            _psf_stack()[:, 10:21, 10:21],
            [20, 20],
            4,
            max_iter=2,
            process="cpu",
        )
    )


def _data_driven(function, **options):
    sim = _sim(function)
    excitation = _settings(sim)
    emission = _settings(sim, wl=600.0)
    grid = _grid(sim, pxsizez=300)
    result = _printed(
        lambda: function(
            shifted_spot_dataset(),
            grid,
            excitation,
            emission,
            **{"max_iter": 3, "z_out_of_focus": 300.0, **options},
        )
    )
    return result, grid


def _stop_inputs():
    rng = np.random.default_rng(0)
    old = torch.from_numpy(rng.uniform(1, 2, (3, 6, 6)))
    return old, old.clone() + torch.from_numpy(rng.uniform(0, 5, (3, 6, 6)))


def _amd_stop(function, stop, k, same, pre_flag=True, threshold=1e-3, max_iter=5):
    old, new = _stop_inputs()
    if same:
        new = old.clone()
    return _printed(
        lambda: function(
            old, new, pre_flag, True, stop, max_iter, threshold, float(old.sum()), 3, k
        )
    )


def _amd_update_inputs():
    psf = torch.from_numpy(_psf_stack())
    image = torch.from_numpy(ism_dataset()[:31, :31])
    obj = torch.from_numpy(np.random.default_rng(0).uniform(1, 2, (2, 31, 31)))
    eps = torch.finfo(torch.float).eps
    forward = rfftn(psf, dim=[1, 2], s=[31, 31])
    backward = rfftn(torch.flip(psf, [1, 2]), dim=[1, 2], s=[31, 31])
    return image, obj, forward, backward, eps


def _amd_update(function):
    return function(*_amd_update_inputs())


def _convolution_inputs():
    rng = np.random.default_rng(0)
    return (
        torch.from_numpy(rng.uniform(0, 1, (8, 8, 3))),
        torch.from_numpy(rng.uniform(0, 1, (4, 8, 8))),
    )


def _convolution(function, fourier=(False, False), **options):
    kernel, volume = _convolution_inputs()
    if fourier[0]:
        kernel = rfftn(kernel, dim=[0, 1], s=[8, 8])
    if fourier[1]:
        volume = rfftn(volume, dim=[1, 2], s=[8, 8])
    return _printed(
        lambda: function(kernel, volume, "xyc", "zxy", "xy", fourier=fourier, **options)
    )


def _mcs_signal(function, **options):
    with tempfile.TemporaryDirectory() as folder:
        return function(mcs_h5_file(Path(folder) / "file.h5"), **options)


def _mcs_legacy(function, **options):
    with tempfile.TemporaryDirectory() as folder:
        return function(mcs_legacy_h5_file(Path(folder) / "file.h5"), **options)


def _plain_h5(function):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "plain.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("values", data=np.arange(4))
        return function(path)


def _loaded_mcs(function):
    with tempfile.TemporaryDirectory() as folder:
        path = mcs_h5_file(Path(folder) / "file.h5")
        dataset, metadata = function(path, data_format="h5")
        try:
            content = (dataset.name, dataset.shape, dataset[...])
        finally:
            dataset.file.close()
        return content, metadata


def _flus2ism_data():
    psf, phantom = flus2ism_inputs()
    decay = np.exp(-np.arange(8)[None] * 0.3 / np.array([3.0, 6.0])[:, None])
    irf = decay[:, :, None] * (1 + 0.1 * np.arange(psf.shape[-1]))
    psf_irf = np.stack([np.einsum("jkl,mnl->mjknl", plane, irf) for plane in psf])
    psf_irf /= psf_irf.sum(axis=(2, 3, 4, 5), keepdims=True)

    def convolve(kernel, truth):
        product = (
            np.fft.fft2(kernel, axes=(1, 2))
            * np.fft.fft2(truth, axes=(1, 2))[..., None, None]
        )
        return np.fft.fftshift(np.fft.ifft2(product, axes=(1, 2)).real, axes=(1, 2))

    truth = phantom * 100
    clean = (convolve(psf_irf[0], truth[:2]) + convolve(psf_irf[1], truth[2:])).sum(0)
    dataset = np.random.default_rng(0).poisson(np.clip(clean, 0, None))
    return dataset.astype(np.float64), psf_irf


def _lifetime(function, crop=None, **options):
    dataset, psf = _flus2ism_data()
    if crop is not None:
        dataset, psf = dataset[:20, :20], psf[:, :, 5:16, 5:16]
    return _printed(
        lambda: function(dataset, psf, **{"max_iter": 3, "process": "cpu", **options})
    )


def _pulse_data():
    dataset, psf = _flus2ism_data()
    return dataset[:, :1], psf[:, :, None, :, :1]


def _pulse(function, even=False, padded=False, **options):
    dataset, psf = _pulse_data()
    if even:
        dataset = dataset[:20]
    if even or padded:
        psf = psf[:, :, :, 5:16]
    return _printed(
        lambda: function(dataset, psf, **{"max_iter": 3, "process": "cpu", **options})
    )


def _pulse_batch(function, **options):
    dataset, psf = _pulse_data()
    return _printed(
        lambda: function(
            dataset[:20],
            psf[:, :, :, 8:13],
            [15, 15],
            5,
            **{"max_iter": 2, "process": "cpu", **options},
        )
    )


def _lifetime_update(function):
    dataset, psf = _flus2ism_data()
    image = torch.from_numpy(dataset)
    obj = torch.from_numpy(np.random.default_rng(0).uniform(1, 2, (2, 2, 21, 21)))
    psf = torch.from_numpy(psf)
    forward = rfftn(psf, dim=[2, 3])
    backward = rfftn(torch.flip(psf, [2, 3]), dim=[2, 3])
    return _printed(
        lambda: function(
            image, obj, forward, backward, torch.finfo(torch.float).eps, "cpu"
        )
    )


def _pulse_update(function):
    dataset, psf = _pulse_data()
    image = torch.from_numpy(dataset)
    obj = torch.from_numpy(np.random.default_rng(0).uniform(1, 2, (2, 2, 21, 1)))
    psf = torch.from_numpy(psf)
    forward = rfftn(psf, dim=[3, 4])
    backward = rfftn(torch.flip(psf, [3, 4]), dim=[3, 4])
    return _printed(
        lambda: function(
            image, obj, forward, backward, torch.finfo(torch.float).eps, "cpu"
        )
    )


def _seeded(function):
    with (
        mock.patch.object(qmc, "Sobol", partial(qmc.Sobol, seed=0)),
        mock.patch.object(qmc, "LatinHypercube", partial(qmc.LatinHypercube, seed=0)),
    ):
        return function()


def _distiller_inputs(distiller, nz=1, **grid_changes):
    sim = distiller.sim
    excitation = sim.simSettings(mask_sampl=32, wl=640.0)
    excitation.abe_index = np.arange(4)
    emission = excitation.copy()
    emission.wl = 660.0
    grid = sim.GridParameters(
        Nx=16, Nz=nz, N=3, pxsizex=40, pxsizez=100, pxdim=50e3, **grid_changes
    )
    return excitation, emission, grid, sim.custom_detector(grid, "cpu")


def _coefficients():
    return torch.tensor([0.5, -0.3, 0.2])


def _forward(distiller, nz=1, use_object=False, independent=False, **options):
    excitation, emission, grid, spad = _distiller_inputs(distiller, nz)
    coefficients = _coefficients()
    if independent:
        coefficients = torch.stack([coefficients, 2 * coefficients.flip(0)])
    obj = distiller.disk(grid, 100.0) if use_object else None
    result = distiller.forward(
        coefficients,
        excitation,
        emission,
        grid,
        spad=spad,
        obj=obj,
        independent_wavefront=independent,
        **options,
    )
    return result, excitation.abe_ampli, emission.abe_ampli


def _vectorized_forward(distiller, independent):
    excitation, emission, grid, spad = _distiller_inputs(distiller)
    coefficients = torch.stack([_coefficients(), 2 * _coefficients()])
    if independent:
        coefficients = torch.stack([coefficients, coefficients.flip(1)], dim=1)
    return distiller.vectorized_forward(
        coefficients,
        excitation,
        emission,
        grid,
        spad,
        independent_wavefront=independent,
    )


def _measured(distiller, nz=1, independent=False, use_object=False):
    excitation, emission, grid, spad = _distiller_inputs(distiller, nz)
    obj = distiller.disk(grid, 100.0) if use_object else None
    coefficients = torch.tensor([0.4, 0.1, -0.2])
    if independent:
        coefficients = torch.stack([coefficients, -coefficients])
    data = distiller.forward(
        coefficients,
        excitation,
        emission,
        grid,
        spad=spad,
        obj=obj,
        independent_wavefront=independent,
    ).detach()
    return data, excitation, emission, grid, spad, obj


def _loss(distiller, loss="mse", fit_mag=False, vectorized=False, independent=False):
    data, excitation, emission, grid, spad, _ = _measured(
        distiller, independent=independent
    )
    coefficients = _coefficients()
    if independent:
        coefficients = torch.stack([coefficients, coefficients.flip(0)])
    if vectorized:
        coefficients = torch.stack([coefficients, 2 * coefficients])
    if fit_mag:
        coefficients = torch.cat([torch.tensor([420.0]), coefficients])
    result = distiller.loss_function(
        coefficients,
        data,
        excitation,
        emission,
        grid,
        spad,
        fit_mag=fit_mag,
        loss=loss,
        independent_wavefront=independent,
        vectorized=vectorized,
    )
    return result, grid.M


def _minimizer(distiller, independent=False, verbose=False):
    data, excitation, emission, grid, spad, _ = _measured(
        distiller, independent=independent
    )
    start = torch.zeros(6 if independent else 3)
    return _printed(
        lambda: distiller.minimizer(
            start,
            4,
            verbose=verbose,
            data=data,
            expar=excitation,
            empar=emission,
            grid=grid,
            spad=spad,
            obj=None,
            independent_wavefront=independent,
            fit_mag=False,
            loss="mae",
            vectorized=False,
            process="cpu",
        )
    )


def _fit_psf(distiller, independent=False, use_object=False, **options):
    data, excitation, emission, grid, spad, _ = _measured(
        distiller, independent=independent, use_object=use_object
    )
    return _seeded(
        lambda: _printed(
            lambda: distiller.fit_psf(
                data,
                excitation,
                emission,
                grid,
                spad,
                **{
                    "zer_max": 4,
                    "num_starts": 2,
                    "max_iter": 3,
                    "independent_wavefront": independent,
                    "bead_radius": 100.0 if use_object else 0.0,
                    **options,
                },
            )
        )
    )


def _fit_detector(function):
    sim = _module(function).sim
    excitation = sim.simSettings(mask_sampl=32, wl=640.0)
    emission = excitation.copy()
    emission.wl = 660.0
    grid = sim.GridParameters(Nx=41, Nz=1, N=3, pxsizex=60, pxdim=50e3)
    return _printed(
        lambda: function(shifted_spot_dataset(), grid, excitation, emission)
    )


CASES: list[Case] = [
    Case(f"{DATAIO}:is_mcs_file", lambda f: _mcs_signal(f), "current schema"),
    Case(f"{DATAIO}:is_mcs_file", lambda f: _mcs_legacy(f), "legacy layout"),
    Case(f"{DATAIO}:is_mcs_file", _plain_h5, "plain HDF5 file"),
    Case(f"{DATAIO}:list_mcs_datasets", lambda f: _mcs_signal(f)),
    Case(f"{DATAIO}:load_mcs", lambda f: _mcs_signal(f), "numpy"),
    Case(
        f"{DATAIO}:load_mcs",
        lambda f: _mcs_signal(f, key="aux"),
        "aux dataset",
    ),
    Case(f"{DATAIO}:load_mcs", _loaded_mcs, "HDF5 dataset"),
    Case(f"{DATAIO}:read_mcs_signal", lambda f: _mcs_signal(f), "default"),
    Case(
        f"{DATAIO}:read_mcs_signal",
        lambda f: _mcs_signal(f, dataset="/raw/aux", channel=1, time=0, depth=0),
        "aux channel",
    ),
    Case(
        f"{DATAIO}:read_mcs_signal",
        lambda f: _mcs_signal(f, channel="sum"),
        "channel sum",
    ),
    Case(f"{MAG}:scalar_psf", lambda f: f(*_scalar_psf_inputs())),
    Case(f"{MAG}:scalar_psf_det", lambda f: _detector_psf(_module(f))),
    Case(f"{MAG}:rect", lambda f: f(np.linspace(-5.0, 5.0, 11), 4.0)),
    Case(f"{MAG}:shift_value", lambda f: f(450.0, 640.0, 660.0, 75e3, 50e3, 1.4)),
    Case(f"{MAG}:loss_shift", lambda f: f(450.0, 60.0, 640.0, 660.0, 75e3, 50e3, 1.4)),
    Case(
        f"{MAG}:loss_minimizer",
        lambda f: _minimizer_inputs(_module(f), maxiter=200),
        "converged",
    ),
    Case(
        f"{MAG}:loss_minimizer",
        lambda f: _minimizer_inputs(_module(f), maxiter=2),
        "iteration limit",
    ),
    Case(f"{MAG}:find_mag", lambda f: f(60.0, 640.0, 660.0, 75e3, 50e3, 1.4)),
    Case(f"{SHIFTS}:shift_matrix", lambda f: f(), "rect"),
    Case(f"{SHIFTS}:shift_matrix", lambda f: f("hex"), "hex"),
    Case(f"{SHIFTS}:rotation_matrix", lambda f: f(0.7)),
    Case(f"{SHIFTS}:mirror_matrix", lambda f: f(-1)),
    Case(f"{SHIFTS}:crop_shift", lambda f: _crop_shift(_module(f), "rect"), "rect"),
    Case(f"{SHIFTS}:crop_shift", lambda f: _crop_shift(_module(f), "hex"), "hex"),
    Case(
        f"{SHIFTS}:transform_shift_vectors",
        lambda f: f([2.5, 0.4, -1], _module(f).shift_matrix()),
    ),
    Case(
        f"{SHIFTS}:loss_shifts",
        lambda f: f(
            [2.0, 0.3],
            _module(f).shift_matrix() * 2.5,
            _module(f).shift_matrix(),
            -1,
        ),
    ),
    Case(
        f"{SHIFTS}:loss_minimizer",
        lambda f: _shift_loss_minimizer(_module(f), 1, maxiter=500),
        "converged",
    ),
    Case(
        f"{SHIFTS}:loss_minimizer",
        lambda f: _shift_loss_minimizer(_module(f), -1, maxiter=500),
        "mirrored",
    ),
    Case(
        f"{SHIFTS}:loss_minimizer",
        lambda f: _shift_loss_minimizer(_module(f), 1, maxiter=2),
        "iteration limit",
    ),
    Case(
        f"{SHIFTS}:find_parameters",
        lambda f: _find_parameters(_module(f), "rect", [2.5, 0.4, -1]),
        "rect, mirrored",
    ),
    Case(
        f"{SHIFTS}:find_parameters",
        lambda f: _find_parameters(_module(f), "rect", [3.0, 1.2, 1]),
        "rect",
    ),
    Case(
        f"{SHIFTS}:find_parameters",
        lambda f: _find_parameters(_module(f), "hex", [2.0, 0.6, 1]),
        "hex",
    ),
    Case(
        f"{SHIFTS}:calc_shift_vectors",
        lambda f: f([2.5, 0.4, -1]),
        "rect",
    ),
    Case(
        f"{SHIFTS}:calc_shift_vectors",
        lambda f: f([2.5, 0.4, 1], "hex"),
        "hex",
    ),
    Case(
        f"{MISALIGNMENT}:gaussian_2d",
        lambda f: f([1.0, 4.0, 3.0, 2.0, 0.1], *_gaussian_inputs()),
    ),
    Case(
        f"{MISALIGNMENT}:residuals",
        lambda f: f(
            [1.0, 4.0, 3.0, 2.0, 0.1],
            *_gaussian_inputs(),
            np.random.default_rng(0).uniform(0, 1, (7, 9)),
        ),
    ),
    Case(
        f"{MISALIGNMENT}:gaussian_fit",
        lambda f: f(_fingerprint(shifted_spot_dataset())),
    ),
    Case(
        f"{MISALIGNMENT}:find_misalignment",
        lambda f: _find_misalignment(f, shifted_spot_dataset()),
        "not mirrored",
    ),
    Case(
        f"{MISALIGNMENT}:find_misalignment",
        lambda f: _find_misalignment(f, _mirrored_dataset()),
        "mirrored",
    ),
    Case(
        f"{MISALIGNMENT}:realign_psf",
        lambda f: f(np.random.default_rng(0).uniform(0, 1, (3, 9, 11, 4))),
    ),
    Case(f"{ESTIMATOR}:GridFinder", _grid_finder),
    Case(
        f"{ESTIMATOR}:psf_width",
        lambda f: f(40, 100, 3, _settings(_sim(f)), 3e3, "positive"),
        "positive",
    ),
    Case(
        f"{ESTIMATOR}:psf_width",
        lambda f: f(40, 100, 3, _settings(_sim(f)), 3e3, "negative"),
        "negative",
    ),
    Case(
        f"{ESTIMATOR}:psf_width",
        lambda f: f(40, 100, 3, _settings(_sim(f)), 3e3, "symmetrical"),
        "symmetrical",
    ),
    Case(f"{ESTIMATOR}:find_upsampling", lambda f: f(60, 4), "target 4"),
    Case(f"{ESTIMATOR}:find_upsampling", lambda f: f(60.0), "default target"),
    Case(f"{ESTIMATOR}:find_upsampling", lambda f: f(25, 2), "target 2"),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _psf_from_data(f, z_out_of_focus=300.0),
        "downsampled",
    ),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _psf_from_data(f, z_out_of_focus=300.0, downsample=False),
        "not downsampled",
    ),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _psf_from_data(f, downsample=False),
        "out-of-focus depth found",
    ),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _psf_from_data(
            f, z_out_of_focus=300.0, downsample=False, stack="positive"
        ),
        "positive stack",
    ),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _psf_from_data(
            f, z_out_of_focus=300.0, downsample=False, n_photon_excitation=2
        ),
        "two-photon",
    ),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _psf_from_data(
            f, z_out_of_focus=300.0, downsample=False, check_alignment=True
        ),
        "alignment checked",
    ),
    Case(
        f"{ESTIMATOR}:psf_estimator_from_data",
        lambda f: _raised(
            lambda: _sted_settings(f, z_out_of_focus=300.0, downsample=False)
        ),
        "STED fails",
    ),
    Case(
        f"{ESTIMATOR}:find_out_of_focus",
        lambda f: _defocus_stack(_module(f), "KL"),
        "KL",
    ),
    Case(
        f"{ESTIMATOR}:find_out_of_focus",
        lambda f: _defocus_stack(_module(f), "Pearson"),
        "Pearson",
    ),
    Case(
        f"{ESTIMATOR}:find_out_of_focus_from_param",
        lambda f: f(
            100,
            _module(f).sim.simSettings(mask_sampl=32),
            _module(f).sim.simSettings(mask_sampl=32, wl=600.0),
            mode="KL",
            stack="positive",
        ),
        "KL from pixel size",
    ),
    Case(
        f"{ESTIMATOR}:find_out_of_focus_from_param",
        lambda f: _from_param(_module(f), pxsizex=100),
        "Pearson from pixel size",
    ),
    Case(
        f"{ESTIMATOR}:find_out_of_focus_from_param",
        lambda f: _from_param_grid(_module(f), mode="KL", stack="positive"),
        "grid, KL",
    ),
    Case(
        f"{ESTIMATOR}:find_out_of_focus_from_param",
        lambda f: _raised(lambda: _from_param_sted(_module(f))),
        "STED fails",
    ),
    Case(
        f"{ESTIMATOR}:find_max_discrepancy",
        lambda f: _max_discrepancy(_module(f), "KL", False),
        "KL",
    ),
    Case(
        f"{ESTIMATOR}:find_max_discrepancy",
        lambda f: _max_discrepancy(_module(f), "Pearson", True),
        "Pearson, graph",
    ),
    Case(
        f"{ESTIMATOR}:conditioning",
        lambda f: _conditioning(_module(f), "Pearson", "positive"),
        "Pearson",
    ),
    Case(
        f"{ESTIMATOR}:conditioning",
        lambda f: _conditioning(_module(f), "KL", "symmetrical"),
        "KL symmetrical",
    ),
    Case(
        f"{ESTIMATOR}:conditioning",
        lambda f: _raised(lambda: _conditioning_sted(_module(f))),
        "STED fails",
    ),
    Case(
        f"{ESTIMATOR}:conditioning",
        lambda f: _conditioning_input_psf(_module(f), "Pearson"),
        "given stack, Pearson",
    ),
    Case(
        f"{ESTIMATOR}:conditioning",
        lambda f: _conditioning_input_psf(_module(f), "KL"),
        "given stack, KL",
    ),
    Case(f"{ESTIMATOR}:combine_psf_irf", _combine_psf_irf),
    Case(f"{S2ISM}:partial_convolution_rfft", lambda f: _convolution(f), "spatial"),
    Case(
        f"{S2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, (True, False)),
        "Fourier kernel",
    ),
    Case(
        f"{S2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, (False, True), padding=[8, 8]),
        "Fourier volume",
    ),
    Case(
        f"{S2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, (True, True)),
        "both in Fourier space",
    ),
    Case(
        f"{S2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, padding=[12, 12]),
        "padded",
    ),
    Case(f"{S2ISM}:amd_update_fft", _amd_update),
    Case(
        f"{S2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 1, True, pre_flag=True),
        "auto, first time below threshold",
    ),
    Case(
        f"{S2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 1, True, pre_flag=False),
        "auto, second time below threshold",
    ),
    Case(
        f"{S2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 5, False),
        "auto, iteration limit",
    ),
    Case(
        f"{S2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 2, False),
        "auto, still running",
    ),
    Case(
        f"{S2ISM}:amd_stop",
        lambda f: _amd_stop(f, "fixed", 5, False),
        "fixed, iteration limit",
    ),
    Case(
        f"{S2ISM}:amd_stop",
        lambda f: _amd_stop(f, "fixed", 2, False),
        "fixed, still running",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction(f),
        "even image",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction_padded(f),
        "odd image, padded PSF",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction(f, rep_to_save="all"),
        "all iterations",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction_padded(f, rep_to_save=[0, 2]),
        "chosen iterations",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction_padded(f, stop="auto", max_iter=5, threshold=1e-1),
        "auto stop",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction_padded(f, stop="auto", threshold=0.0),
        "auto stop at the iteration limit",
    ),
    Case(
        f"{S2ISM}:max_likelihood_reconstruction",
        lambda f: _reconstruction_padded(f, initialization="sum"),
        "sum initialization",
    ),
    Case(f"{S2ISM}:batch_reconstruction", _batch),
    Case(
        f"{S2ISM}:data_driven_reconstruction",
        lambda f: _data_driven(f),
        "depth given",
    ),
    Case(
        f"{S2ISM}:data_driven_reconstruction",
        lambda f: _data_driven(f, rep_to_save="all", initialization="sum"),
        "all iterations, sum initialization",
    ),
    Case(
        f"{S2ISM}:data_driven_reconstruction",
        lambda f: _data_driven(f, z_out_of_focus="ToFind", downsample=False),
        "depth found, not downsampled",
    ),
    Case(
        f"{FLUS2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f),
        "spatial",
    ),
    Case(
        f"{FLUS2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, (True, False)),
        "Fourier kernel",
    ),
    Case(
        f"{FLUS2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, (False, True), padding=[8, 8]),
        "Fourier volume",
    ),
    Case(
        f"{FLUS2ISM}:partial_convolution_rfft",
        lambda f: _convolution(f, padding=[12, 12], dim3="czxy"),
        "padded, chosen output order",
    ),
    Case(f"{FLUS2ISM}:flus2ism_update", _lifetime_update),
    Case(f"{FLUS2ISM}:flus2ism_pulseinterl__update", _pulse_update),
    Case(
        f"{FLUS2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 1, True, pre_flag=True),
        "auto, first time below threshold",
    ),
    Case(
        f"{FLUS2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 1, True, pre_flag=False),
        "auto, second time below threshold",
    ),
    Case(
        f"{FLUS2ISM}:amd_stop",
        lambda f: _amd_stop(f, "auto", 5, False),
        "auto, iteration limit",
    ),
    Case(
        f"{FLUS2ISM}:amd_stop",
        lambda f: _amd_stop(f, "fixed", 5, False),
        "fixed, iteration limit",
    ),
    Case(
        f"{FLUS2ISM}:amd_stop",
        lambda f: _amd_stop(f, "fixed", 2, False),
        "fixed, still running",
    ),
    Case(f"{FLUS2ISM}:flus2ism", lambda f: _lifetime(f), "odd image"),
    Case(
        f"{FLUS2ISM}:flus2ism",
        lambda f: _lifetime(f, crop=True),
        "even image, padded PSF",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism",
        lambda f: _lifetime(f, rep_to_save="all"),
        "all iterations",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism",
        lambda f: _lifetime(f, rep_to_save=[0, 2]),
        "chosen iterations",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism",
        lambda f: _lifetime(f, stop="auto", max_iter=5, threshold=1e-1),
        "auto stop",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism",
        lambda f: _raised(lambda: _lifetime(f, initialization="sum")),
        "sum initialization fails",
    ),
    Case(f"{FLUS2ISM}:flus2ism_pulseinterl", lambda f: _pulse(f), "default"),
    Case(
        f"{FLUS2ISM}:flus2ism_pulseinterl",
        lambda f: _pulse(f, rep_to_save="all"),
        "all iterations",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism_pulseinterl",
        lambda f: _pulse(f, rep_to_save=[0, 2]),
        "chosen iterations",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism_pulseinterl",
        lambda f: _pulse(f, stop="auto", max_iter=5, threshold=1e-1),
        "auto stop",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism_pulseinterl",
        lambda f: _pulse(f, even=True),
        "even image, padded PSF",
    ),
    Case(
        f"{FLUS2ISM}:flus2ism_pulseinterl",
        lambda f: _pulse(f, padded=True),
        "odd image, padded PSF",
    ),
    Case(
        f"{FLUS2ISM}:batch_reconstruction_flus2ism_pulse_inter",
        _pulse_batch,
        "fixed stop",
    ),
    Case(
        f"{FLUS2ISM}:batch_reconstruction_flus2ism_pulse_inter",
        lambda f: _pulse_batch(f, stop="auto", max_iter=5, threshold=1e-1),
        "auto stop",
    ),
    Case(
        f"{DISTILLER}:disk",
        lambda f: f(_distiller_inputs(_module(f))[2], 100.0),
        "Fourier",
    ),
    Case(
        f"{DISTILLER}:disk",
        lambda f: f(_distiller_inputs(_module(f), 3)[2], 120.0, fourier=False),
        "three-dimensional, spatial",
    ),
    Case(f"{DISTILLER}:fit_detector", _fit_detector),
    Case(f"{DISTILLER}:forward", lambda f: _forward(_module(f)), "two-dimensional"),
    Case(
        f"{DISTILLER}:forward",
        lambda f: _forward(_module(f), nz=3),
        "three-dimensional",
    ),
    Case(
        f"{DISTILLER}:forward",
        lambda f: _forward(_module(f), use_object=True),
        "with object",
    ),
    Case(
        f"{DISTILLER}:forward",
        lambda f: _forward(_module(f), independent=True),
        "independent wavefronts",
    ),
    Case(
        f"{DISTILLER}:vectorized_forward",
        lambda f: _vectorized_forward(_module(f), False),
        "shared wavefront",
    ),
    Case(
        f"{DISTILLER}:vectorized_forward",
        lambda f: _vectorized_forward(_module(f), True),
        "independent wavefronts",
    ),
    Case(f"{DISTILLER}:loss_function", lambda f: _loss(_module(f)), "mse"),
    Case(f"{DISTILLER}:loss_function", lambda f: _loss(_module(f), "mae"), "mae"),
    Case(f"{DISTILLER}:loss_function", lambda f: _loss(_module(f), "kl"), "kl"),
    Case(
        f"{DISTILLER}:loss_function",
        lambda f: _loss(_module(f), fit_mag=True),
        "magnification fitted",
    ),
    Case(
        f"{DISTILLER}:loss_function",
        lambda f: _loss(_module(f), vectorized=True),
        "vectorized",
    ),
    Case(
        f"{DISTILLER}:loss_function",
        lambda f: _loss(_module(f), independent=True),
        "independent wavefronts",
    ),
    Case(
        f"{DISTILLER}:structure_result",
        lambda f: f(torch.tensor([1.0, 2.0, 3.0]), 4),
        "shared wavefront",
    ),
    Case(
        f"{DISTILLER}:structure_result",
        lambda f: f(torch.arange(1.0, 7.0), 4, True),
        "independent wavefronts",
    ),
    Case(
        f"{DISTILLER}:generate_starting_points",
        lambda f: _seeded(lambda: f(3, 4, 2.0, "sobol")),
        "Sobol",
    ),
    Case(
        f"{DISTILLER}:generate_starting_points",
        lambda f: _seeded(lambda: f(3, 4, 2.0, "lhs")),
        "Latin hypercube",
    ),
    Case(f"{DISTILLER}:minimizer", lambda f: _minimizer(_module(f)), "quiet"),
    Case(
        f"{DISTILLER}:minimizer",
        lambda f: _minimizer(_module(f), verbose=True),
        "verbose",
    ),
    Case(
        f"{DISTILLER}:minimizer",
        lambda f: _minimizer(_module(f), independent=True),
        "independent wavefronts",
    ),
    Case(f"{DISTILLER}:fit_psf", lambda f: _fit_psf(_module(f)), "mae, Sobol"),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _fit_psf(_module(f), loss="mse"),
        "mse",
    ),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _fit_psf(_module(f), loss="mae", initialization="lhs"),
        "mae, Latin hypercube",
    ),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _raised(lambda: _fit_psf(_module(f), loss="kl")),
        "kl fails",
    ),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _fit_psf(_module(f), use_object=True),
        "bead",
    ),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _fit_psf(_module(f), independent=True),
        "independent wavefronts",
    ),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _fit_psf(_module(f), independent=True, equal_start=True),
        "independent wavefronts, equal start",
    ),
    Case(
        f"{DISTILLER}:fit_psf",
        lambda f: _fit_psf(_module(f), verbose=False),
        "quiet",
    ),
]

SKIPPED: dict[str, str] = {}
