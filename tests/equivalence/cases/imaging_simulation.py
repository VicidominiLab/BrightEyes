"""Equivalence cases for `brighteyes_ism.simulation`."""

import contextlib
import io
from importlib import import_module

import numpy as np
import torch
from _registry import Case

PSF = "brighteyes_ism.simulation.PSF_sim"
TUBULIN = "brighteyes_ism.simulation.Tubulin_sim"
DETECTOR = "brighteyes_ism.simulation.detector"
SIMULATOR = "brighteyes_ism.simulation.simulator"
UTILS = "brighteyes_ism.simulation.utils"

NX = 16
CPU = torch.device("cpu")


def _psf_sim(module):
    return import_module(f"{module.__package__}.PSF_sim")


def _printed(function):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = function()
    return result, output.getvalue()


def _grid(module, **changes):
    grid = _psf_sim(module).GridParameters(Nx=NX, Nz=3, N=3)
    for name, value in changes.items():
        setattr(grid, name, value)
    return grid


def _settings(module, **changes):
    return _psf_sim(module).simSettings(mask_sampl=32, **changes)


def _grid_case(module):
    grid = _grid(module)
    hexagonal = _grid(module, geometry="hex", N=5)
    return (
        grid.rangex,
        grid.rangez,
        grid.Nch,
        hexagonal.Nch,
        module.GridParameters(N=(3, 4)).Nch,
        grid.spad_size(),
        grid.spad_size("real"),
        grid.spad_size(simPar=_settings(module)),
        grid.copy(),
        _printed(grid.Print),
        grid,
    )


def _settings_case(module, **changes):
    settings = _settings(module, **changes)
    return (
        settings.f,
        settings.alpha,
        settings.airy_unit,
        settings.depth_of_field,
        settings.aberration,
        settings.wavefront,
        settings.copy(),
        _printed(settings.Print),
        settings,
    )


def _single_psf(module, **changes):
    return module.singlePSF(_settings(module, **changes), 40, NX, [-50, 50], 3)


def _gaussian_psf(module):
    settings = _settings(module, field="Gaussian")
    settings.wo = 0.5
    return module.singlePSF(settings, 40, NX, [-50, 50], 3)


def _psf_3d(module, settings=None, **options):
    grid = options.pop("grid", None) or _grid(module)
    settings = settings or _settings(module)
    return module.SPAD_PSF_3D(grid, settings, settings, process="cpu", **options)


def _psf_2d(module, **options):
    settings = _settings(module)
    return module.SPAD_PSF_2D(
        _grid(module), settings, settings, process="cpu", **options
    )


def _precomputed_spad(module):
    return module.custom_detector(_grid(module), CPU)


def _tubulin_settings(module, **changes):
    tub = module.tubSettings()
    tub.xy_dimension = 32
    tub.n_filament = 3
    tub.intensity_filament = [1, 2]
    for name, value in changes.items():
        setattr(tub, name, value)
    return tub


def _tubulin(module, **changes):
    return _printed(lambda: module.functionPhTub(_tubulin_settings(module, **changes)))


def _pinholes(module, shape):
    coords = np.array([[-2, 0, 3], [1, 0, -3]])
    return module.pinhole_array(coords, NX, 450, 40, 50e3, shape, CPU)


def _transformed(module, **changes):
    grid = _grid(module, **changes)
    detector = module.custom_detector(_grid(module), CPU)
    return module.transform_detector(grid, detector)


def _custom_detector(module, **changes):
    return module.custom_detector(_grid(module, **changes), CPU)


def _phantom_2d() -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 1, (8, 8))


def _psf_2d_stack() -> np.ndarray:
    return np.random.default_rng(1).uniform(0, 1, (3, 3, 4))


def _simulator(cls, phantom, psf, **options):
    simulator = cls(phantom, psf, **options)
    simulator.forward()
    return (
        simulator.ground_truth,
        simulator.clean_image,
        simulator.image,
        simulator.copy(),
        simulator,
    )


def _partial_convolution(module, **options):
    rng = np.random.default_rng(0)
    volume = torch.from_numpy(rng.uniform(0, 1, (4, 8, 8)))
    kernel = torch.from_numpy(rng.uniform(0, 1, (8, 8, 3)))
    return module.partial_convolution(volume, kernel, "zxy", "xyc", "xy", **options)


def _fourier_inputs():
    rng = np.random.default_rng(0)
    volume = torch.fft.fftn(torch.from_numpy(rng.uniform(0, 1, (4, 8, 8))), dim=[1, 2])
    kernel = torch.fft.fftn(torch.from_numpy(rng.uniform(0, 1, (8, 8, 3))), dim=[0, 1])
    return volume, kernel


def _partial_convolution_fourier(module, fourier):
    volume, kernel = _fourier_inputs()
    spatial_volume = torch.fft.ifftn(volume, dim=[1, 2])
    spatial_kernel = torch.fft.ifftn(kernel, dim=[0, 1])
    return module.partial_convolution(
        volume if fourier[0] else spatial_volume,
        kernel if fourier[1] else spatial_kernel,
        "zxy",
        "xyc",
        "xy",
        fourier=fourier,
    )


def _module(obj):
    return import_module(obj.__module__)


CASES: list[Case] = [
    Case(f"{PSF}:GridParameters", lambda f: _grid_case(_module(f))),
    Case(f"{PSF}:simSettings", lambda f: _settings_case(_module(f)), "default"),
    Case(
        f"{PSF}:simSettings",
        lambda f: _settings_case(
            _module(f), mask="vortex", abe_index=[4, 6], abe_ampli=[0.5, 0.2]
        ),
        "vortex mask and aberrations",
    ),
    Case(f"{PSF}:singlePSF", lambda f: _single_psf(_module(f)), "default"),
    Case(
        f"{PSF}:singlePSF",
        lambda f: _single_psf(
            _module(f), mask="vortex", abe_index=[4], abe_ampli=[0.5]
        ),
        "vortex mask and aberration",
    ),
    Case(f"{PSF}:singlePSF", lambda f: _gaussian_psf(_module(f)), "Gaussian field"),
    Case(f"{PSF}:SPAD_PSF_3D", lambda f: _psf_3d(_module(f)), "symmetrical"),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), stack="positive"),
        "positive",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), stack="negative"),
        "negative",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), stack=[-50.0, 0.0, 50.0], normalize=False),
        "explicit planes",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), normalize=False),
        "not normalized",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), n_photon_excitation=2),
        "two-photon",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), output="tensor"),
        "tensor output",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _psf_3d(_module(f), spad=_precomputed_spad(_module(f))),
        "precomputed detector",
    ),
    Case(f"{PSF}:SPAD_PSF_2D", lambda f: _psf_2d(_module(f)), "default"),
    Case(f"{PSF}:SPAD_PSF_2D", lambda f: _psf_2d(_module(f), z_shift=50), "z shift"),
    Case(
        f"{PSF}:SPAD_PSF_2D",
        lambda f: _psf_2d(_module(f), normalize=False),
        "not normalized",
    ),
    Case(
        f"{PSF}:SPAD_PSF_2D",
        lambda f: _psf_2d(_module(f), n_photon_excitation=2),
        "two-photon",
    ),
    Case(
        f"{PSF}:SPAD_PSF_2D",
        lambda f: _psf_2d(_module(f), output="tensor"),
        "tensor output",
    ),
    Case(f"{TUBULIN}:tubSettings", lambda f: _tubulin_settings(_module(f))),
    Case(f"{TUBULIN}:thetaVariation", lambda f: f(1), "two-dimensional"),
    Case(f"{TUBULIN}:thetaVariation", lambda f: f(3), "three-dimensional"),
    Case(
        f"{TUBULIN}:getElipsoid",
        lambda f: f(
            8,
            8,
            1,
            4.0,
            4.0,
            2.0,
            *np.meshgrid(np.arange(16), np.arange(16), np.arange(1, 3)),
        ),
    ),
    Case(f"{TUBULIN}:functionPhTub", lambda f: _tubulin(_module(f)), "two-dimensional"),
    Case(
        f"{TUBULIN}:functionPhTub",
        lambda f: _tubulin(_module(f), xz_dimension=3, z_pixel=2),
        "three-dimensional",
    ),
    Case(f"{DETECTOR}:circle", lambda f: f(16, 3.5), "even size"),
    Case(f"{DETECTOR}:circle", lambda f: f(15, 3.5), "odd size"),
    Case(f"{DETECTOR}:square", lambda f: f(16, 5), "even size"),
    Case(f"{DETECTOR}:square", lambda f: f(15, 5), "odd size"),
    Case(f"{DETECTOR}:hexagon", lambda f: f(16, 4)),
    Case(f"{DETECTOR}:rect_grid", lambda f: f(3, np.arange(-1, 2))),
    Case(f"{DETECTOR}:hex_grid", lambda f: f(5, np.arange(-2, 3))),
    Case(f"{DETECTOR}:det_coords", lambda f: f(3, "rect"), "rect"),
    Case(f"{DETECTOR}:det_coords", lambda f: f(5, "hex"), "hex"),
    Case(
        f"{DETECTOR}:pinhole_array", lambda f: _pinholes(_module(f), "square"), "square"
    ),
    Case(
        f"{DETECTOR}:pinhole_array", lambda f: _pinholes(_module(f), "circle"), "circle"
    ),
    Case(
        f"{DETECTOR}:pinhole_array",
        lambda f: _pinholes(_module(f), "hexagon"),
        "hexagon",
    ),
    Case(
        f"{DETECTOR}:transform_detector",
        lambda f: _transformed(_module(f)),
        "unchanged",
    ),
    Case(
        f"{DETECTOR}:transform_detector",
        lambda f: _transformed(_module(f), mirroring=-1),
        "mirrored",
    ),
    Case(
        f"{DETECTOR}:transform_detector",
        lambda f: _transformed(_module(f), rotation=0.5),
        "rotated",
    ),
    Case(f"{DETECTOR}:custom_detector", lambda f: _custom_detector(_module(f)), "rect"),
    Case(
        f"{DETECTOR}:custom_detector",
        lambda f: _custom_detector(_module(f), geometry="hex", N=5, Nx=32),
        "hex",
    ),
    Case(
        f"{DETECTOR}:custom_detector",
        lambda f: _custom_detector(
            _module(f), pinhole_shape="circle", mirroring=-1, rotation=0.3
        ),
        "circle, mirrored and rotated",
    ),
    Case(
        f"{DETECTOR}:custom_detector",
        lambda f: _custom_detector(_module(f), name="airyscan", Nx=32),
        "airyscan",
    ),
    Case(f"{DETECTOR}:hex_to_airy", lambda f: f(np.arange(90).reshape(2, 45))),
    Case(f"{DETECTOR}:airy_to_hex", lambda f: f(np.arange(64.0).reshape(2, 32))),
    Case(
        f"{SIMULATOR}:ImageSimulator",
        lambda f: _simulator(f, _phantom_2d(), _psf_2d_stack(), signal=50),
        "channels",
    ),
    Case(
        f"{SIMULATOR}:ImageSimulator",
        lambda f: _simulator(f, _phantom_2d(), _psf_2d_stack()[..., 0], signal=50),
        "single channel",
    ),
    Case(
        f"{SIMULATOR}:ImageSimulator",
        lambda f: _simulator(
            f,
            np.stack([_phantom_2d()] * 2),
            np.stack([_psf_2d_stack()] * 2)[..., 0],
            signal=np.array([20, 40]),
        ),
        "per-plane signal",
    ),
    Case(
        f"{SIMULATOR}:ImageSimulator",
        lambda f: _simulator(
            f,
            np.stack([_phantom_2d()] * 2),
            np.stack([_psf_2d_stack()] * 2),
            signal=50,
            z_projection=True,
        ),
        "z projection, channels",
    ),
    Case(
        f"{SIMULATOR}:ImageSimulator",
        lambda f: _simulator(
            f,
            np.stack([_phantom_2d()] * 2),
            np.stack([_psf_2d_stack()[..., 0]] * 2),
            z_projection=True,
        ),
        "z projection, single channel",
    ),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution(_module(f)),
        "spatial inputs",
    ),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution_fourier(_module(f), (True, False)),
        "Fourier volume",
    ),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution_fourier(_module(f), (False, True)),
        "Fourier kernel",
    ),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution_fourier(_module(f), (True, True)),
        "both in Fourier space",
    ),
]

SKIPPED: dict[str, str] = {}
