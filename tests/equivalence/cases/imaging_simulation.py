import contextlib
import importlib
import io

import numpy as np
import torch

from .._registry import Case

PSF = "brighteyes_ism.simulation.PSF_sim"
TUB = "brighteyes_ism.simulation.Tubulin_sim"
DET = "brighteyes_ism.simulation.detector"
SIM = "brighteyes_ism.simulation.simulator"
UTILS = "brighteyes_ism.simulation.utils"


def _module(obj):
    return importlib.import_module(obj.__module__)


def _sibling(obj, name):
    package = obj.__module__.rpartition(".")[0]
    return importlib.import_module(f"{package}.{name}")


def _capture(function, *args, **kwargs):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def _raises(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        return error


def _grid(module, **changes):
    grid = module.GridParameters(Nx=16, Nz=3, N=3)
    for name, value in changes.items():
        setattr(grid, name, value)
    return grid


def _settings(module, **changes):
    return module.simSettings(mask_sampl=16, **changes)


def _psf_inputs(function):
    module = _module(function)
    return _grid(module), _settings(module), _settings(module, wl=520.0)


def _small_pinholes(function):
    grid = _grid(_module(function))
    return _sibling(function, "detector").custom_detector(grid, torch.device("cpu"))


def _grid_parameters(cls):
    rect = cls(Nx=16, Nz=3, N=3)
    hexagonal = cls(Nx=16, Nz=3, N=5)
    hexagonal.geometry = "hex"
    tuple_n = cls(Nx=16, N=(2, 3))
    simulation = importlib.import_module(cls.__module__).simSettings()
    return (
        rect,
        rect.Nch,
        rect.rangex,
        rect.rangez,
        rect.spad_size(),
        rect.spad_size("real"),
        rect.spad_size(simPar=simulation),
        hexagonal.Nch,
        tuple_n.Nch,
        rect.copy(),
        _capture(rect.Print),
    )


def _sim_settings(cls):
    plain = cls()
    aberrated = cls(abe_index=[4, 5], abe_ampli=[0.5, 0.25], mask_sampl=16)
    scalar = cls(abe_index=4, abe_ampli=0.5, mask_sampl=16)
    vortex = cls(mask="vortex", mask_sampl=16)
    return (
        plain,
        plain.f,
        plain.alpha,
        plain.airy_unit,
        plain.depth_of_field,
        plain.aberration,
        aberrated.aberration,
        scalar.aberration,
        aberrated.wavefront,
        vortex.wavefront,
        cls(mask_sampl=16).wavefront,
        plain.copy(),
        _capture(plain.Print),
    )


def _single_psf(function, **changes):
    settings = _settings(_module(function), **changes)
    return function(settings, 40, 16, [0.0, 100.0], 2)


def _spad_3d(function, **kwargs):
    grid, excitation, emission = _psf_inputs(function)
    return function(grid, excitation, emission, process="cpu", **kwargs)


def _spad_2d(function, **kwargs):
    grid, excitation, emission = _psf_inputs(function)
    return function(grid, excitation, emission, process="cpu", **kwargs)


def _tub_settings(cls):
    settings = cls()
    return settings, vars(settings)


def _phantom(function, **changes):
    settings = _module(function).tubSettings()
    settings.xy_dimension = 32
    settings.n_filament = 3
    for name, value in changes.items():
        setattr(settings, name, value)
    return function(settings)


def _ellipsoid(function):
    cols, rows, depth = np.meshgrid(np.arange(8), np.arange(8), np.arange(1, 4))
    return function(3.0, 4.0, 2.0, 2.5, 3.0, 1.5, cols, rows, depth)


def _pinhole_array(function, shape):
    s = np.array([[-2, 0, 2], [1, 0, -1]])
    return function(s, 16, 450, 40, 50e3, shape, torch.device("cpu"))


def _transform(function, **changes):
    psf = _sibling(function, "PSF_sim")
    pinholes = function.__globals__["custom_detector"](_grid(psf), torch.device("cpu"))
    return function(_grid(psf, **changes), pinholes)


def _custom_detector(function, **changes):
    grid = _grid(_sibling(function, "PSF_sim"), **changes)
    return function(grid, torch.device("cpu"))


def _airy_detector(function):
    grid = _sibling(function, "PSF_sim").GridParameters(Nx=64, Nz=1, N=3)
    grid.name = "airyscan"
    return function(grid, torch.device("cpu"))


def _simulator(cls, phantom_shape, psf_shape, signal=1, **kwargs):
    rng = np.random.default_rng(0)
    simulator = cls(
        phantom=rng.random(phantom_shape),
        psf=rng.random(psf_shape),
        signal=signal * 100,
        **kwargs,
    )
    simulator.forward()
    return simulator, simulator.ground_truth, simulator.copy()


def _simulator_steps(cls):
    simulator = cls(
        phantom=np.random.default_rng(0).random((8, 8)),
        psf=np.random.default_rng(1).random((3, 3, 4)),
        signal=50,
    )
    simulator.blur()
    clean = simulator.clean_image.copy()
    simulator.poisson_noise()
    return simulator, clean


def _partial_convolution(function, fourier=(False, False)):
    rng = np.random.default_rng(0)
    volume = torch.from_numpy(rng.random((3, 8, 8)))
    kernel = torch.from_numpy(rng.random((8, 8, 4)))
    if fourier[0]:
        volume = torch.fft.fftn(volume, dim=(1, 2))
    if fourier[1]:
        kernel = torch.fft.fftn(kernel, dim=(0, 1))
    return function(volume, kernel, "zxy", "xyc", "xy", fourier)


def _given_detector_3d(function):
    grid, excitation, emission = _psf_inputs(function)
    return function(
        grid, excitation, emission, spad=_small_pinholes(function), process="cpu"
    )


def _given_detector_2d(function):
    grid, excitation, emission = _psf_inputs(function)
    return function(
        grid, excitation, emission, spad=_small_pinholes(function), process="cpu"
    )


CASES: list[Case] = [
    Case(f"{PSF}:GridParameters", _grid_parameters),
    Case(f"{PSF}:simSettings", _sim_settings),
    Case(f"{PSF}:singlePSF", _single_psf, "plane wave"),
    Case(f"{PSF}:singlePSF", lambda f: _single_psf(f, mask="vortex"), "vortex mask"),
    Case(
        f"{PSF}:singlePSF",
        lambda f: _single_psf(f, abe_index=[4], abe_ampli=[0.5]),
        "aberration",
    ),
    Case(f"{PSF}:SPAD_PSF_3D", _spad_3d, "symmetrical"),
    Case(f"{PSF}:SPAD_PSF_3D", lambda f: _spad_3d(f, stack="positive"), "positive"),
    Case(f"{PSF}:SPAD_PSF_3D", lambda f: _spad_3d(f, stack="negative"), "negative"),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _spad_3d(f, stack=[-50.0, 0.0, 50.0], normalize=False),
        "explicit planes",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _spad_3d(f, n_photon_excitation=2),
        "two-photon",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _spad_3d(f, normalize=False),
        "not normalized",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _spad_3d(f, output="tensor"),
        "tensor output",
    ),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _raises(_spad_3d, f, output="other"),
        "unknown output",
    ),
    Case(f"{PSF}:SPAD_PSF_3D", _given_detector_3d, "given detector"),
    Case(
        f"{PSF}:SPAD_PSF_3D",
        lambda f: _raises(_spad_3d, f, stedPar=_settings(_module(f))),
        "STED beam",
    ),
    Case(f"{PSF}:SPAD_PSF_2D", _spad_2d, "focal plane"),
    Case(f"{PSF}:SPAD_PSF_2D", lambda f: _spad_2d(f, z_shift=50.0), "shifted plane"),
    Case(
        f"{PSF}:SPAD_PSF_2D",
        lambda f: _spad_2d(f, n_photon_excitation=2),
        "two-photon",
    ),
    Case(f"{PSF}:SPAD_PSF_2D", lambda f: _spad_2d(f, normalize=False), "not normalized"),
    Case(f"{PSF}:SPAD_PSF_2D", lambda f: _spad_2d(f, output="tensor"), "tensor output"),
    Case(f"{PSF}:SPAD_PSF_2D", _given_detector_2d, "given detector"),
    Case(f"{TUB}:tubSettings", _tub_settings),
    Case(f"{TUB}:thetaVariation", lambda f: (f(1), f(3)), "flat and stacked"),
    Case(f"{TUB}:getElipsoid", _ellipsoid),
    Case(f"{TUB}:functionPhTub", _phantom, "2D"),
    Case(
        f"{TUB}:functionPhTub",
        lambda f: _phantom(f, xz_dimension=3, intensity_filament=[1, 3]),
        "3D",
    ),
    Case(
        f"{DET}:circle",
        lambda f: (f(15, 4.5), f(16, 3.0), f(15, 0.5)),
        "odd, even and tiny",
    ),
    Case(f"{DET}:square", lambda f: (f(15, 5.0), f(16, 6.5)), "odd and even"),
    Case(f"{DET}:hexagon", lambda f: (f(15, 4.0), f(16, 5.0)), "odd and even"),
    Case(f"{DET}:rect_grid", lambda f: f(3, np.arange(-1, 2))),
    Case(f"{DET}:hex_grid", lambda f: (f(5, np.arange(-2, 3)), f(3, np.arange(-1, 2)))),
    Case(
        f"{DET}:det_coords",
        lambda f: (f(3, "rect"), f(5, "hex"), f(3, "hex")),
        "rect and hex",
    ),
    Case(f"{DET}:det_coords", lambda f: _raises(f, 3, "round"), "invalid geometry"),
    Case(f"{DET}:pinhole_array", lambda f: _pinhole_array(f, "square"), "square"),
    Case(f"{DET}:pinhole_array", lambda f: _pinhole_array(f, "circle"), "circle"),
    Case(f"{DET}:pinhole_array", lambda f: _pinhole_array(f, "hexagon"), "hexagon"),
    Case(
        f"{DET}:pinhole_array",
        lambda f: _raises(_pinhole_array, f, "star"),
        "invalid shape",
    ),
    Case(f"{DET}:transform_detector", _transform, "identity"),
    Case(
        f"{DET}:transform_detector",
        lambda f: _transform(f, mirroring=-1),
        "mirrored",
    ),
    Case(
        f"{DET}:transform_detector",
        lambda f: _transform(f, rotation=0.4),
        "rotated",
    ),
    Case(
        f"{DET}:transform_detector",
        lambda f: _transform(f, mirroring=-1, rotation=0.4, N=(3, 3)),
        "mirrored and rotated, tuple N",
    ),
    Case(f"{DET}:custom_detector", _custom_detector, "rect"),
    Case(
        f"{DET}:custom_detector",
        lambda f: _custom_detector(f, geometry="hex", N=5, Nx=32),
        "hex",
    ),
    Case(
        f"{DET}:custom_detector",
        lambda f: _custom_detector(f, pinhole_shape="circle"),
        "circle pinholes",
    ),
    Case(f"{DET}:custom_detector", _airy_detector, "airyscan"),
    Case(f"{DET}:hex_to_airy", lambda f: f(np.arange(90).reshape(2, 45))),
    Case(f"{DET}:airy_to_hex", lambda f: f(np.arange(64, dtype=float).reshape(2, 32))),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _simulator(c, (8, 8), (3, 3, 4)),
        "2D phantom, channel PSF",
    ),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _simulator(c, (8, 8), (3, 3)),
        "2D phantom, single PSF",
    ),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _simulator(c, (3, 8, 8), (3, 3, 3, 4), z_projection=True),
        "3D phantom projected, channel PSF",
    ),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _simulator(c, (3, 8, 8), (3, 3, 3, 4)),
        "3D phantom, channel PSF",
    ),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _simulator(c, (3, 8, 8), (3, 3, 3), z_projection=True),
        "3D phantom projected, single PSF",
    ),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _simulator(c, (3, 8, 8), (3, 3, 3, 4), signal=np.array([1, 2, 3])),
        "per-plane signal",
    ),
    Case(f"{SIM}:ImageSimulator", _simulator_steps, "blur then noise"),
    Case(
        f"{SIM}:ImageSimulator",
        lambda c: _raises(_simulator, c, (8, 8), (3,)),
        "PSF with fewer dimensions",
    ),
    Case(f"{UTILS}:partial_convolution", _partial_convolution, "spatial inputs"),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution(f, (True, False)),
        "volume in Fourier space",
    ),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution(f, (False, True)),
        "kernel in Fourier space",
    ),
    Case(
        f"{UTILS}:partial_convolution",
        lambda f: _partial_convolution(f, (True, True)),
        "both in Fourier space",
    ),
]

SKIPPED: dict[str, str] = {}
