"""Equivalence cases for `brighteyes_ism.analysis`."""

import contextlib
import io

import numpy as np
from _inputs import CHANNELS, SIZE, ism_dataset, ism_psf, ism_series
from _registry import SKIP_GUI, SKIP_PLOT_ONLY, Case

APR = "brighteyes_ism.analysis.APR_lib"
DECONV = "brighteyes_ism.analysis.Deconv_lib"
FRC = "brighteyes_ism.analysis.FRC_lib"
FOCUS = "brighteyes_ism.analysis.FocusISM_lib"
GRAPH = "brighteyes_ism.analysis.Graph_lib"
TOOLS = "brighteyes_ism.analysis.Tools_lib"

REF = 12


def _summed_series() -> np.ndarray:
    return ism_series().sum(axis=2)


def _frc_images() -> tuple[np.ndarray, np.ndarray]:
    series = ism_series()
    return series[:, :, 0, REF], series[:, :, 1, REF]


def _meshgrid() -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(SIZE) - SIZE // 2
    return np.meshgrid(axis, axis)


def _small_kernel_and_image() -> tuple[np.ndarray, np.ndarray]:
    kernel = np.exp(
        -((np.arange(3) - 1.0) ** 2)[:, None] / 2
        - ((np.arange(3) - 1.0) ** 2)[None, :] / 2
    )
    return kernel / kernel.sum(), ism_dataset()[:6, :6, 0]


def _gaussian_image() -> np.ndarray:
    return ism_psf()[:, :, REF]


def _focus_calibration() -> np.ndarray:
    return _summed_series()[8:24, 8:24]


def _color_map(call):
    color_map = call()
    color_map.var_bounds = [0, 1]
    color_map.int_bounds = [0, 20]
    image = color_map.image(ism_dataset()[:, :, 0], ism_dataset()[:, :, 1] / 20)
    return image, color_map.colorbar(16), color_map


def _cloud_input() -> np.ndarray:
    return ism_dataset()[:4, :5, :3]


def _kl_inputs() -> tuple[np.ndarray, np.ndarray]:
    dset = ism_dataset()
    return dset[:, :, 0] + 1, np.moveaxis(dset[:, :, 1:4], -1, 0) + 1


def _check_saturation(call):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = call(np.minimum(ism_dataset(), 15), np.full(CHANNELS, 15))
    return result, output.getvalue()


def _gauss_mult_var(call):
    x, y = _meshgrid()
    return call(x, y, np.array([1.0, -2.0]), np.array([[9.0, 2.0], [2.0, 16.0]]))


CASES: list[Case] = [
    Case(f"{APR}:hann2d", lambda f: f((SIZE, SIZE + 4))),
    Case(f"{APR}:ShiftVectors", lambda f: f(_summed_series(), 10, REF, filter_sigma=1)),
    Case(
        f"{APR}:ShiftVectors",
        lambda f: f(_summed_series(), 10, REF, apodize=False),
        "no apodization, no filter",
    ),
    Case(
        f"{APR}:Reassignment",
        lambda f: f(np.random.default_rng(1).normal(0, 1.5, (25, 2)), ism_dataset()),
        "interp",
    ),
    Case(
        f"{APR}:Reassignment",
        lambda f: f(
            np.random.default_rng(1).normal(0, 1.5, (25, 2)),
            ism_dataset(),
            mode="fourier",
        ),
        "fourier",
    ),
    Case(f"{APR}:APR", lambda f: f(_summed_series(), 10, REF, pxsize=0.05), "interp"),
    Case(
        f"{APR}:APR",
        lambda f: f(_summed_series(), 10, REF, pxsize=0.05, mode="fourier"),
        "fourier",
    ),
    Case(f"{DECONV}:gauss2d", lambda f: f(*_meshgrid(), 2.0, -3.0, 4.0)),
    Case(f"{DECONV}:disk2d", lambda f: f(*_meshgrid(), 2.0, -3.0, 6.0)),
    Case(f"{DECONV}:convolution_matrix", lambda f: f(*_small_kernel_and_image())),
    Case(
        f"{DECONV}:deconv_Wiener",
        lambda f: f(*_small_kernel_and_image(), reg=0.1),
        "Tikhonov",
    ),
    Case(
        f"{DECONV}:deconv_Wiener",
        lambda f: f(*_small_kernel_and_image(), reg=0.1, regularization="Laplace"),
        "Laplace",
    ),
    Case(
        f"{DECONV}:deconv_Wiener_FFT",
        lambda f: f(ism_psf()[:, :, REF], ism_dataset()[:, :, REF], reg=0.1),
    ),
    Case(
        f"{DECONV}:deconv_RL_FFT",
        lambda f: f(ism_psf()[:, :, REF], ism_dataset()[:, :, REF], max_iter=5),
        "last",
    ),
    Case(
        f"{DECONV}:deconv_RL_FFT",
        lambda f: f(
            ism_psf()[:, :, REF],
            ism_dataset()[:, :, REF],
            max_iter=5,
            reg=0.01,
            out="all",
        ),
        "all",
    ),
    Case(f"{DECONV}:PadDataset", lambda f: f(ism_dataset(), 3)),
    Case(
        f"{DECONV}:UnpadDataset",
        lambda f: f(np.pad(ism_dataset(), ((3, 3), (3, 3), (0, 0))), 3),
    ),
    Case(
        f"{DECONV}:MultiImg_RL_FFT",
        lambda f: f(ism_psf(), ism_dataset(), max_iter=3),
        "last",
    ),
    Case(
        f"{DECONV}:MultiImg_RL_FFT",
        lambda f: f(
            ism_psf(),
            ism_dataset(),
            max_iter=3,
            bkg=np.full((SIZE + 4, SIZE + 4, CHANNELS), 0.1),
            pad=2,
            reg=0.01,
        ),
        "last, padded",
    ),
    Case(
        f"{DECONV}:MultiImg_RL_FFT",
        lambda f: f(ism_psf(), ism_dataset(), max_iter=3, out="all"),
        "all",
    ),
    Case(
        f"{DECONV}:MultiImg_RL_FFT_2", lambda f: f(ism_psf(), ism_dataset(), max_iter=3)
    ),
    Case(f"{DECONV}:PSF_FRC", lambda f: f(*(ism_series()[:, :, k] for k in (0, 1)))),
    Case(f"{DECONV}:FRC_MultiImg_RL_FFT", lambda f: f(ism_series(), max_iter=3)),
    Case(
        f"{FRC}:smooth",
        lambda f: f(np.linspace(0, 1, 20), np.random.default_rng(0).normal(size=20)),
    ),
    Case(f"{FRC}:hann2d", lambda f: f(SIZE), "square"),
    Case(f"{FRC}:hann2d", lambda f: f(SIZE, SIZE + 6), "rectangular"),
    Case(
        f"{FRC}:radial_profile",
        lambda f: f(ism_dataset()[:, :, 0], (SIZE // 2, SIZE // 2)),
        "real",
    ),
    Case(
        f"{FRC}:radial_profile",
        lambda f: f(
            np.fft.fftshift(np.fft.fft2(ism_dataset()[:, :, 0])), (SIZE // 2, SIZE // 2)
        ),
        "complex",
    ),
    Case(f"{FRC}:FRC", lambda f: f(*_frc_images())),
    Case(
        f"{FRC}:fixed_threshold", lambda f: f(np.linspace(1, 0, 50), 1 / 7), "crossing"
    ),
    Case(
        f"{FRC}:fixed_threshold",
        lambda f: f(np.linspace(1, 0.5, 50), 1 / 7),
        "no crossing",
    ),
    Case(
        f"{FRC}:nsigma_threshold",
        lambda f: f(
            np.linspace(0, 1 / np.sqrt(2), 23),
            np.linspace(1, 0, 2300),
            np.zeros((32, 32)),
            3,
        ),
    ),
    Case(f"{FRC}:FRC_resolution", lambda f: f(*_frc_images(), px=0.05), "fixed"),
    Case(
        f"{FRC}:FRC_resolution", lambda f: f(*_frc_images(), method="3sigma"), "3sigma"
    ),
    Case(
        f"{FRC}:FRC_resolution", lambda f: f(*_frc_images(), method="5sigma"), "5sigma"
    ),
    Case(f"{FRC}:FRC_resolution", lambda f: f(*_frc_images(), smoothing="fit"), "fit"),
    Case(f"{FRC}:timeFRC", lambda f: f(ism_series()[:, :, :, REF], px=0.05), "even"),
    Case(
        f"{FRC}:timeFRC",
        lambda f: f(ism_series()[:, :, :3, REF], method="3sigma"),
        "odd, 3sigma",
    ),
    Case(
        f"{FOCUS}:fingerprint_model",
        lambda f: f(np.random.default_rng(0).normal(size=(2, 25)), 1.0, 1.5, 0.1),
    ),
    Case(f"{FOCUS}:gaussian", lambda f: f(*_meshgrid(), 1.0, -1.0, 2.0, 3.0)),
    Case(
        f"{FOCUS}:gaussian_model",
        lambda f: f(np.random.default_rng(0).normal(size=(2, 25)), 1.0, 0.3, 4.0),
    ),
    Case(f"{FOCUS}:FitFingerprint", lambda f: f(_focus_calibration())),
    Case(
        f"{FOCUS}:pixel_fit_1", lambda f: f(_summed_series()[10, 10], 1.2, 4.0), "fit"
    ),
    Case(
        f"{FOCUS}:pixel_fit_1",
        lambda f: f(_summed_series()[10, 10], 1.2, 4.0, threshold=1e9),
        "below threshold",
    ),
    Case(f"{FOCUS}:pixel_fit_2", lambda f: f(_summed_series()[10, 10], 1.2), "fit"),
    Case(
        f"{FOCUS}:pixel_fit_2",
        lambda f: f(_summed_series()[10, 10], 1.2, 2.0, threshold=1e9),
        "below threshold",
    ),
    Case(
        f"{FOCUS}:focusISM",
        lambda f: f(
            _summed_series()[:20, :20],
            calibration=_focus_calibration(),
            parallelize=False,
        ),
        "apr, summed",
    ),
    Case(
        f"{FOCUS}:focusISM",
        lambda f: f(
            _summed_series()[:20, :20],
            sigma_B_bound=3,
            apr=False,
            calibration=_focus_calibration(),
            sum_results=False,
            parallelize=False,
        ),
        "no apr, per channel",
    ),
    Case(f"{GRAPH}:ColorMap2D", _color_map),
    Case(f"{TOOLS}:sigmoid", lambda f: f(np.linspace(0, 1, 50), 0.3, 0.05)),
    Case(f"{TOOLS}:low_pass", lambda f: f(ism_dataset(), 0.2, 0.05), "real"),
    Case(
        f"{TOOLS}:low_pass",
        lambda f: f(
            np.fft.fftshift(np.fft.fftn(ism_dataset(), axes=(0, 1)), axes=(0, 1)),
            0.2,
            0.05,
            data="fourier",
        ),
        "fourier",
    ),
    Case(f"{TOOLS}:Reorder", lambda f: f(ism_dataset(), "xyc"), "add dimensions"),
    Case(
        f"{TOOLS}:Reorder", lambda f: f(ism_dataset(), "cyx", "xyc"), "same dimensions"
    ),
    Case(
        f"{TOOLS}:Reorder",
        lambda f: f(f(ism_dataset(), "xyc"), "rzxytc", "xyc"),
        "remove dimensions",
    ),
    Case(
        f"{TOOLS}:CropEdge",
        lambda f: f(ism_dataset(), npx=3, edges="lrud", order="xyc"),
    ),
    Case(f"{TOOLS}:DownSample", lambda f: f(ism_dataset(), ds=2, order="xyc")),
    Case(f"{TOOLS}:UpSample", lambda f: f(ism_dataset(), us=2, order="xyc"), "even"),
    Case(
        f"{TOOLS}:UpSample",
        lambda f: f(ism_dataset(), us=3, npx="odd", order="xyc"),
        "odd",
    ),
    Case(f"{TOOLS}:ArgMaxND", lambda f: f(ism_dataset())),
    Case(
        f"{TOOLS}:FWHM",
        lambda f: f(
            np.linspace(-5, 5, 101), np.exp(-(np.linspace(-5, 5, 101) ** 2) / 2), 0.4
        ),
    ),
    Case(f"{TOOLS}:RadialSpectrum", lambda f: f(ism_dataset()[:, :, 0], pxsize=0.05)),
    Case(f"{TOOLS}:fingerprint", lambda f: f(ism_dataset()), "summed"),
    Case(
        f"{TOOLS}:fingerprint",
        lambda f: f(np.moveaxis(ism_series(), 2, 0), volumetric=True),
        "volumetric",
    ),
    Case(f"{TOOLS}:point_cloud_from_img", lambda f: f(_cloud_input())),
    Case(f"{TOOLS}:kl_divergence", lambda f: f(*_kl_inputs()), "relative entropy"),
    Case(
        f"{TOOLS}:kl_divergence",
        lambda f: f(*_kl_inputs(), intensity_offset=True),
        "intensity offset",
    ),
    Case(
        f"{TOOLS}:kl_divergence",
        lambda f: f(*_kl_inputs(), normalize_entries=True),
        "normalized",
    ),
    Case(
        f"{TOOLS}:normalized_absolute_difference",
        lambda f: f(ism_dataset()[:, :, 0] + 1, ism_dataset()[:, :, 1] + 1),
    ),
    Case(f"{TOOLS}:check_saturation", _check_saturation),
    Case(f"{TOOLS}:GaussMultVar", _gauss_mult_var),
    Case(
        f"{TOOLS}:fit_to_gaussian",
        lambda f: f(_gaussian_image(), 1.0, p0=(0, 0, 4, 0, 4, 0.04)),
        "no baseline",
    ),
    Case(
        f"{TOOLS}:fit_to_gaussian",
        lambda f: f(
            _gaussian_image() + 0.001,
            1.0,
            baseline=True,
            p0=(0, 0, 4, 0, 4, 0.04, 0.001),
        ),
        "baseline",
    ),
]

SKIPPED: dict[str, str] = {
    f"{DECONV}:FRC_MultiImg_RL_FFT_2": "fails with the old code too: ValueError: too many values to unpack (expected 5)",
    f"{FRC}:plotFRC": SKIP_PLOT_ONLY,
    f"{FOCUS}:Selector": SKIP_GUI,
    f"{GRAPH}:ShowImg": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowStack": SKIP_PLOT_ONLY,
    f"{GRAPH}:StackSlider": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowDataset": SKIP_PLOT_ONLY,
    f"{GRAPH}:PlotShiftVectors": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowFingerprint": SKIP_PLOT_ONLY,
    f"{GRAPH}:show_flim": SKIP_PLOT_ONLY,
    f"{GRAPH}:depth_stack": SKIP_PLOT_ONLY,
}
