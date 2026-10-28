import contextlib
import io

import numpy as np
from scipy.ndimage import gaussian_filter

from .._inputs import ism_dataset, ism_psf
from .._registry import SKIP_GUI, SKIP_PLOT_ONLY, Case

APR = "brighteyes_ism.analysis.APR_lib"
DECONV = "brighteyes_ism.analysis.Deconv_lib"
FRC = "brighteyes_ism.analysis.FRC_lib"
FOCUS = "brighteyes_ism.analysis.FocusISM_lib"
GRAPH = "brighteyes_ism.analysis.Graph_lib"
TOOLS = "brighteyes_ism.analysis.Tools_lib"


def _small_dataset() -> np.ndarray:
    return ism_dataset()[:16, :16, :]


def _small_psf() -> np.ndarray:
    return ism_psf()[:16, :16, :]


def _grid(n: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(n) - n // 2
    return np.meshgrid(axis, axis)


def _grid_points(n: int) -> np.ndarray:
    return np.vstack([a.ravel() for a in _grid(n)]).astype(float)


def _tiny_image() -> tuple[np.ndarray, np.ndarray]:
    kernel = np.random.default_rng(1).random((3, 3)) + 0.1
    image = np.random.default_rng(2).random((5, 5)) + 0.1
    return kernel, image


def _gaussian_image(n: int = 32) -> np.ndarray:
    x, y = _grid(n)
    noise = 0.01 * np.random.default_rng(3).random((n, n))
    return np.exp(-(x**2 + y**2) / (2 * 3.0**2)) + noise


def _smooth(seed: int) -> np.ndarray:
    field = np.random.default_rng(0).random((32, 32, 25))
    return gaussian_filter(field, (2, 2, 0)) + 0.002 * np.random.default_rng(seed).random(field.shape)


def _frc_pair() -> tuple[np.ndarray, np.ndarray]:
    return _smooth(1)[:, :, 12], _smooth(2)[:, :, 12]


def _capture(function, *args, **kwargs) -> tuple[object, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = function(*args, **kwargs)
    return result, buffer.getvalue()


def _fingerprint_stack() -> np.ndarray:
    x, y = _grid(5)
    pattern = np.exp(-(x**2 + y**2) / (2 * 1.2**2)).ravel()
    rng = np.random.default_rng(5)
    return 50 * pattern[None, None, :] * (1 + 0.05 * rng.random((6, 6, 25)))


def _fit_image() -> np.ndarray:
    x, y = _grid(16)
    noise = 0.001 * np.random.default_rng(9).random((16, 16))
    return np.exp(-(x**2 + y**2) / (2 * 4.0**2)) + noise


def _shift_vectors() -> np.ndarray:
    return np.random.default_rng(6).normal(size=(25, 2))


def _color_map_2d(cls):
    cmap = cls()
    intensity = np.random.default_rng(10).random((8, 8))
    variable = np.random.default_rng(11).random((8, 8))
    cmap.equalize = True
    first = cmap.image(intensity, variable)
    cmap.equalize = False
    cmap.var_bounds = [0.1, 0.9]
    cmap.int_bounds = [0.0, 0.8]
    return first, cmap.image(intensity, variable), cmap.colorbar(8), cmap


CASES: list[Case] = [
    Case(f"{APR}:hann2d", lambda f: f((16, 20))),
    Case(f"{APR}:ShiftVectors", lambda f: f(_small_dataset(), 10, 12, apodize=True, filter_sigma=1), "apodize_filter"),
    Case(f"{APR}:ShiftVectors", lambda f: f(_small_dataset(), 10, 12, apodize=False, filter_sigma=0), "plain"),
    Case(f"{APR}:Reassignment", lambda f: f(_shift_vectors(), _small_dataset(), mode="interp"), "interp"),
    Case(f"{APR}:Reassignment", lambda f: f(_shift_vectors(), _small_dataset(), mode="fourier"), "fourier"),
    Case(f"{APR}:APR", lambda f: f(_small_dataset(), 10, 12, pxsize=0.05, mode="interp"), "interp"),
    Case(f"{APR}:APR", lambda f: f(_small_dataset(), 10, 12, pxsize=0.05, apodize=False, filter_sigma=0, mode="fourier"), "fourier"),
    Case(f"{DECONV}:gauss2d", lambda f: f(*_grid(16), 1, -1, 2.5)),
    Case(f"{DECONV}:disk2d", lambda f: f(*_grid(16), 1, -1, 4)),
    Case(f"{DECONV}:convolution_matrix", lambda f: f(*_tiny_image())),
    Case(f"{DECONV}:deconv_Wiener", lambda f: f(*_tiny_image(), reg=0.1, regularization="Tikhonov"), "tikhonov"),
    Case(f"{DECONV}:deconv_Wiener", lambda f: f(*_tiny_image(), reg=0.1, regularization="Laplace"), "laplace"),
    Case(f"{DECONV}:deconv_Wiener_FFT", lambda f: f(_small_psf()[:, :, 12], _small_dataset()[:, :, 12], reg=0.1)),
    Case(f"{DECONV}:deconv_RL_FFT", lambda f: f(_small_psf()[:, :, 12], _small_dataset()[:, :, 12], max_iter=5, reg=0.01, out="last"), "last"),
    Case(f"{DECONV}:deconv_RL_FFT", lambda f: f(_small_psf()[:, :, 12], _small_dataset()[:, :, 12], max_iter=5, out="all"), "all"),
    Case(f"{DECONV}:MultiImg_RL_FFT", lambda f: f(_small_psf(), _small_dataset(), max_iter=3, out="last"), "last"),
    Case(f"{DECONV}:MultiImg_RL_FFT", lambda f: f(_small_psf(), _small_dataset(), bkg=np.full((16, 16, 25), 0.01), max_iter=3, reg=0.01, out="all"), "all_bkg"),
    # out="all" with pad raises ValueError in the old code, so padding is tested with out="last".
    Case(f"{DECONV}:MultiImg_RL_FFT", lambda f: f(_small_psf(), _small_dataset(), bkg=np.full((20, 20, 25), 0.01), max_iter=3, pad=2, out="last"), "padded_bkg"),
    Case(f"{DECONV}:PadDataset", lambda f: f(_small_dataset(), 3)),
    Case(f"{DECONV}:UnpadDataset", lambda f: f(np.pad(_small_dataset(), ((3, 3), (3, 3), (0, 0))), 3)),
    Case(f"{DECONV}:MultiImg_RL_FFT_2", lambda f: f(_small_psf(), _small_dataset(), max_iter=3, pad=2)),
    Case(f"{DECONV}:PSF_FRC", lambda f: f(_smooth(1), _smooth(2))),
    Case(f"{DECONV}:FRC_MultiImg_RL_FFT", lambda f: f(np.stack((_smooth(1), _smooth(2)), axis=-2), max_iter=3)),
    Case(f"{FRC}:smooth", lambda f: f(np.linspace(0, 1, 30), np.random.default_rng(7).random(30))),
    Case(f"{FRC}:hann2d", lambda f: f(16, 20), "rectangular"),
    Case(f"{FRC}:hann2d", lambda f: f(16), "square"),
    Case(f"{FRC}:radial_profile", lambda f: f(np.random.default_rng(8).random((16, 16)), (8, 8))),
    Case(f"{FRC}:FRC", lambda f: f(*_frc_pair())),
    Case(f"{FRC}:fixed_threshold", lambda f: f(np.linspace(1, 0, 40), 1 / 7)),
    Case(f"{FRC}:nsigma_threshold", lambda f: f(np.linspace(0, 0.7, 23), np.linspace(1, 0, 23 * 100), _frc_pair()[0], 3)),
    Case(f"{FRC}:FRC_resolution", lambda f: f(*_frc_pair(), px=0.05, method="fixed", smoothing="lowess"), "fixed_lowess"),
    Case(f"{FRC}:FRC_resolution", lambda f: f(*_frc_pair(), px=0.05, method="3sigma", smoothing="lowess"), "3sigma_lowess"),
    Case(f"{FRC}:FRC_resolution", lambda f: f(*_frc_pair(), px=0.05, method="5sigma", smoothing="lowess"), "5sigma_lowess"),
    Case(f"{FRC}:FRC_resolution", lambda f: f(*_frc_pair(), px=0.05, method="fixed", smoothing="fit"), "fixed_fit"),
    Case(f"{FRC}:timeFRC", lambda f: f(_smooth(1), px=0.05, method="fixed", smoothing="lowess"), "even"),
    Case(f"{FRC}:timeFRC", lambda f: f(_smooth(1)[:, :, :-1], px=0.05, method="fixed", smoothing="lowess"), "odd"),
    Case(f"{FOCUS}:fingerprint_model", lambda f: f(_grid_points(5), 2.0, 1.2, 0.1)),
    Case(f"{FOCUS}:gaussian", lambda f: f(*_grid(5), 0.0, 0.0, 1.0, 1.2)),
    Case(f"{FOCUS}:gaussian_model", lambda f: f(_grid_points(5), 1.2, 0.2, 3.6)),
    Case(f"{FOCUS}:FitFingerprint", lambda f: f(_fingerprint_stack(), p0=[1, 1], Ndet=5)),
    Case(f"{FOCUS}:pixel_fit_1", lambda f: f(_fingerprint_stack()[0, 0], 1.2, 3.6, threshold=0)),
    Case(f"{FOCUS}:pixel_fit_2", lambda f: f(_fingerprint_stack()[0, 0], 1.2, sigma_B_bound=2.4, threshold=0)),
    Case(f"{FOCUS}:pixel_fit_1", lambda f: f(_fingerprint_stack()[0, 0], 1.2, 3.6, threshold=1e6), "below_threshold"),
    Case(f"{FOCUS}:pixel_fit_2", lambda f: f(_fingerprint_stack()[0, 0], 1.2, sigma_B_bound=2.4, threshold=1e6), "below_threshold"),
    Case(f"{FOCUS}:focusISM", lambda f: _capture(f, _fingerprint_stack(), apr=False, calibration=_fingerprint_stack(), sum_results=True, parallelize=False, threshold=433.8), "mid_threshold"),
    Case(f"{FOCUS}:focusISM", lambda f: _capture(f, _fingerprint_stack(), apr=False, calibration=_fingerprint_stack(), sum_results=True, parallelize=False), "sum"),
    Case(f"{FOCUS}:focusISM", lambda f: _capture(f, _fingerprint_stack(), apr=True, calibration=_fingerprint_stack(), sum_results=False, parallelize=False), "apr_no_sum"),
    Case(f"{GRAPH}:ColorMap2D", _color_map_2d),
    Case(f"{TOOLS}:sigmoid", lambda f: f(np.linspace(0, 1, 20), 0.4, 0.05)),
    Case(f"{TOOLS}:low_pass", lambda f: f(ism_dataset()[:, :, 0], 0.2, 0.05, data="real"), "real"),
    Case(f"{TOOLS}:low_pass", lambda f: f(np.fft.fftshift(np.fft.fft2(ism_dataset()[:, :, 0])), 0.2, 0.05, data="fourier"), "fourier"),
    Case(f"{TOOLS}:Reorder", lambda f: f(ism_dataset(), "xyc"), "add_dimensions"),
    Case(f"{TOOLS}:Reorder", lambda f: f(ism_dataset(), "yxc", "xyc"), "same_size_swap"),
    Case(f"{TOOLS}:Reorder", lambda f: f(ism_dataset()[None, None, :, :, None, :], "rzxytc", "xyc"), "drop_dimensions"),
    Case(f"{TOOLS}:CropEdge", lambda f: f(ism_dataset(), npx=3, edges="lrud", order="xyc")),
    Case(f"{TOOLS}:DownSample", lambda f: f(ism_dataset(), ds=2, order="xyc")),
    Case(f"{TOOLS}:UpSample", lambda f: f(ism_dataset(), us=2, npx="even", order="xyc"), "even"),
    Case(f"{TOOLS}:UpSample", lambda f: f(ism_dataset(), us=2, npx="odd", order="xyc"), "odd"),
    Case(f"{TOOLS}:ArgMaxND", lambda f: f(ism_dataset())),
    Case(f"{TOOLS}:FWHM", lambda f: f(np.linspace(-10, 10, 201), np.exp(-np.linspace(-10, 10, 201) ** 2 / (2 * 2.0**2)))),
    Case(f"{TOOLS}:RadialSpectrum", lambda f: f(_gaussian_image(), pxsize=0.05, normalize=True), "normalized"),
    Case(f"{TOOLS}:RadialSpectrum", lambda f: f(_gaussian_image(), pxsize=0.05, normalize=False), "raw"),
    Case(f"{TOOLS}:fingerprint", lambda f: f(ism_dataset(), volumetric=False), "single"),
    Case(f"{TOOLS}:fingerprint", lambda f: f(np.stack((ism_dataset(), 2 * ism_dataset())), volumetric=True), "volumetric"),
    Case(f"{TOOLS}:point_cloud_from_img", lambda f: f(ism_dataset()[:4, :4, :3])),
    Case(f"{TOOLS}:kl_divergence", lambda f: f(_gaussian_image() + 0.1, np.stack((_gaussian_image(), 2 * _gaussian_image())) + 0.1, remove_inf=True, intensity_offset=False, normalize_entries=False), "rel_entr"),
    Case(f"{TOOLS}:kl_divergence", lambda f: f(_gaussian_image() + 0.1, np.stack((_gaussian_image(), 2 * _gaussian_image())) + 0.1, remove_inf=True, intensity_offset=True, normalize_entries=True), "kl_div_normalized"),
    Case(f"{TOOLS}:kl_divergence", lambda f: f(_gaussian_image() + 0.1, np.stack((_gaussian_image(), 2 * _gaussian_image())) + 0.1, remove_inf=False), "keep_inf"),
    Case(f"{TOOLS}:normalized_absolute_difference", lambda f: f(_gaussian_image(), 2 * _gaussian_image() ** 2)),
    Case(f"{TOOLS}:check_saturation", lambda f: _capture(f, (ism_dataset() * 40).astype(int), sat_map=None), "default_map"),
    Case(f"{TOOLS}:check_saturation", lambda f: _capture(f, (ism_dataset() * 40).astype(int), sat_map=np.full(25, 20)), "custom_map"),
    Case(f"{TOOLS}:GaussMultVar", lambda f: f(*_grid(16), np.array([1.0, -1.0]), np.array([[4.0, 1.0], [1.0, 9.0]]))),
    Case(f"{TOOLS}:fit_to_gaussian", lambda f: f(_fit_image(), 1.0, baseline=False, p0=(0, 0, 20, 0, 20, 1)), "no_baseline"),
    Case(f"{TOOLS}:fit_to_gaussian", lambda f: f(_fit_image() + 0.1, 1.0, baseline=True, p0=(0, 0, 20, 0, 20, 1, 0)), "baseline"),
]

SKIPPED: dict[str, str] = {
    f"{DECONV}:FRC_MultiImg_RL_FFT_2": "TypeError: APR() got an unexpected keyword argument 'degree' (old code too)",
    f"{FRC}:plotFRC": SKIP_PLOT_ONLY,
    f"{FOCUS}:Selector": SKIP_GUI,
    f"{GRAPH}:PlotShiftVectors": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowDataset": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowFingerprint": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowImg": SKIP_PLOT_ONLY,
    f"{GRAPH}:ShowStack": SKIP_PLOT_ONLY,
    f"{GRAPH}:StackSlider": SKIP_PLOT_ONLY,
    f"{GRAPH}:depth_stack": SKIP_PLOT_ONLY,
    f"{GRAPH}:show_flim": SKIP_PLOT_ONLY,
}
