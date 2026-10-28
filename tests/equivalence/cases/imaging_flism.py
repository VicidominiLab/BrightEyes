import numpy as np
import torch

from .._inputs import (
    flim_decay,
    flim_decay_histogram,
    flim_image,
    flim_time_axis,
)
from .._registry import SKIP_FILE_IO, SKIP_PLOT_ONLY, Case, resolve

DATA = "brighteyes_flism.flism_data"
FIT = "brighteyes_flism._fit_maps"
GRAPH = "brighteyes_flism.graph_tools"
PHASOR = "brighteyes_flism.tools_phasor"

PERIOD_NS = 25.0
FREQ_HZ = 1e9 / PERIOD_NS
EDGE_STYLE = {
    "minTau": 1.0,
    "maxTau": 4.0,
    "satFactor": 0.7,
    "minInt": 0.0,
    "maxInt": 100.0,
    "outOfBoundsHue": 0.9,
}


def _error(function, *args, **kwargs) -> BaseException:
    try:
        function(*args, **kwargs)
    except Exception as error:
        return error
    raise AssertionError("the call did not raise")


def _lifetime_map() -> np.ndarray:
    rng = np.random.default_rng(4)
    lifetime = 1.0 + 4.0 * rng.random((8, 8))
    lifetime[0, 0] = np.nan
    lifetime[3, 3] = np.inf
    return lifetime


def _intensity_map() -> np.ndarray:
    rng = np.random.default_rng(5)
    intensity = 100.0 * rng.random((8, 8))
    intensity[1, 1] = np.nan
    return intensity


def _phasor_map() -> np.ndarray:
    rng = np.random.default_rng(6)
    return rng.random((8, 8)) * np.exp(1j * rng.random((8, 8)))


def _flim_data() -> np.ndarray:
    return flim_image(6)


def _reference_data() -> np.ndarray:
    return flim_image(6, tau_ns=2.5)


def _flim_data_session(cls, **kwargs):
    obj = cls(**kwargs)
    obj.data_hist = flim_decay_histogram()
    obj.data_hist_irf = flim_decay_histogram()[::-1].copy()
    obj.data_laser_hist = flim_decay(0.5, 3.0)
    obj.data_laser_hist_irf = flim_decay(0.5, 1.0)
    results = (
        obj.calculate_phasor_global(),
        obj.calculate_phasor_global_irf(),
        obj.calculate_phasor_laser(),
        obj.calculate_phasor_laser_irf(),
        obj.calculate_correction_coeff(),
        obj.calculate_correction_coeff(dfd_freq=FREQ_HZ, tau=2e-9),
        obj.phasor_global_corrected(0.9 + 0.1j),
        obj.phasor_global_corrected(0.9 + 0.1j, laser_correction=False),
        obj.phasor_global_corrected_irf(0.9 + 0.1j),
        obj.phasor_global_corrected_irf(0.9 + 0.1j, laser_correction=False),
    )
    return results, obj


def _flux_split_image() -> tuple[np.ndarray, float]:
    data = flim_image(4)[:, :, :, 0]
    return data, float(np.median(data.sum(-1)))


def _equalizer_reference() -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.concatenate(
        (rng.normal(2.0, 0.3, 500), rng.normal(4.0, 0.5, 300), [np.nan, 9.0])
    )


def _equalizer() -> tuple[np.ndarray, np.ndarray]:
    return resolve(f"{PHASOR}:build_lifetime_equalizer")(
        _equalizer_reference(), (1.0, 5.0), strength=1.5
    )


def _apply_equalizer(function):
    support, axis = _equalizer()
    return function(_lifetime_map() * 2.0, support, axis)


def _cmap_inputs() -> tuple[np.ndarray, np.ndarray]:
    return (
        100.0 * np.random.default_rng(8).random((8, 8)),
        np.random.default_rng(9).random((8, 8)) * 6.0,
    )


def _fit_data() -> np.ndarray:
    return np.stack([flim_decay(3.0, 5.0)] * 4).reshape(2, 2, 64) * 5


def _fit_irf() -> np.ndarray:
    return flim_decay(0.4, 5.0)


CASES: list[Case] = [
    Case(f"{DATA}:FlimData", lambda cls: _flim_data_session(cls), "empty_object"),
    Case(
        f"{DATA}:FlimData",
        lambda cls: _flim_data_session(cls, correction_coeff=[0.5, 0.2]),
        "list_coefficient",
    ),
    Case(
        f"{DATA}:FlimData",
        lambda cls: _flim_data_session(cls, correction_coeff=np.array([0.5, -0.2])),
        "array_coefficient",
    ),
    Case(
        f"{DATA}:FlimData",
        lambda cls: _flim_data_session(cls, correction_coeff=0.5 + 0.2j),
        "complex_coefficient",
    ),
    Case(
        f"{GRAPH}:crop_2d",
        lambda f: f(np.arange(100.0).reshape(10, 10), crop=2),
        "crop",
    ),
    Case(
        f"{GRAPH}:crop_2d",
        lambda f: f(np.arange(100.0).reshape(10, 10), crop=0),
        "no_crop",
    ),
    Case(
        f"{GRAPH}:crop_2d", lambda f: _error(f, np.zeros((4, 4)), crop=2), "too_large"
    ),
    Case(
        f"{GRAPH}:normalize_histogram",
        lambda f: f(flim_decay_histogram()[:, 0]),
        "normal",
    ),
    Case(f"{GRAPH}:normalize_histogram", lambda f: f(np.zeros(8)), "zero_sum"),
    Case(
        f"{GRAPH}:threshold_lifetime_map",
        lambda f: f(_lifetime_map(), intensity=_intensity_map(), threshold=0.3),
        "intensity",
    ),
    Case(
        f"{GRAPH}:threshold_lifetime_map",
        lambda f: f(
            _lifetime_map(),
            intensity=_intensity_map(),
            threshold=0.3,
            finite_only=False,
        ),
        "intensity_keep_nonfinite",
    ),
    Case(
        f"{GRAPH}:threshold_lifetime_map",
        lambda f: f(_lifetime_map(), intensity=np.zeros((8, 8))),
        "zero_intensity",
    ),
    Case(
        f"{GRAPH}:threshold_lifetime_map", lambda f: f(_lifetime_map()), "no_intensity"
    ),
    Case(
        f"{GRAPH}:threshold_lifetime_map",
        lambda f: _error(f, _lifetime_map(), intensity=np.ones((3, 3))),
        "shape_mismatch",
    ),
    Case(
        f"{GRAPH}:weighted_lifetime_stats", lambda f: f(_lifetime_map()), "unweighted"
    ),
    Case(
        f"{GRAPH}:weighted_lifetime_stats",
        lambda f: f(_lifetime_map(), weights=_intensity_map()),
        "weighted",
    ),
    Case(f"{GRAPH}:weighted_lifetime_stats", lambda f: f(np.full(4, np.nan)), "empty"),
    Case(
        f"{GRAPH}:weighted_lifetime_stats",
        lambda f: f(np.arange(4.0), weights=np.zeros(4)),
        "zero_weights",
    ),
    Case(
        f"{PHASOR}:correct_phasor",
        lambda f: f(_phasor_map(), 0.4 + 0.3j, coeff=0.9 + 0.1j),
        "laser_correction",
    ),
    Case(
        f"{PHASOR}:correct_phasor",
        lambda f: f(
            _phasor_map(), 0.4 + 0.3j, coeff=0.9 + 0.1j, laser_correction=False
        ),
        "no_laser_correction",
    ),
    Case(
        f"{PHASOR}:correct_phasor",
        lambda f: _error(f, None, 0.4 + 0.3j),
        "missing_phasor",
    ),
    Case(
        f"{PHASOR}:sum_adjacent_pixel",
        lambda f: f(np.random.default_rng(0).random((10, 10)), n=2),
        "2d",
    ),
    Case(
        f"{PHASOR}:sum_adjacent_pixel",
        lambda f: f(np.random.default_rng(0).random((2, 2, 8, 8, 5, 3)), n=2),
        "6d",
    ),
    Case(
        f"{PHASOR}:sum_adjacent_pixel",
        lambda f: f(np.random.default_rng(0).random((10, 10)), n=1),
        "n_one",
    ),
    Case(
        f"{PHASOR}:periodic_single_exponential_model",
        lambda f: f(flim_time_axis(), 100.0, 3.0, PERIOD_NS),
    ),
    Case(
        f"{PHASOR}:pad_tensor",
        lambda f: f(torch.arange(24.0).reshape(4, 6), 2, 3, 1, mode="reflect"),
        "reflect",
    ),
    Case(
        f"{PHASOR}:pad_tensor",
        lambda f: f(torch.arange(24.0).reshape(4, 6), 2, 3, 0, mode="replicate"),
        "replicate",
    ),
    Case(
        f"{PHASOR}:torch_median_filter",
        lambda f: f(
            torch.as_tensor(flim_decay_histogram()),
            window_size=3,
            dims=[0],
            mode="reflect",
        ),
        "reflect",
    ),
    Case(
        f"{PHASOR}:torch_median_filter",
        lambda f: f(
            torch.as_tensor(flim_decay_histogram()),
            window_size=5,
            dims=[0, 1],
            mode="replicate",
        ),
        "replicate_two_dims",
    ),
    Case(
        f"{PHASOR}:partial_convolution_fft",
        lambda f: f(
            torch.as_tensor(flim_decay()),
            torch.as_tensor(flim_decay(1.0, 0.0)),
            dim1="t",
            dim2="t",
            axis="t",
            fourier=(False, False),
        ),
        "time_domain",
    ),
    Case(
        f"{PHASOR}:partial_convolution_fft",
        lambda f: f(
            torch.as_tensor(flim_decay_histogram()),
            torch.fft.fftn(torch.as_tensor(flim_decay(1.0, 0.0)), dim=0),
            dim1="tc",
            dim2="t",
            axis="t",
            fourier=(False, True),
        ),
        "kernel_in_fourier_domain",
    ),
    Case(
        f"{PHASOR}:richardson_lucy_deconvolution",
        lambda f: f(
            torch.as_tensor(flim_decay(2.0, 8.0)),
            flim_time_axis(),
            1.0,
            3.0,
            PERIOD_NS,
            iterations=10,
            regularization=3,
        ),
        "regularized",
    ),
    Case(
        f"{PHASOR}:richardson_lucy_deconvolution",
        lambda f: f(
            flim_decay(2.0, 8.0),
            flim_time_axis(),
            1.0,
            3.0,
            PERIOD_NS,
            iterations=10,
            regularization=1,
        ),
        "array_input_unregularized",
    ),
    Case(
        f"{PHASOR}:estimate_irf",
        lambda f: f(
            torch.as_tensor(flim_decay(2.0, 8.0)),
            flim_time_axis(),
            1.0,
            3.0,
            PERIOD_NS,
            iterations=10,
        ),
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        lambda f: f(
            flim_time_axis(), flim_decay_histogram(), axis=0, return_bounds=True
        ),
        "multi_channel_bounds",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        lambda f: f(flim_time_axis(), flim_decay(), threshold=0.2),
        "single_channel",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        lambda f: _error(f, flim_time_axis(), flim_decay(), window_length=10),
        "even_window",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(flim_decay(), flim_time_axis(), PERIOD_NS / 64, 64, PERIOD_NS),
        "fit",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(
            flim_decay(1.0, 20.0),
            flim_time_axis(),
            PERIOD_NS / 64,
            64,
            PERIOD_NS,
            start_level=0.9,
            end_level=0.1,
        ),
        "wrapped_peak",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(flim_decay(), flim_time_axis(), -1.0, 64, PERIOD_NS),
        "invalid_bin_width",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(np.ones(64), flim_time_axis(), PERIOD_NS / 64, 64, PERIOD_NS),
        "flat_decay",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        lambda f: f(flim_decay(), 0.5, repetition_rate_MHz=40.0),
        "plain",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        lambda f: f(
            flim_decay(),
            0.5,
            repetition_rate_MHz=40.0,
            background_per_bin=1.0,
            truncate_ns=12.0,
        ),
        "background_and_truncation",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        lambda f: f(np.zeros(64), 0.5),
        "no_photons",
    ),
    Case(f"{PHASOR}:flatten_and_remove_nan", lambda f: f(_lifetime_map()), "single"),
    Case(
        f"{PHASOR}:flatten_and_remove_nan",
        lambda f: f(_lifetime_map(), _intensity_map()),
        "pair",
    ),
    Case(
        f"{PHASOR}:g0s0_to_m_phi",
        lambda f: f(np.array([0.3, -0.4]), np.array([0.4, 0.2])),
    ),
    Case(
        f"{PHASOR}:m_phi_to_g0s0",
        lambda f: f(np.array([0.5, 0.9]), np.array([0.3, -2.0])),
    ),
    Case(
        f"{PHASOR}:phasor",
        lambda f: f(flim_image(4)[:, :, :, 0], threshold=0, harmonic=1),
        "image",
    ),
    Case(
        f"{PHASOR}:phasor",
        lambda f: f(
            _flux_split_image()[0], threshold=_flux_split_image()[1], harmonic=2
        ),
        "threshold_and_harmonic",
    ),
    Case(
        f"{PHASOR}:phasor",
        lambda f: f(np.moveaxis(flim_image(4)[:, :, :, 0], -1, 0), time_axis=0),
        "time_axis_first",
    ),
    Case(f"{PHASOR}:phasor", lambda f: f(flim_decay()), "single_decay"),
    Case(f"{PHASOR}:calculate_phasor", lambda f: f(flim_decay()), "1d"),
    Case(
        f"{PHASOR}:calculate_phasor",
        lambda f: f(flim_decay(), threshold=10**6),
        "1d_below_threshold",
    ),
    Case(
        f"{PHASOR}:calculate_phasor",
        lambda f: f(flim_decay(), harmonic=2),
        "1d_second_harmonic",
    ),
    Case(
        f"{PHASOR}:calculate_phasor",
        lambda f: f(
            np.where(np.arange(25) == 3, 0.0, flim_decay_histogram()), threshold=5
        ),
        "2d_with_empty_channel",
    ),
    Case(
        f"{PHASOR}:calculate_phasor",
        lambda f: f(
            np.where(np.arange(4)[None, :, None] == 2, 0.0, flim_image(4)[:, :, :, 0]),
            threshold=5,
        ),
        "3d_with_empty_pixels",
    ),
    Case(
        f"{PHASOR}:calculate_phasor",
        lambda f: _error(f, np.zeros((2, 2, 4, 4))),
        "four_dimensions",
    ),
    Case(
        f"{PHASOR}:calculate_tau_phi",
        lambda f: f(np.array([0.4, 0.6]), np.array([0.3, 0.2]), dfd_freq=FREQ_HZ),
        "g0_s0",
    ),
    Case(
        f"{PHASOR}:calculate_tau_phi",
        lambda f: f(np.array([0.4 + 0.3j, 0.6 + 0.2j])),
        "complex",
    ),
    Case(
        f"{PHASOR}:calculate_tau_m",
        lambda f: f(np.array([0.4, 0.6]), np.array([0.3, 0.2]), dfd_freq=FREQ_HZ),
        "g0_s0",
    ),
    Case(
        f"{PHASOR}:calculate_tau_m",
        lambda f: f(np.array([0.4 + 0.3j, 0.6 + 0.2j])),
        "complex",
    ),
    Case(
        f"{PHASOR}:calculate_m_phi_tau_phi_tau_m",
        lambda f: f(np.array([0.4, 0.6]), np.array([0.3, 0.2]), dfd_freq=FREQ_HZ),
        "g0_s0",
    ),
    Case(
        f"{PHASOR}:calculate_m_phi_tau_phi_tau_m",
        lambda f: f(np.array([0.4 + 0.3j, 0.6 + 0.2j])),
        "complex",
    ),
    Case(
        f"{PHASOR}:calculate_phasor_on_img",
        lambda f: f(
            flim_image(3)[:, :, :, 0].reshape(1, 3, 3, 64), threshold=1, harmonic=1
        ),
        "4d",
    ),
    Case(
        f"{PHASOR}:calculate_phasor_on_img",
        lambda f: f(flim_image(3)[:, :, :, 0].reshape(1, 1, 3, 3, 64)),
        "5d",
    ),
    Case(
        f"{PHASOR}:calculate_phasor_on_img",
        lambda f: _error(f, np.zeros((3, 3, 8))),
        "3d",
    ),
    Case(
        f"{PHASOR}:fourier_shift", lambda f: f(flim_decay_histogram(), shift_angle=0.01)
    ),
    Case(
        f"{PHASOR}:cmap2d",
        lambda f: f(*_cmap_inputs(), EDGE_STYLE | {"invertColormap": False}),
        "direct",
    ),
    Case(
        f"{PHASOR}:cmap2d",
        lambda f: f(*_cmap_inputs(), EDGE_STYLE | {"invertColormap": True}),
        "inverted",
    ),
    Case(
        f"{PHASOR}:linear_shift", lambda f: f(flim_decay(), 2.5, cyclic=True), "cyclic"
    ),
    Case(
        f"{PHASOR}:linear_shift",
        lambda f: f(flim_decay(), -3.2, cyclic=False),
        "not_cyclic",
    ),
    Case(
        f"{PHASOR}:correction_phasor",
        lambda f: f(flim_decay(0.5, 3.0), flim_decay(0.5, 1.0)),
    ),
    Case(
        f"{PHASOR}:threshold_intensity",
        lambda f: f(_intensity_map()[2:, 2:], threshold=0.5),
    ),
    Case(
        f"{PHASOR}:threshold_phasor",
        lambda f: f(_intensity_map()[2:, 2:], _phasor_map()[2:, 2:], threshold=0.5),
    ),
    Case(f"{PHASOR}:calculate_irf_correction", lambda f: f(flim_decay(0.5, 4.0), 64)),
    Case(
        f"{PHASOR}:m_phi_to_complex",
        lambda f: f(np.array([0.5, 0.9]), np.array([0.3, -2.0])),
    ),
    Case(
        f"{PHASOR}:align_decays", lambda f: f(_flim_data(), np.linspace(-2.0, 2.0, 25))
    ),
    Case(
        f"{PHASOR}:align_image", lambda f: f(_flim_data(), np.linspace(-2.0, 2.0, 25))
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(_equalizer_reference(), (1.0, 5.0)),
        "default",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(
            _equalizer_reference(),
            (1.0, 5.0),
            weights=np.linspace(0.0, 1.0, 802),
            strength=2.0,
            equalization_bins=100,
        ),
        "weighted_strong",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(_equalizer_reference(), (2.0, 2.0)),
        "equal_bounds",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(_equalizer_reference(), (1.0, 5.0), strength=0.0),
        "zero_strength",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(_equalizer_reference(), (20.0, 30.0)),
        "no_reference_in_bounds",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: _error(f, _equalizer_reference(), (1.0, 5.0), weights=np.ones(3)),
        "weights_mismatch",
    ),
    Case(f"{PHASOR}:apply_lifetime_equalizer", _apply_equalizer),
    Case(
        f"{PHASOR}:equalized_lifetime_tick_values",
        lambda f: f(_equalizer_reference(), (1.0, 5.0), n_ticks=7, strength=2.0),
    ),
    Case(
        f"{PHASOR}:median_phasors",
        lambda f: f(np.abs(_phasor_map()), window=3),
        "real_map",
    ),
    Case(
        f"{PHASOR}:median_phasors",
        lambda f: _error(f, _phasor_map(), window=3),
        "complex_map",
    ),
    # skimage 0.26 returns shape (1,) shifts for 1D input and numpy 2 refuses to assign that to one element.
    Case(
        f"{PHASOR}:find_shifts_irf_data",
        lambda f: _error(f, _reference_data(), _flim_data()),
    ),
    Case(f"{PHASOR}:find_temporal_shifts", lambda f: _error(f, _flim_data())),
    Case(
        f"{PHASOR}:calibrate_phasor",
        lambda f: _error(f, _flim_data(), _reference_data()),
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(f, np.zeros((2, 2)), _fit_irf(), flim_time_axis(), PERIOD_NS),
        "data_ndim",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(
            f, _fit_data(), flim_decay()[:10], flim_time_axis(), PERIOD_NS
        ),
        "irf_shape",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(f, _fit_data(), _fit_irf(), flim_time_axis()[:10], PERIOD_NS),
        "t_shape",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(f, _fit_data(), _fit_irf(), np.full(64, np.nan), PERIOD_NS),
        "t_not_finite",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(f, _fit_data(), np.zeros(64), flim_time_axis(), PERIOD_NS),
        "irf_zero_sum",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(
            f,
            _fit_data(),
            _fit_irf(),
            flim_time_axis(),
            PERIOD_NS,
            valid_mask=np.ones(3),
        ),
        "mask_shape",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: f(
            _fit_data(),
            _fit_irf(),
            flim_time_axis(),
            PERIOD_NS,
            min_counts=1e9,
            show_progress=False,
        ),
        "no_valid_pixel",
    ),
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: f(
            _fit_data(),
            _fit_irf(),
            flim_time_axis(),
            PERIOD_NS,
            valid_mask=np.zeros((2, 2), bool),
            show_progress=False,
        ),
        "empty_mask",
    ),
    # brighteyes-mcs-dataprep 0.1.0 needs nine more keyword arguments in Alignment._fit_map_one_pixel than this code passes.
    Case(
        f"{FIT}:generate_fit_maps",
        lambda f: _error(
            f, _fit_data(), _fit_irf(), flim_time_axis(), PERIOD_NS, show_progress=False
        ),
        "pixel_fit",
    ),
]

SKIPPED: dict[str, str] = {
    f"{GRAPH}:load_calibration_fit_traces": SKIP_FILE_IO,
    f"{GRAPH}:load_calibration_summary": SKIP_FILE_IO,
    f"{GRAPH}:plot_calibration_fit_traces": SKIP_PLOT_ONLY,
    f"{GRAPH}:plot_calibration_lifetime_summary": SKIP_PLOT_ONLY,
    f"{GRAPH}:plot_calibration_shift_summary": SKIP_PLOT_ONLY,
    f"{GRAPH}:plot_channel_skew_correction": SKIP_PLOT_ONLY,
    f"{GRAPH}:plot_equalized_lifetime_summary": SKIP_PLOT_ONLY,
    f"{GRAPH}:plot_lifetime_histogram": SKIP_PLOT_ONLY,
    f"{GRAPH}:plot_lifetime_summary": SKIP_PLOT_ONLY,
    f"{PHASOR}:calculate_phasor_on_img_ch": SKIP_FILE_IO,
    f"{PHASOR}:calculate_phasor_on_img_pixels": SKIP_FILE_IO,
    f"{PHASOR}:phasor_h5": SKIP_FILE_IO,
    f"{PHASOR}:plot_flim_image": SKIP_PLOT_ONLY,
    f"{PHASOR}:plot_funny_single_phasor": SKIP_PLOT_ONLY,
    f"{PHASOR}:plot_phasor": SKIP_PLOT_ONLY,
    f"{PHASOR}:plot_tau": SKIP_PLOT_ONLY,
    f"{PHASOR}:plot_universal_circle": SKIP_PLOT_ONLY,
    f"{PHASOR}:show_flim_equalized": SKIP_PLOT_ONLY,
    f"{PHASOR}:show_lifetime_histogram": SKIP_PLOT_ONLY,
}
