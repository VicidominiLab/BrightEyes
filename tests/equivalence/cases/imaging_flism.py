"""Equivalence cases for `brighteyes_flism`."""

import numpy as np
import torch
from _inputs import decay_histogram, flim_reference_stack, flim_stack
from _registry import SKIP_FILE_IO, SKIP_PLOT_ONLY, Case
from matplotlib import pyplot as plt

PHASOR = "brighteyes_flism.tools_phasor"
GRAPH = "brighteyes_flism.graph_tools"
DATA = "brighteyes_flism.flism_data"
FIT = "brighteyes_flism._fit_maps"

T_NS = np.arange(256) * 0.1
PERIOD_NS = 25.6
SHORT_T_NS = np.arange(64) * 0.4


def _patch_vertices(patch):
    return patch.get_patch_transform().transform(patch.get_path().vertices)


def _drawn(call):
    plt.close("all")
    result = call()
    content = [
        (
            [line.get_xydata() for line in axes.lines],
            [image.get_array() for image in axes.images],
            [
                (
                    collection.get_array(),
                    collection.get_offsets(),
                    [path.vertices for path in collection.get_paths()],
                )
                for collection in axes.collections
            ],
            [(type(patch).__name__, _patch_vertices(patch)) for patch in axes.patches],
            [text.get_text() for text in axes.texts],
            [axes.get_title(), axes.get_xlabel(), axes.get_ylabel()],
            axes.get_xscale(),
            axes.get_yscale(),
        )
        for number in plt.get_fignums()
        for axes in plt.figure(number).axes
    ]
    plt.close("all")
    return result, content


def _rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _torch(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(_rng().uniform(size=shape))


def _lifetime_map() -> tuple[np.ndarray, np.ndarray]:
    rng = _rng()
    intensity = rng.uniform(0, 100, size=(16, 16))
    lifetime = rng.uniform(1.0, 4.0, size=(16, 16))
    lifetime[0, 0] = np.nan
    return intensity, lifetime


def _phasor_map() -> np.ndarray:
    summed = flim_stack().sum(-1)
    return np.fft.fft(summed, axis=-1)[..., 1].conj() / summed.sum(-1)


def _calibration_table() -> dict[str, np.ndarray]:
    rng = _rng()
    return {
        "channel": np.arange(25),
        "channel_skew": rng.normal(0, 0.5, size=25),
        "channel_skew_err": np.abs(rng.normal(0.05, 0.01, size=25)),
        "delay_correction_ns": rng.normal(0, 0.2, size=25),
        "delay_correction_err_ns": np.abs(rng.normal(0.02, 0.005, size=25)),
        "tau_ns": 2.5 + rng.normal(0, 0.05, size=25),
        "tau_err_ns": np.abs(rng.normal(0.03, 0.01, size=25)),
        "tau_reference_ns": np.full(25, 2.5),
        "residual_error": np.abs(rng.normal(0.1, 0.03, size=25)),
    }


def _second_calibration_table() -> dict[str, np.ndarray]:
    table = _calibration_table()
    return {name: values * 1.5 for name, values in table.items()} | {"channel": table["channel"]}


def _fit_traces() -> dict[str, np.ndarray | None]:
    rng = _rng()
    return {
        "measured_trace": rng.uniform(1, 100, size=64),
        "reference_trace": rng.uniform(1, 100, size=64),
        "fitted_trace": rng.uniform(1, 100, size=64),
        "skipped_trace": None,
        "unstyled_trace": rng.uniform(1, 100, size=64),
    }


def _equalizer() -> tuple[np.ndarray, np.ndarray]:
    return np.linspace(1.0, 5.0, 20), np.linspace(0.0, 1.0, 20) ** 2


def _reference_lifetime() -> np.ndarray:
    return _rng().gamma(4.0, 0.5, size=2000) + 1.0


def _cmap2d(f, invert):
    intensity, lifetime = _lifetime_map()
    parameters = {
        "minTau": 1.5,
        "maxTau": 3.5,
        "satFactor": 0.8,
        "invertColormap": invert,
        "outOfBoundsHue": 0.9,
        "minInt": 0.0,
        "maxInt": 100.0,
    }
    return f(intensity, lifetime, parameters)


def _six_dimensional() -> np.ndarray:
    return _rng().poisson(5.0, size=(2, 2, 8, 8, 4, 3)).astype(np.float64)


def _decays(*leading: int) -> np.ndarray:
    return np.broadcast_to(flim_stack().sum(-1)[:4, :4, :16], (*leading, 4, 4, 16)).copy()


def _shifts() -> np.ndarray:
    return np.linspace(-2.0, 3.0, 25)


def _flim_data(cls, **options):
    data = cls(freq_exc=40e6, **options)
    data.data_hist = flim_stack().sum((0, 1))
    data.data_hist_irf = flim_reference_stack().sum((0, 1))
    data.data_laser_hist = flim_stack().sum((0, 1, 3))
    data.data_laser_hist_irf = np.roll(flim_reference_stack().sum((0, 1, 3)), 3)
    return (
        data.calculate_phasor_global(),
        data.calculate_phasor_global_irf(),
        data.calculate_phasor_laser(),
        data.calculate_phasor_laser_irf(),
        data.calculate_correction_coeff(dfd_freq=40e6, tau=2e-9),
        data.phasor_global_corrected(),
        data.phasor_global_corrected(coeff=0.5 + 0.25j, laser_correction=False),
        data.phasor_global_corrected_irf(),
        data.phasor_global_corrected_irf(coeff=0.5 + 0.25j, laser_correction=False),
        data,
    )


CASES: list[Case] = [
    Case(
        f"{PHASOR}:correct_phasor",
        lambda f: f(_phasor_map(), _phasor_map()[0, 0] * 1j, coeff=0.5),
        "laser correction",
    ),
    Case(
        f"{PHASOR}:correct_phasor",
        lambda f: f(_phasor_map(), _phasor_map()[0, 0] * 1j, coeff=0.5, laser_correction=False),
        "no laser correction",
    ),
    Case(f"{PHASOR}:sum_adjacent_pixel", lambda f: f(_rng().poisson(5.0, size=(16, 16)).astype(float), 4), "2D"),
    Case(f"{PHASOR}:sum_adjacent_pixel", lambda f: f(_six_dimensional(), 2), "6D"),
    Case(f"{PHASOR}:sum_adjacent_pixel", lambda f: f(_six_dimensional(), 1), "no summing"),
    Case(f"{PHASOR}:periodic_single_exponential_model", lambda f: f(T_NS, 3.0, 2.5, PERIOD_NS)),
    Case(f"{PHASOR}:pad_tensor", lambda f: f(_torch((6, 5)), 2, 3, 0), "reflect"),
    Case(f"{PHASOR}:pad_tensor", lambda f: f(_torch((6, 5)), 2, 3, 1, mode="replicate"), "replicate"),
    Case(f"{PHASOR}:pad_tensor", lambda f: f(_torch((6, 5)), 1, 2, 0, mode="constant"), "constant"),
    Case(f"{PHASOR}:torch_median_filter", lambda f: f(_torch((8, 9))), "all dimensions"),
    Case(
        f"{PHASOR}:torch_median_filter",
        lambda f: f(_torch((8, 9)), window_size=5, dims=[1], mode="replicate"),
        "one dimension",
    ),
    Case(f"{PHASOR}:partial_convolution_fft", lambda f: f(_torch((4, 6, 5)), _torch((6, 5, 3))), "spatial inputs"),
    Case(
        f"{PHASOR}:partial_convolution_fft",
        lambda f: f(torch.fft.fftn(_torch((4, 6, 5)), dim=[1, 2]), _torch((6, 5, 3)), fourier=(True, False)),
        "Fourier volume",
    ),
    Case(
        f"{PHASOR}:partial_convolution_fft",
        lambda f: f(_torch((4, 6, 5)), torch.fft.fftn(_torch((6, 5, 3)), dim=[0, 1]), fourier=(False, True)),
        "Fourier kernel",
    ),
    Case(
        f"{PHASOR}:richardson_lucy_deconvolution",
        lambda f: f(decay_histogram()[:64], SHORT_T_NS, 1.0, 2.5, PERIOD_NS, iterations=5),
        "regularized",
    ),
    Case(
        f"{PHASOR}:richardson_lucy_deconvolution",
        lambda f: f(
            torch.from_numpy(decay_histogram()[:64]),
            SHORT_T_NS,
            1.0,
            2.5,
            PERIOD_NS,
            iterations=5,
            regularization=1,
        ),
        "tensor input, not regularized",
    ),
    Case(
        f"{PHASOR}:estimate_irf",
        lambda f: f(decay_histogram()[:64], SHORT_T_NS, 1.0, 2.5, PERIOD_NS, iterations=5),
    ),
    Case(f"{PHASOR}:estimate_lifetime_from_birfi", lambda f: f(T_NS, decay_histogram()), "one decay"),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        lambda f: f(
            SHORT_T_NS,
            flim_stack().sum((0, 1)),
            window_length=7,
            polyorder=2,
            persistence=3,
            threshold=0.1,
            return_bounds=True,
        ),
        "channels with bounds",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_birfi",
        lambda f: f(SHORT_T_NS, flim_stack().sum((0, 1)).T, axis=1, return_bounds=True),
        "time on the last axis",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(decay_histogram(), T_NS, 0.1, 256, PERIOD_NS),
        "default levels",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(decay_histogram(), T_NS, 0.1, 256, PERIOD_NS, start_level=0.9, end_level=0.2),
        "other levels",
    ),
    Case(
        f"{PHASOR}:estimate_lifetime_from_log",
        lambda f: f(np.roll(decay_histogram(), 7) + 3.0, T_NS, 0.1, 256, PERIOD_NS),
        "peak away from the first bin",
    ),
    Case(f"{PHASOR}:estimate_lifetime_from_circmean", lambda f: f(decay_histogram(), 0.5), "default"),
    Case(
        f"{PHASOR}:estimate_lifetime_from_circmean",
        lambda f: f(decay_histogram(), 0.5, repetition_rate_MHz=20.0, background_per_bin=5.0, truncate_ns=30.0),
        "background and truncation",
    ),
    Case(f"{PHASOR}:flatten_and_remove_nan", lambda f: f(_lifetime_map()[1]), "one array"),
    Case(
        f"{PHASOR}:flatten_and_remove_nan",
        lambda f: f(_lifetime_map()[1], _lifetime_map()[0] * np.where(_lifetime_map()[0] > 90, np.nan, 1.0)),
        "two arrays",
    ),
    Case(f"{PHASOR}:g0s0_to_m_phi", lambda f: f(*_lifetime_map())),
    Case(f"{PHASOR}:m_phi_to_g0s0", lambda f: f(*_lifetime_map())),
    Case(f"{PHASOR}:m_phi_to_complex", lambda f: f(*_lifetime_map())),
    Case(f"{PHASOR}:phasor", lambda f: f(flim_stack().sum(-1)), "last axis"),
    Case(
        f"{PHASOR}:phasor",
        lambda f: f(flim_stack().sum(-1).transpose(2, 0, 1), threshold=300, harmonic=2, time_axis=0),
        "first axis, threshold and harmonic",
    ),
    Case(f"{PHASOR}:calculate_phasor", lambda f: f(decay_histogram()), "1D"),
    Case(f"{PHASOR}:calculate_phasor", lambda f: f(decay_histogram(), threshold=1e9), "1D below threshold"),
    Case(f"{PHASOR}:calculate_phasor", lambda f: f(flim_stack().sum((0, 1)), harmonic=2), "2D"),
    Case(f"{PHASOR}:calculate_phasor", lambda f: f(flim_stack().sum((0, 1)), threshold=9400), "2D, threshold"),
    Case(f"{PHASOR}:calculate_phasor", lambda f: f(flim_stack().sum(-1)), "3D"),
    Case(
        f"{PHASOR}:calculate_phasor",
        lambda f: f(flim_stack().sum(-1), threshold=1300, harmonic=2),
        "3D, threshold and harmonic",
    ),
    Case(f"{PHASOR}:calculate_tau_phi", lambda f: f(_phasor_map()), "complex"),
    Case(
        f"{PHASOR}:calculate_tau_phi",
        lambda f: f(_phasor_map().real, _phasor_map().imag * 0.8, dfd_freq=40e6),
        "real and imaginary parts",
    ),
    Case(f"{PHASOR}:calculate_tau_m", lambda f: f(_phasor_map()), "complex"),
    Case(
        f"{PHASOR}:calculate_tau_m",
        lambda f: f(_phasor_map().real, _phasor_map().imag * 0.8, dfd_freq=40e6),
        "real and imaginary parts",
    ),
    Case(f"{PHASOR}:calculate_m_phi_tau_phi_tau_m", lambda f: f(_phasor_map()), "complex"),
    Case(
        f"{PHASOR}:calculate_m_phi_tau_phi_tau_m",
        lambda f: f(_phasor_map().real, _phasor_map().imag * 0.8, dfd_freq=40e6),
        "real and imaginary parts",
    ),
    # The old code writes complex values into a float array, so only the real part is compared.
    Case(f"{PHASOR}:calculate_phasor_on_img", lambda f: f(_decays(2)), "4D"),
    Case(f"{PHASOR}:calculate_phasor_on_img", lambda f: f(_decays(1, 2), 2, 2), "5D"),
    Case(f"{PHASOR}:fourier_shift", lambda f: f(flim_stack().sum((0, 1)), 0.1)),
    Case(f"{PHASOR}:cmap2d", lambda f: _cmap2d(f, False), "not inverted"),
    Case(f"{PHASOR}:cmap2d", lambda f: _cmap2d(f, True), "inverted"),
    Case(f"{PHASOR}:linear_shift", lambda f: f(decay_histogram(), 5.5), "cyclic"),
    Case(f"{PHASOR}:linear_shift", lambda f: f(decay_histogram(), 5.5, cyclic=False), "not cyclic"),
    Case(
        f"{PHASOR}:correction_phasor",
        lambda f: f(decay_histogram(), np.roll(decay_histogram(), 7) + 3.0),
    ),
    Case(f"{PHASOR}:threshold_intensity", lambda f: f(_lifetime_map()[0], 0.4)),
    Case(f"{PHASOR}:threshold_phasor", lambda f: f(_phasor_map().real * 100, _phasor_map(), 0.7)),
    Case(f"{PHASOR}:median_phasors", lambda f: f(_phasor_map().real), "real map"),
    Case(f"{PHASOR}:median_phasors", lambda f: f(_phasor_map().imag, window=5), "real map, wider window"),
    Case(f"{PHASOR}:calculate_irf_correction", lambda f: f(decay_histogram(), 256)),
    Case(f"{PHASOR}:align_decays", lambda f: f(flim_stack(), _shifts())),
    Case(f"{PHASOR}:align_image", lambda f: f(flim_stack(), _shifts())),
    Case(
        f"{PHASOR}:plot_phasor",
        lambda f: _drawn(lambda: f(_phasor_map(), bins_2dplot=20)),
        "logarithmic",
    ),
    Case(
        f"{PHASOR}:plot_phasor",
        lambda f: _drawn(
            lambda: f(
                _phasor_map(),
                bins_2dplot=20,
                log_scale=False,
                draw_universal_circle=False,
                tau_labels=False,
                quadrant="first",
                dfd_freq=40e6,
                cmap="magma",
            )
        ),
        "linear, first quadrant, no circle",
    ),
    Case(
        f"{PHASOR}:plot_flim_image",
        lambda f: _drawn(lambda: f(flim_stack(), _phasor_map(), pxsize=0.05, pxdwelltime=50, log_scale=True)),
        "tau_phi",
    ),
    Case(
        f"{PHASOR}:plot_flim_image",
        lambda f: _drawn(lambda: f(flim_stack(), _phasor_map(), method="tau_m", lifetime_bounds=(0.5, 3))),
        "tau_m",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(_reference_lifetime(), (1.0, 5.0)),
        "default",
    ),
    Case(
        f"{PHASOR}:build_lifetime_equalizer",
        lambda f: f(
            _reference_lifetime(),
            (1.5, 4.0),
            weights=np.linspace(0.1, 2, 2000),
            strength=2.0,
            equalization_bins=64,
        ),
        "weighted, other strength and bins",
    ),
    Case(f"{PHASOR}:build_lifetime_equalizer", lambda f: f(_reference_lifetime(), (3.0, 3.0)), "empty range"),
    Case(f"{PHASOR}:build_lifetime_equalizer", lambda f: f(_reference_lifetime(), (1.0, 5.0), strength=0.0), "no strength"),
    Case(f"{PHASOR}:build_lifetime_equalizer", lambda f: f(_reference_lifetime(), (20.0, 30.0)), "no reference in range"),
    Case(
        f"{PHASOR}:apply_lifetime_equalizer",
        lambda f: f(_lifetime_map()[1] * 1.5, *_equalizer()),
    ),
    Case(
        f"{PHASOR}:equalized_lifetime_tick_values",
        lambda f: f(_reference_lifetime(), (1.0, 5.0)),
        "default",
    ),
    Case(
        f"{PHASOR}:equalized_lifetime_tick_values",
        lambda f: f(
            _reference_lifetime(),
            (1.5, 4.0),
            weights=np.linspace(0.1, 2, 2000),
            n_ticks=4,
            strength=2.0,
            equalization_bins=64,
        ),
        "weighted, other ticks, strength and bins",
    ),
    Case(
        f"{PHASOR}:show_flim_equalized",
        lambda f: _drawn(lambda: f(*_lifetime_map(), 0.05, 50, n_quantiles=64)),
        "default",
    ),
    Case(
        f"{PHASOR}:show_flim_equalized",
        lambda f: _drawn(
            lambda: f(
                *_lifetime_map(),
                0.05,
                50,
                lifetime_bounds=(1.5, 3.5),
                intensity_bounds=(10, 80),
                colormap="viridis",
                equalization_reference=_reference_lifetime(),
                equalization_weights=np.linspace(0.1, 2, 2000),
                colorbar_ticks=4,
                equalization_strength=2.0,
                equalization_bins=64,
            )
        ),
        "bounds, reference and weights",
    ),
    Case(f"{GRAPH}:crop_2d", lambda f: f(_lifetime_map()[0], 3), "cropped"),
    Case(f"{GRAPH}:crop_2d", lambda f: f(_lifetime_map()[0]), "not cropped"),
    Case(f"{GRAPH}:normalize_histogram", lambda f: f(decay_histogram()), "positive sum"),
    Case(f"{GRAPH}:normalize_histogram", lambda f: f(np.zeros(8)), "zero sum"),
    Case(f"{GRAPH}:threshold_lifetime_map", lambda f: f(*_lifetime_map()[::-1], threshold=0.3), "intensity"),
    Case(f"{GRAPH}:threshold_lifetime_map", lambda f: f(_lifetime_map()[1]), "no intensity"),
    Case(
        f"{GRAPH}:threshold_lifetime_map",
        lambda f: f(*_lifetime_map()[::-1], threshold=0.3, finite_only=False),
        "not only finite",
    ),
    Case(f"{GRAPH}:weighted_lifetime_stats", lambda f: f(_lifetime_map()[1]), "unweighted"),
    Case(f"{GRAPH}:weighted_lifetime_stats", lambda f: f(*_lifetime_map()[::-1]), "weighted"),
    Case(
        f"{GRAPH}:plot_calibration_lifetime_summary",
        lambda f: _drawn(lambda: f(_calibration_table())),
        "histogram on the lifetime axis y",
    ),
    Case(
        f"{GRAPH}:plot_calibration_lifetime_summary",
        lambda f: _drawn(lambda: f(_calibration_table(), histogram_lifetime_axis="x")),
        "histogram on the lifetime axis x",
    ),
    Case(
        f"{GRAPH}:plot_calibration_shift_summary",
        lambda f: _drawn(lambda: f(_calibration_table())),
        "one table",
    ),
    Case(
        f"{GRAPH}:plot_calibration_shift_summary",
        lambda f: _drawn(
            lambda: f(
                [_calibration_table(), _second_calibration_table()],
                labels=["first", "second"],
                reference_channel=12,
            )
        ),
        "two tables with labels and reference channel",
    ),
    Case(
        f"{GRAPH}:plot_calibration_fit_traces",
        lambda f: _drawn(lambda: f(SHORT_T_NS, _fit_traces(), title="channel 3")),
        "logarithmic",
    ),
    Case(
        f"{GRAPH}:plot_calibration_fit_traces",
        lambda f: _drawn(lambda: f(SHORT_T_NS, _fit_traces(), log_scale=False)),
        "linear",
    ),
    Case(
        f"{GRAPH}:plot_channel_skew_correction",
        lambda f: _drawn(
            lambda: f(
                decay_histogram(),
                np.roll(decay_histogram(), 3),
                decay_histogram() + 1,
                np.roll(decay_histogram(), 5) + 1,
            )
        ),
        "without reversed traces",
    ),
    Case(
        f"{GRAPH}:plot_channel_skew_correction",
        lambda f: _drawn(
            lambda: f(
                decay_histogram(),
                np.roll(decay_histogram(), 3),
                decay_histogram() + 1,
                np.roll(decay_histogram(), 5) + 1,
                irf_reversed=np.roll(decay_histogram(), -3),
                data_reversed=np.roll(decay_histogram(), -5) + 1,
            )
        ),
        "with reversed traces",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        lambda f: _drawn(lambda: f(_lifetime_map()[1], bins=50)),
        "lifetime on x",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        lambda f: _drawn(
            lambda: f(
                _lifetime_map()[1],
                weights=_lifetime_map()[0],
                lifetime_bounds=(1.5, 3.5),
                bins=30,
                lifetime_axis="y",
                label="data",
                count_label="Counts",
            )
        ),
        "lifetime on y, weighted, with bounds",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_histogram",
        lambda f: _drawn(lambda: f(_lifetime_map()[1], bins=50, gaussian=False)),
        "without Gaussian",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_summary",
        lambda f: _drawn(lambda: f(*_lifetime_map(), 0.05, 50, bins=50)),
        "weighted histogram",
    ),
    Case(
        f"{GRAPH}:plot_lifetime_summary",
        lambda f: _drawn(
            lambda: f(
                *_lifetime_map(),
                0.05,
                50,
                lifetime_bounds=(1.5, 3.5),
                crop=2,
                threshold=0.2,
                bins=30,
                colormap="viridis",
                weighted_histogram=False,
            )
        ),
        "cropped, unweighted histogram",
    ),
    Case(
        f"{GRAPH}:plot_equalized_lifetime_summary",
        lambda f: _drawn(lambda: f(*_lifetime_map(), 0.05, 50, bins=50, equalization_bins=64)),
        "default",
    ),
    Case(
        f"{GRAPH}:plot_equalized_lifetime_summary",
        lambda f: _drawn(
            lambda: f(
                *_lifetime_map(),
                0.05,
                50,
                lifetime_bounds=(1.5, 3.5),
                crop=2,
                threshold=0.2,
                bins=30,
                colormap="viridis",
                equalization_reference=_reference_lifetime(),
                equalization_strength=2.0,
                equalization_bins=64,
                colorbar_ticks=4,
            )
        ),
        "cropped, with reference",
    ),
    Case(f"{DATA}:FlimData", lambda f: _flim_data(f), "no correction coefficient"),
    Case(f"{DATA}:FlimData", lambda f: _flim_data(f, correction_coeff=[0.5, 0.25]), "coefficient from a list"),
    Case(f"{DATA}:FlimData", lambda f: _flim_data(f, correction_coeff=0.5 + 0.25j), "complex coefficient"),
]

_SHIFT_ERROR = "fails with the old code too: ValueError: setting an array element with a sequence."

SKIPPED: dict[str, str] = {
    f"{PHASOR}:calculate_phasor_on_img_ch": SKIP_FILE_IO,
    f"{PHASOR}:calculate_phasor_on_img_pixels": SKIP_FILE_IO,
    f"{PHASOR}:phasor_h5": SKIP_FILE_IO,
    f"{PHASOR}:find_shifts_irf_data": _SHIFT_ERROR,
    f"{PHASOR}:find_temporal_shifts": _SHIFT_ERROR,
    f"{PHASOR}:calibrate_phasor": _SHIFT_ERROR,
    f"{PHASOR}:plot_universal_circle": SKIP_PLOT_ONLY,
    f"{PHASOR}:plot_tau": SKIP_PLOT_ONLY,
    f"{PHASOR}:plot_funny_single_phasor": SKIP_PLOT_ONLY,
    f"{PHASOR}:show_lifetime_histogram": SKIP_PLOT_ONLY,
    f"{GRAPH}:load_calibration_summary": SKIP_FILE_IO,
    f"{GRAPH}:load_calibration_fit_traces": SKIP_FILE_IO,
    f"{FIT}:generate_fit_maps": (
        "fails with the old code too: TypeError: Alignment._fit_map_one_pixel() missing 9 required positional arguments"
    ),
}
