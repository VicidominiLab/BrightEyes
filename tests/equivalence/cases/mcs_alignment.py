"""Equivalence cases for `brighteyes_mcs_dataprep.alignment`."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from _capture import error, warned
from _inputs import periodic_decay
from _registry import Case

ALIGN = "brighteyes_mcs_dataprep.alignment:Alignment"
PERIOD = 25.0
BINS = 64
TIME = np.arange(BINS) * PERIOD / BINS


def _data(**kwargs: Any) -> np.ndarray:
    return periodic_decay(**kwargs)[1]


def _irf(center: float = 3.0, width: float = 0.8) -> np.ndarray:
    return np.exp(-((np.arange(BINS) - center) ** 2) / (2 * width**2)) + 1e-4


def _on(*arguments: Callable[[], Any], **kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: warned(lambda: f(*(make() for make in arguments), **kwargs))


def _fails(*arguments: Callable[[], Any], **kwargs: Any) -> Callable[[Any], Any]:
    return lambda f: error(lambda: f(*(make() for make in arguments), **kwargs))


def _t() -> np.ndarray:
    return TIME.copy()


def _decay_data() -> np.ndarray:
    return _data(tau_ns=2.5, delay_ns=6.0, counts=4000.0)


def _short_data() -> np.ndarray:
    return _data(tau_ns=3.0, delay_ns=4.0, counts=3000.0, seed=3)


def _reference() -> np.ndarray:
    return _data(tau_ns=1.2, delay_ns=3.0, counts=20000.0, seed=1)


def _irf_copy() -> np.ndarray:
    return _irf()


def _wave(x: np.ndarray, amplitude: float, phase: float) -> np.ndarray:
    return amplitude * np.cos(x - phase)


def _circular_fit(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        x = np.linspace(0, 4 * np.pi, 80)
        noise = np.random.default_rng(0).normal(0, 0.05, size=x.shape)
        y = _wave(x, 1.5, 2.8) + noise
        return f(_wave, x, y, **kwargs)

    return call


def _circular_fit_error(f: Callable[..., Any]) -> str:
    return error(lambda: _circular_fit()(f))


def _custom_model(
    t: np.ndarray, irf: np.ndarray, period: float, C: float, dT: float, tau: float
) -> np.ndarray:
    decay = np.exp(-np.mod(t, period) / tau)
    decay = decay / decay.sum()
    phase = np.exp(-2j * np.pi * np.fft.fftfreq(len(t)) * dT)
    shifted = np.fft.ifft(np.fft.fft(irf / irf.sum()) * phase).real
    return C * np.fft.ifftshift(
        np.fft.ifft(np.fft.fft(decay) * np.fft.fft(shifted)).real
    )


def _scaled_model(
    t: np.ndarray,
    irf: np.ndarray,
    period: float,
    C: float,
    tau: float,
    scale: float = 1.0,
) -> np.ndarray:
    decay = np.exp(-np.mod(t, period) / tau)
    return C * scale * decay / decay.sum()


def _fit_data(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = {
            "period": PERIOD,
            "ref": _reference(),
            "tau_ref": 1.2,
            "irf_iterations": 8,
            **kwargs,
        }
        if options.pop("bright_reference", False):
            options["ref"] = _data(tau_ns=3.0, delay_ns=10.0, counts=2e5, seed=1)
        if options.get("irf") is not None:
            options.pop("ref")
            options.pop("tau_ref")
        return warned(lambda: f(_t(), _decay_data(), **options))

    return call


def _fit_data_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = {
            "period": PERIOD,
            "ref": _reference(),
            "tau_ref": 1.2,
            "irf_iterations": 3,
            **kwargs,
        }
        data = options.pop("data", _decay_data())
        time = options.pop("t", _t())
        return error(lambda: f(time, data, **options))

    return call


def _perform_fit(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        data = options.pop("data", _decay_data())
        irf = options.pop("irf", _irf())
        return warned(lambda: f(_t(), data, irf, PERIOD, **options))

    return call


def _perform_fit_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        data = options.pop("data", _decay_data())
        irf = options.pop("irf", _irf())
        time = options.pop("t", _t())
        return error(lambda: f(time, data, irf, PERIOD, **options))

    return call


def _image(**kwargs: Any) -> np.ndarray:
    return np.stack(
        [
            [
                _data(
                    tau_ns=2.0 + 0.4 * x, delay_ns=5.0 + y, counts=800.0, seed=3 * y + x
                )
                for x in range(3)
            ]
            for y in range(2)
        ]
    )


def _maps(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        data = options.pop("data", _image())
        irf = options.pop("irf", _irf())
        time = options.pop("t", _t())
        return warned(
            lambda: f(data, irf, time, PERIOD, show_progress=False, **options)
        )

    return call


def _maps_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        options = dict(kwargs)
        data = options.pop("data", _image())
        irf = options.pop("irf", _irf())
        time = options.pop("t", _t())
        return error(lambda: f(data, irf, time, PERIOD, show_progress=False, **options))

    return call


def _fit_maps() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    return {
        "parameter_names": ["C", "dT", "tau"],
        "C": rng.uniform(size=(2, 3)),
        "dT": rng.uniform(size=(2, 3)),
        "tau": rng.uniform(size=(2, 3)),
        "C_err": rng.uniform(size=(2, 3)),
        "dT_err": rng.uniform(size=(2, 3)),
        "tau_err": rng.uniform(size=(2, 3)),
    }


def _stack(f: Callable[..., Any]) -> Any:
    maps = _fit_maps()
    renamed = {key: value for key, value in maps.items() if key != "parameter_names"}
    renamed["param_names"] = ["C", "tau"]
    no_names = {
        key: value
        for key, value in maps.items()
        if key not in ("parameter_names", "C_err")
    }
    return [
        f(maps),
        f(maps, names=["tau", "C"]),
        f(renamed),
        f(no_names, names=("C", "dT", "tau", "dT_err", "tau_err")),
        error(lambda: f(maps, names=["missing"])),
    ]


def _shifted_sum(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        rng = np.random.default_rng(0)
        data = rng.poisson(5.0, size=(1, 2, 3, 4, 16, 5)).astype(float)
        shifts = np.array([0.0, 1.5, -2.25, 3.0, 0.4])
        return f(data, shifts, show_progress=False, **kwargs)

    return call


def _shifted_sum_error(**kwargs: Any) -> Callable[[Any], Any]:
    def call(f: Callable[..., Any]) -> Any:
        data = np.ones((1, 2, 3, 4, 16, 5))
        options = dict(kwargs)
        shifts = options.pop("shifts", np.zeros(5))
        return error(lambda: f(data, shifts, show_progress=False, **options))

    return call


def _tensor(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(seed).uniform(size=shape))


def _pad(mode: str, left: int, right: int, dim: int) -> Callable[[Any], Any]:
    return lambda f: f(_tensor((4, 6)), left, right, dim, mode=mode)


CASES: list[Case] = [
    Case(
        f"{ALIGN}.to_numpy_1d",
        lambda f: [
            f([1, 2, 3]),
            f(torch.arange(4.0), dtype=np.float32),
            f(np.arange(3), dtype=float),
            f(np.arange(3.0)),
        ],
    ),
    Case(
        f"{ALIGN}.to_torch_1d",
        lambda f: [
            f([1, 2, 3], dtype=torch.float64),
            f(torch.arange(4), dtype=torch.float32, device="cpu"),
            f(torch.arange(4.0)),
            f(np.arange(3.0)),
        ],
    ),
    Case(
        f"{ALIGN}.centroid",
        lambda f: [f(_irf()), f(np.zeros(8)), f(torch.arange(5.0)), f([0, 0, 1, 0])],
    ),
    Case(
        f"{ALIGN}.clean_irf",
        lambda f: [
            f(_irf() / _irf().max()),
            f(_irf(), threshold=0.5, window=3),
            f(np.zeros(16)),
            f(_irf() / _irf().max(), threshold=0.9, window=1.5),
        ],
    ),
    Case(
        f"{ALIGN}.clean_irf_stack",
        lambda f: [
            f(np.stack([_irf(3.0), _irf(20.0, 1.5), _irf(40.0)], axis=1)),
            f(
                np.stack([_irf(3.0) * 4, np.zeros(BINS), _irf(40.0) * 2]),
                threshold=0.4,
                window=4,
                time_axis=1,
                normalize=True,
            ),
        ],
    ),
    Case(f"{ALIGN}.curve_fit_circular", _circular_fit(p0=[1.0, 2.0]), "plain"),
    Case(
        f"{ALIGN}.curve_fit_circular",
        _circular_fit(
            p0=[1.0, 6.0],
            bounds=([0.0, 0.0], [5.0, 2 * np.pi]),
            circular_params={1: 2 * np.pi},
        ),
        "circular parameter",
    ),
    Case(
        f"{ALIGN}.curve_fit_circular",
        _circular_fit(
            p0=[1.0, 0.5],
            bounds=([0.0, -np.pi], [5.0, np.pi]),
            circular_params=[1],
            sigma=np.full(80, 0.05),
            circular_curve_period=2 * np.pi,
            circular_curve_center=0.3,
            method="trf",
            maxfev=200,
        ),
        "inferred period and sigma",
    ),
    Case(
        f"{ALIGN}.curve_fit_circular",
        _circular_fit(
            p0=[1.0, 2.0],
            sigma=np.full(80, 0.05),
            absolute_sigma=True,
            circular_params=[(1, 2 * np.pi)],
            max_nfev=100,
        ),
        "absolute sigma",
    ),
    Case(f"{ALIGN}.curve_fit_circular", _circular_fit_error, "missing p0"),
    Case(
        f"{ALIGN}.model_data",
        lambda f: [
            f(_t(), 1.0, 2.5, PERIOD),
            f(_t(), 2.0, 1.0, PERIOD, shift_bins=3.5, mode="binned"),
            f(_t(), 1.0, 2.5, PERIOD, shift_bins=-2.0, mode="sampled"),
            f(_t(), 0.5, 4.0, PERIOD, mode="SAMPLED"),
        ],
    ),
    Case(
        f"{ALIGN}.model_data",
        _fails(_t, lambda: 1.0, lambda: 2.5, lambda: PERIOD, mode="x"),
        "invalid mode",
    ),
    Case(
        f"{ALIGN}.model_data",
        _fails(
            lambda: np.array([0.0, 1.0, 3.0]), lambda: 1.0, lambda: 2.5, lambda: PERIOD
        ),
        "non-uniform axis",
    ),
    Case(
        f"{ALIGN}.model_data",
        _fails(lambda: np.array([0.0]), lambda: 1.0, lambda: 2.5, lambda: PERIOD),
        "single sample",
    ),
    Case(
        f"{ALIGN}.rectangular_IRF",
        lambda f: [
            f(_t(), 0.5),
            f(np.linspace(0, 10, 11), 1.0),
            f(torch.arange(8.0), 2),
        ],
    ),
    Case(f"{ALIGN}.pad_tensor", _pad("reflect", 2, 3, 1), "reflect"),
    Case(f"{ALIGN}.pad_tensor", _pad("replicate", 3, 1, 0), "replicate"),
    Case(f"{ALIGN}.pad_tensor", _pad("constant", 1, 2, 1), "constant"),
    Case(f"{ALIGN}.pad_tensor", _pad("reflect", 0, 0, 1), "no padding"),
    Case(
        f"{ALIGN}.pad_tensor",
        lambda f: error(lambda: f(_tensor((4, 6)), 1, 1, 0, mode="x")),
        "invalid mode",
    ),
    Case(f"{ALIGN}.median_filter", lambda f: f(_tensor((5, 7))), "default"),
    Case(
        f"{ALIGN}.median_filter",
        lambda f: f(_tensor((5, 7)), window_size=[3, 5], dims=[1, 0], mode="replicate"),
        "per-axis window",
    ),
    Case(
        f"{ALIGN}.median_filter",
        lambda f: f(_tensor((4, 9)), window_size=5, dims=[1], mode="constant"),
        "constant padding",
    ),
    Case(
        f"{ALIGN}.median_filter",
        lambda f: [
            error(lambda: f(_tensor((4, 4)), window_size=2)),
            error(lambda: f(_tensor((4, 4)), window_size=[3], dims=[0, 1])),
        ],
        "invalid windows",
    ),
    Case(
        f"{ALIGN}.partial_convolution_fft",
        lambda f: f(_tensor((3, 4, 5)), _tensor((4, 5, 6), 1)),
        "default",
    ),
    Case(
        f"{ALIGN}.partial_convolution_fft",
        lambda f: f(
            _tensor((8,)),
            torch.fft.fftn(_tensor((8,), 1), dim=0),
            dim1="t",
            dim2="t",
            axis="t",
            fourier=(False, True),
        ),
        "kernel given in Fourier space",
    ),
    Case(
        f"{ALIGN}.partial_convolution_fft",
        lambda f: f(
            torch.fft.fftn(_tensor((3, 4)), dim=[0]),
            _tensor((4, 5), 1),
            dim1="ij",
            dim2="jk",
            axis="i",
            fourier=(True, False),
        ),
        "volume given in Fourier space",
    ),
    Case(
        f"{ALIGN}.IRF_from_data_deconvolution",
        lambda f: f(
            _data(counts=20000.0, seed=1) / 20000.0,
            _t(),
            1.0,
            1.2,
            PERIOD,
            iterations=6,
        ),
        "median filter",
    ),
    Case(
        f"{ALIGN}.IRF_from_data_deconvolution",
        lambda f: f(
            _reference() / _reference().sum(),
            _t(),
            2.0,
            1.5,
            PERIOD,
            iterations=4,
            eps=1e-6,
            regularization=0,
        ),
        "no regularization",
    ),
    Case(
        f"{ALIGN}.linear_shift",
        lambda f: [
            f(_irf(), 2.5),
            f(_irf(), -3.25),
            f(_irf(), 70.5),
            f(_irf(), 2.5, cyclic=False),
            f(_irf(), -4.0, cyclic=False),
        ],
    ),
    Case(f"{ALIGN}.fit_data_with_ref_or_irf", _fit_data(), "reference with lifetime"),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(irf=_irf(), irf_output="shifted", shift_output="data"),
        "irf shifted and data shifted",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(tau_ref=None, shift_output="ref", mode="model_shift"),
        "log lifetime and model shift",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(tau_ref="circmean", shift_output="reference", irf_output="estimated"),
        "circmean lifetime",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(
            tau_ref="birfi", fit_type="curve_fit_circular", bright_reference=True
        ),
        "birfi lifetime with circular curve fit",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(tau_ref="1.1", fit_type="curve_fit", regularization=0),
        "numeric string lifetime with curve fit",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(
            irf=_irf(),
            force_C_normalized=True,
            initial_tau=2.0,
            initial_dT=1.0,
            initial_C=1.0,
        ),
        "fixed amplitude and initial guesses",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(
            irf=_irf(),
            model_fn=_custom_model,
            p0=[1.0, 0.0, 2.0],
            parameter_names=["C", "dT", "tau"],
            bounds=([0.0, -10.0, 0.1], [2.0, 10.0, 20.0]),
        ),
        "custom model",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data(irf=np.zeros(BINS)),
        "empty irf",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(tau_ref="birfi"),
        "lifetime not estimable by birfi",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(ref=None),
        "neither reference nor irf",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(irf=_irf()),
        "both reference and irf",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(irf_output="x"),
        "invalid irf output",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(shift_output="x"),
        "invalid shift output",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(tau_ref="nonsense"),
        "invalid lifetime selector",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(data=np.ones(10)),
        "length mismatch",
    ),
    Case(
        f"{ALIGN}.fit_data_with_ref_or_irf",
        _fit_data_error(ref=np.zeros(BINS), tau_ref=None),
        "lifetime not estimable",
    ),
    Case(
        f"{ALIGN}.fit_model_data",
        lambda f: [
            f(_t(), 1.0, 2.0, 2.5, _irf(), PERIOD),
            f(_t(), 0.8, -3.5, 1.5, _irf(10.0), PERIOD, mode="model_shift"),
        ],
        "modes",
    ),
    Case(
        f"{ALIGN}.fit_model_data",
        lambda f: [
            error(lambda: f(_t(), 1.0, 0.0, 2.5, _irf()[:10], PERIOD)),
            error(lambda: f(_t()[:1], 1.0, 0.0, 2.5, _irf()[:1], PERIOD)),
            error(lambda: f(_t(), 1.0, 0.0, 2.5, _irf(), PERIOD, mode="x")),
            error(lambda: f(np.arange(BINS) ** 2.0, 1.0, 0.0, 2.5, _irf(), PERIOD)),
            error(lambda: f(_t(), 1.0, 0.0, 2.5, np.zeros(BINS), PERIOD)),
        ],
        "errors",
    ),
    Case(
        f"{ALIGN}.estimate_peak_dT_bins",
        lambda f: [
            f(_data(delay_ns=10.0), _irf(3.0)),
            f(_irf(3.0), _irf(60.0)),
            f(_irf(60.0), _irf(3.0)),
            error(lambda: f(np.ones(4), np.ones(5))),
        ],
    ),
    Case(f"{ALIGN}.perform_fit_data", _perform_fit(), "default"),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit(
            mode="model_shift", initial_tau=2.0, initial_dT=2.0, initial_C=0.9
        ),
        "model shift with initial guesses",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit(fit_type="curve_fit_circular", force_C_normalized=True),
        "circular curve fit with fixed amplitude",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit(fit_type="curve_fit", mode="model_shift"),
        "curve fit",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit(
            p0=[0.9, 1.0, 2.0],
            bounds=([0.0, -5.0, 0.5], [2.0, 5.0, 10.0]),
            parameter_names=["C", "dT", "tau"],
            irf_min=0.2,
        ),
        "p0 and bounds",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit(
            model_fn=_scaled_model,
            p0=[1.0, 2.0, 1.0],
            param_names=["C", "tau", "scale"],
            model_kwargs={},
            amplitude_param="C",
            lifetime_param="tau",
            delay_param="shift",
            force_C_normalized=True,
        ),
        "custom model fixed amplitude",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit(
            model_fn=_custom_model,
            p0=[1.0, 0.5, 2.0],
            fit_type="curve_fit_circular",
        ),
        "custom model circular fit",
    ),
    Case(f"{ALIGN}.perform_fit_data", _perform_fit(data=np.zeros(BINS)), "empty data"),
    Case(f"{ALIGN}.perform_fit_data", _perform_fit(irf=np.zeros(BINS)), "empty irf"),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit_error(irf=np.ones(10)),
        "length mismatch",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit_error(mode="x"),
        "invalid mode",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit_error(fit_type="x"),
        "invalid fit type",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit_error(irf_min=0.0),
        "invalid minimum lifetime",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit_error(model_fn=_custom_model),
        "custom model without p0",
    ),
    Case(
        f"{ALIGN}.perform_fit_data",
        _perform_fit_error(t=np.arange(BINS) ** 2.0),
        "non-uniform axis",
    ),
    Case(f"{ALIGN}.generate_fit_maps", _maps(), "default"),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(force_C_normalized=False, mode="model_shift", initial_tau=2.0),
        "free amplitude and model shift",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(fit_type="curve_fit_circular", min_counts=800.0),
        "circular curve fit with count limit",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(valid_mask=np.array([[True, False, True], [True, True, False]])),
        "valid mask",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(valid_mask=np.array([True, False, True, True, True, False])),
        "flat valid mask",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(min_peak_counts=1e9),
        "no valid pixel",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(min_nonzero_bins=20, catch_exceptions=False),
        "bin count limit",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps(model_fn=_custom_model, p0=[1.0, 0.0, 2.0]),
        "custom model",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps_error(data=np.ones((3, 64))),
        "data not three-dimensional",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps_error(irf=np.ones(10)),
        "irf length mismatch",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps_error(t=np.ones(10)),
        "time axis length mismatch",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps_error(irf=np.zeros(BINS)),
        "empty irf",
    ),
    Case(
        f"{ALIGN}.generate_fit_maps",
        _maps_error(valid_mask=np.ones(4, dtype=bool)),
        "valid mask shape",
    ),
    Case(f"{ALIGN}.fit_maps_to_stack", _stack),
    Case(
        f"{ALIGN}.phasor_delay_from_hist",
        lambda f: [
            f(_decay_data(), PERIOD),
            f(_decay_data(), PERIOD, harmonic=2),
            f(list(range(1, 9)), 10.0, harmonic=3),
        ],
    ),
    Case(
        f"{ALIGN}.hist_for_plot",
        lambda f: [f(torch.arange(4.0)), f([1, 2, 3]), f(np.arange(3))],
    ),
    Case(f"{ALIGN}.sum_channel_applying_shifts", _shifted_sum(), "default"),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum(axis=None, backend="cpu"),
        "no summed axes",
    ),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum(axis=(), reverse_shifts=False, backend="numpy"),
        "empty axis and forward shifts",
    ),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum(axis=-2, backend="np"),
        "single negative axis",
    ),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum(axis=(0, 2), backend="auto", chunk_size=4),
        "two axes",
    ),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum_error(shifts=np.zeros(3)),
        "shift count mismatch",
    ),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum_error(backend="x"),
        "invalid backend",
    ),
    Case(
        f"{ALIGN}.sum_channel_applying_shifts",
        _shifted_sum_error(axis=9),
        "axis out of range",
    ),
]

SKIPPED: dict[str, str] = {}
