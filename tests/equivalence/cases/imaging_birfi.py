"""Equivalence cases for `birfi`."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from _inputs import decay_histogram, positive_tensor, unit_tensor
from _registry import SKIP_PLOT_ONLY, Case
from torch.fft import fftn

BIRFI = "birfi.birfi"
BIRFI_UTILS = "birfi.utils"


def _raised(call):
    try:
        call()
    except (RuntimeError, ValueError) as error:
        return type(error).__name__, str(error)
    raise AssertionError("no exception raised")


def _smoothed(data, width):
    return np.convolve(data, np.ones(width) / width, mode="same")


def _birfi(cls, channels=True, **options):
    data = decay_histogram()
    if channels:
        data = np.stack(
            [data, 0.6 * _smoothed(data, 3) + 4, 1.3 * _smoothed(data, 7) + 1], axis=1
        )
    return cls(data, device=torch.device("cpu"), **options)


def _found(cls, **options):
    estimator = _birfi(cls)
    estimator.find_t0_t1(**options)
    return estimator


def _fitted(cls, **options):
    estimator = _found(cls)
    estimator.fit_exponential(steps=30, **options)
    return estimator


def _kernel(cls):
    estimator = _fitted(cls)
    estimator.generate_kernel()
    return estimator


def _deconvolved(cls, **options):
    estimator = _kernel(cls)
    estimator.richardson_lucy_deconvolution(**options)
    return estimator


def _state(estimator):
    return {**vars(estimator), "device": str(estimator.device)}


def _data_fit(cls):
    estimator = _fitted(cls)
    estimator.generate_data_fit()
    return estimator


def _run(cls, **options):
    estimator = _birfi(cls)
    options = {"steps": 40, "rl_iterations": 5, **options}
    return estimator.run(**options), _state(estimator)


def _plotted(cls, method):
    estimator = _deconvolved(cls, iterations=3)
    estimator.generate_data_fit()
    plt.close("all")
    getattr(estimator, method)()
    lines = [line.get_ydata() for axis in plt.gcf().axes for line in axis.lines]
    plt.close("all")
    return lines


CASES: list[Case] = [
    Case(f"{BIRFI}:Birfi", lambda c: _state(_birfi(c)), "three channels"),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(_birfi(c, channels=False, dt=0.25)),
        "one channel, step",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: c(np.ones((4, 3, 2)), device=torch.device("cpu"))),
        "three dimensions fail",
    ),
    Case(f"{BIRFI}:Birfi", lambda c: _state(_found(c)), "find_t0_t1: defaults"),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(
            _found(c, window_length=7, polyorder=2, persistence=3, threshold=0.2)
        ),
        "find_t0_t1: other filter and thresholds",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _found(c, window_length=10)),
        "find_t0_t1: even window fails",
    ),
    Case(f"{BIRFI}:Birfi", lambda c: _state(_fitted(c)), "fit_exponential: defaults"),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(_fitted(c, offset=2, lr=0.05)),
        "fit_exponential: offset, learning rate",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _birfi(c).fit_exponential(steps=2)),
        "fit_exponential: before the interval fails",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(_data_fit(c)),
        "generate_data_fit",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _found(c).generate_data_fit()),
        "generate_data_fit: before the fit fails",
    ),
    Case(f"{BIRFI}:Birfi", lambda c: _state(_kernel(c)), "generate_kernel"),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _birfi(c).generate_kernel()),
        "generate_kernel: before the fit fails",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(_deconvolved(c)),
        "richardson_lucy_deconvolution: defaults",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(_deconvolved(c, iterations=4, eps=1e-2, regularization=1)),
        "richardson_lucy_deconvolution: no median filter",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _state(_deconvolved(c, iterations=4, regularization=5)),
        "richardson_lucy_deconvolution: wide median filter",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _fitted(c).richardson_lucy_deconvolution()),
        "richardson_lucy_deconvolution: before the kernel fails",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _run(c),
        "run: defaults",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _run(
            c,
            lr=0.05,
            steps=40,
            rl_iterations=4,
            regularization=1,
            window_length=9,
            polyorder=2,
            persistence=3,
            threshold=0.1,
        ),
        "run: other parameters",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _plotted(c, "plot_forward_model"),
        "plot_forward_model: plotted lines",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _kernel(c).plot_forward_model()),
        "plot_forward_model: before the deconvolution fails",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _plotted(c, "plot_raw_and_fit"),
        "plot_raw_and_fit: plotted lines",
    ),
    Case(
        f"{BIRFI}:Birfi",
        lambda c: _raised(lambda: _found(c).plot_raw_and_fit()),
        "plot_raw_and_fit: before the fit fails",
    ),
    Case(
        f"{BIRFI_UTILS}:pad_tensor",
        lambda f: [
            f(positive_tensor(4, 6), 2, 3, 1, mode=mode)
            for mode in ("reflect", "replicate", "constant")
        ],
        "each mode",
    ),
    Case(
        f"{BIRFI_UTILS}:pad_tensor",
        lambda f: (
            f(positive_tensor(4, 6), 0, 0, 0),
            _raised(lambda: f(positive_tensor(4, 6), 1, 1, 0, mode="other")),
        ),
        "no padding, unknown mode fails",
    ),
    Case(
        f"{BIRFI_UTILS}:median_filter",
        lambda f: f(positive_tensor(9, 8)),
        "all dimensions",
    ),
    Case(
        f"{BIRFI_UTILS}:median_filter",
        lambda f: [
            f(positive_tensor(9, 8), window_size=(5, 3), dims=[0, 1], mode=mode)
            for mode in ("reflect", "replicate", "constant")
        ],
        "window per dimension, each mode",
    ),
    Case(
        f"{BIRFI_UTILS}:median_filter",
        lambda f: f(positive_tensor(9, 8, 2), window_size=5, dims=[0]),
        "one dimension",
    ),
    Case(
        f"{BIRFI_UTILS}:median_filter",
        lambda f: [
            _raised(lambda: f(positive_tensor(9, 8), window_size=4)),
            _raised(
                lambda: f(positive_tensor(9, 8), window_size=(3, 3, 3), dims=[0, 1])
            ),
        ],
        "invalid window fails",
    ),
    Case(
        f"{BIRFI_UTILS}:generate_truncated_exponential",
        lambda f: f(
            torch.arange(20.0),
            {"A": torch.tensor(3.0), "k": 0.3, "C": torch.tensor(0.5), "t0": 4.0},
        ),
        "tensor time",
    ),
    Case(
        f"{BIRFI_UTILS}:generate_truncated_exponential",
        lambda f: f(np.linspace(0, 5, 11), {"A": 1, "k": 0.7, "C": 0, "t0": 1.5}),
        "array time",
    ),
    Case(
        f"{BIRFI_UTILS}:partial_convolution",
        lambda f: f(unit_tensor(6, 2), unit_tensor(6), dim1="xc", dim2="x", axis="x"),
        "time domain",
    ),
    Case(
        f"{BIRFI_UTILS}:partial_convolution",
        lambda f: f(
            fftn(unit_tensor(6, 2), dim=0),
            fftn(unit_tensor(6), dim=0),
            dim1="xc",
            dim2="x",
            axis="x",
            fourier=(True, True),
        ),
        "Fourier domain",
    ),
    Case(
        f"{BIRFI_UTILS}:partial_convolution",
        lambda f: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5)),
        "default labels",
    ),
    Case(
        f"{BIRFI_UTILS}:estimate_lifetime",
        lambda f: f(
            torch.arange(30.0) * 0.5,
            4.0 * torch.exp(-torch.arange(30.0) / 6) + 1,
            2,
            25,
        ),
        "",
    ),
    Case(
        f"{BIRFI_UTILS}:estimate_lifetime",
        lambda f: f(np.arange(40.0), decay_histogram()[:40], 0, 39),
        "arrays",
    ),
]

SKIPPED: dict[str, str] = {
    f"{BIRFI_UTILS}:plot_dataset": SKIP_PLOT_ONLY,
}
