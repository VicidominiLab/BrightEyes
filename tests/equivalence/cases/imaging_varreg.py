"""Equivalence cases for the utility, loss, initialization, preprocessing and phantom modules of `varreg`."""

from importlib import import_module
from unittest import mock

import numpy as np
import torch
from _capture import printed as _printed
from _inputs import (
    positive_tensor,
    unit_tensor,
    varreg_canonical,
    varreg_counts,
    varreg_layout,
    varreg_psf,
)
from _registry import Case
from torch.fft import fftn, rfftn

UTILS = "varreg.utils"
REGULARIZERS = "varreg.regularizers"
LOSS = "varreg.loss"
INITIALIZERS = "varreg.initializers"
PREPROCESSING = "varreg.preprocessing"
PHANTOM = "varreg.phantom_simulator"


def _raised(call):
    try:
        call()
    except (ArithmeticError, NameError, RuntimeError, TypeError, ValueError) as error:
        return type(error).__name__, str(error)
    raise AssertionError("no exception raised")


def _package(obj, name):
    return import_module(obj.__module__.rpartition(".")[0] + "." + name)


def _with_default_seed(function, seed=0):
    real = np.random.default_rng
    with mock.patch.object(
        np.random,
        "default_rng",
        lambda value=None: real(seed if value is None else value),
    ):
        return function()


_LAYOUTS = ("confocal", "MID", "s2ISM")


def _bins():
    return np.random.default_rng(6).poisson(5.0, (3, 3, 2, 12))


def _image_pair():
    rng = np.random.default_rng(2)
    return rng.uniform(1, 2, (8, 8)), rng.uniform(1, 2, (4, 8, 8))


def _padded(function, shape):
    return function(unit_tensor(*shape, seed=5), (12, 11))


def _fourier_pair():
    return fftn(unit_tensor(3, 6, 5), dim=[1, 2]), fftn(
        unit_tensor(6, 5, 2, seed=5), dim=[0, 1]
    )


def _split_counts():
    return np.random.default_rng(3).poisson(20.0, (6, 5))


def _sparse_float():
    return np.random.default_rng(8).uniform(1, 2, (8, 8))


def _otf(utils):
    return utils.compute_otf_from_psf(varreg_canonical()[1], (16, 16))


CASES: list[Case] = [
    Case(f"{UTILS}:as_float_tensor", lambda f: f([1, 2, 3]), "integers become float"),
    Case(
        f"{UTILS}:as_float_tensor",
        lambda f: (f(np.arange(4.0), dtype=torch.float64), f(np.ones(3, np.float32))),
        "dtype given, float kept",
    ),
    Case(f"{UTILS}:standardize_dataset", lambda f: f(varreg_counts()), "confocal"),
    Case(f"{UTILS}:standardize_dataset", lambda f: f(varreg_counts(2)), "MID"),
    Case(
        f"{UTILS}:standardize_dataset",
        lambda f: f(varreg_counts(2), dtype=torch.float64),
        "dtype given",
    ),
    Case(
        f"{UTILS}:standardize_dataset",
        lambda f: _raised(lambda: f(np.ones((4, 4)))),
        "two dimensions fail",
    ),
    Case(
        f"{UTILS}:standardize_psf", lambda f: f(varreg_psf(channels=None)), "confocal"
    ),
    Case(f"{UTILS}:standardize_psf", lambda f: f(varreg_psf()), "MID"),
    Case(f"{UTILS}:standardize_psf", lambda f: f(varreg_psf(layers=2)), "s2ISM"),
    Case(
        f"{UTILS}:standardize_psf",
        lambda f: f(torch.ones(1, 4, 3, 2, dtype=torch.complex64), dtype=torch.float64),
        "otf, float64",
    ),
    Case(
        f"{UTILS}:standardize_psf",
        lambda f: f(
            torch.ones(1, 4, 3, 2, dtype=torch.complex128), dtype=torch.float32
        ),
        "otf, float32",
    ),
    Case(
        f"{UTILS}:standardize_psf",
        lambda f: f(varreg_psf(layers=2), target_shape=(16, 16)),
        "padded to an otf",
    ),
    Case(
        f"{UTILS}:standardize_psf",
        lambda f: _raised(lambda: f(np.ones(5))),
        "one dimension fails",
    ),
    Case(
        f"{UTILS}:standardize_object_history",
        lambda f: f(positive_tensor(8, 8, 3), 1),
        "three dimensions",
    ),
    Case(
        f"{UTILS}:standardize_object_history",
        lambda f: f(positive_tensor(2, 8, 8, 3), 2),
        "single canonical object",
    ),
    Case(
        f"{UTILS}:standardize_object_history",
        lambda f: f(positive_tensor(4, 8, 8, 3), 1),
        "confocal history",
    ),
    Case(
        f"{UTILS}:standardize_object_history",
        lambda f: f(positive_tensor(4, 1, 8, 8, 3), 1, dtype=torch.float64),
        "five dimensions",
    ),
    Case(
        f"{UTILS}:standardize_object_history",
        lambda f: _raised(lambda: f(positive_tensor(8, 8), 1)),
        "two dimensions fail",
    ),
    Case(
        f"{UTILS}:standardize_fidelity_weight",
        lambda f: [
            f(weight, torch.zeros(1, 16, 16, 2, 3))
            for weight in (
                1.0,
                positive_tensor(16, 16),
                positive_tensor(16, 16, 3),
                positive_tensor(16, 16, 2, 3),
            )
        ],
        "scalar and each layout",
    ),
    Case(
        f"{UTILS}:standardize_object_weight",
        lambda f: [
            f(weight, (1, 16, 16, 3))
            for weight in (
                0.5,
                positive_tensor(16, 16),
                positive_tensor(16, 16, 3),
            )
        ],
        "scalar and each layout",
    ),
    Case(
        f"{UTILS}:standardize_object_weight",
        lambda f: f(positive_tensor(16, 16), (1, 16, 16, 3), dtype=torch.float64),
        "dtype given",
    ),
    Case(
        f"{UTILS}:tensor_scalar_product",
        lambda f: (
            f(positive_tensor(2, 4, 5), positive_tensor(2, 4, 5, seed=2)),
            f(
                positive_tensor(2, 4, 5),
                positive_tensor(2, 4, 5, seed=2),
                axis_to_sum=(0, 1),
            ),
        ),
        "all axes and chosen axes",
    ),
    Case(
        f"{UTILS}:partial_convolution_fft",
        lambda f: _printed(
            lambda: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5))
        ),
        "spatial domain",
    ),
    Case(
        f"{UTILS}:partial_convolution_fft",
        lambda f: _printed(lambda: f(*_fourier_pair(), fourier=(True, True))),
        "both in the Fourier domain",
    ),
    Case(
        f"{UTILS}:partial_convolution_fft",
        lambda f: _printed(
            lambda: f(
                unit_tensor(2, 4),
                unit_tensor(4, 3, seed=5),
                dim1="ij",
                dim2="jk",
                axis="j",
            )
        ),
        "one convolved axis",
    ),
    Case(
        f"{UTILS}:partial_convolution_fft",
        lambda f: _raised(
            lambda: _printed(
                lambda: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5), axis="jx")
            )
        ),
        "unknown axis label fails",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5)),
        "default output order",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5), dim3="lkji"),
        "output order given",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5), padding=[8, 7]),
        "padding",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: f(
            rfftn(unit_tensor(3, 6, 5), dim=[1, 2], s=[6, 5]),
            unit_tensor(6, 5, 2, seed=5),
            fourier=(True, False),
        ),
        "kernel in the Fourier domain",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: f(unit_tensor(3, 6, 5).numpy(), unit_tensor(6, 5, 2, seed=5).numpy()),
        "numpy inputs",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: _raised(
            lambda: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5), dim3=3)
        ),
        "output order not a string fails",
    ),
    Case(
        f"{UTILS}:partial_convolution_rfft",
        lambda f: _raised(
            lambda: f(unit_tensor(3, 6, 5), unit_tensor(6, 5, 2, seed=5), axis="jx")
        ),
        "unknown axis label fails",
    ),
    Case(
        f"{UTILS}:forward_model",
        lambda f: f(_otf(_package(f, "utils")), positive_tensor(2, 16, 16, 3)),
        "forward projection",
    ),
    Case(
        f"{UTILS}:forward_model",
        lambda f: f(_otf(_package(f, "utils")), varreg_canonical()[0]),
        "back projection",
    ),
    Case(
        f"{UTILS}:forward_model",
        lambda f: _raised(
            lambda: f(_otf(_package(f, "utils")), positive_tensor(5, 7, 3))
        ),
        "incompatible shapes fail",
    ),
    Case(
        f"{UTILS}:centered_pad",
        lambda f: _padded(f, (7, 7)),
        "two dimensions",
    ),
    Case(
        f"{UTILS}:centered_pad",
        lambda f: _padded(f, (7, 7, 2)),
        "channels",
    ),
    Case(
        f"{UTILS}:centered_pad",
        lambda f: _padded(f, (3, 7, 7)),
        "planes",
    ),
    Case(
        f"{UTILS}:centered_pad",
        lambda f: _padded(f, (2, 7, 7, 3)),
        "planes and channels",
    ),
    Case(
        f"{UTILS}:centered_pad",
        lambda f: _raised(lambda: _padded(f, (6, 5))),
        "non-square fails",
    ),
    Case(
        f"{UTILS}:centered_pad",
        lambda f: _raised(lambda: f(unit_tensor(7, 7), (5, 12))),
        "smaller target fails",
    ),
    Case(
        f"{UTILS}:compute_otf_from_psf",
        lambda f: f(unit_tensor(2, 7, 7, 3), (12, 11)),
        "planes and channels",
    ),
    Case(
        f"{UTILS}:compute_otf_from_psf",
        lambda f: f(unit_tensor(1, 7, 7, 1), (16, 16)),
        "confocal",
    ),
    Case(f"{UTILS}:calculate_snr", lambda f: f(varreg_counts()), "default"),
    Case(
        f"{UTILS}:calculate_snr",
        lambda f: f(
            torch.from_numpy(varreg_counts()).float(), smooth_sigma=1.0, eps=1e-3
        ),
        "smoothed, epsilon given",
    ),
    Case(
        f"{UTILS}:normalization",
        lambda f: (
            f(_sparse_float(), 0.0, 5.0),
            f(torch.from_numpy(_sparse_float()), -1.0, 1.0),
            f(np.full(4, 3.0), 2.0, 7.0),
        ),
        "array, tensor and constant input",
    ),
    Case(f"{UTILS}:calculate_weight_map", lambda f: f(varreg_counts()), "default"),
    Case(
        f"{UTILS}:calculate_weight_map",
        lambda f: f(varreg_counts(), smooth_sigma=1.0, normalize=False),
        "smoothed, not normalized",
    ),
    Case(
        f"{UTILS}:calculate_noise_variance",
        lambda f: f(positive_tensor(3, 6, 8, 4)),
        "even size",
    ),
    Case(
        f"{UTILS}:radial_profile",
        lambda f: (
            f(_sparse_float(), [4, 4]),
            f(_sparse_float() + 1j * _sparse_float().T, [3, 5]),
        ),
        "real and complex image",
    ),
    Case(
        f"{UTILS}:compute_radial_noise_variance",
        lambda f: f(positive_tensor(3, 6, 8, 4)),
        "even size",
    ),
    Case(
        f"{UTILS}:compute_radial_noise_variance",
        lambda f: f(positive_tensor(2, 7, 5, 4)),
        "odd size",
    ),
    Case(
        f"{UTILS}:root_mean_square_error",
        lambda f: f(*_image_pair()[:1], _image_pair()[1][0] * 1.1),
        "single image",
    ),
    Case(
        f"{UTILS}:root_mean_square_error",
        lambda f: f(*_image_pair()),
        "several estimates",
    ),
    Case(
        f"{UTILS}:root_mean_square_error",
        lambda f: f(*(torch.from_numpy(x) for x in _image_pair())),
        "tensors",
    ),
    Case(
        f"{UTILS}:peak_signal_to_noise_ratio",
        lambda f: f(*_image_pair()),
        "normalized estimates",
    ),
    Case(
        f"{UTILS}:peak_signal_to_noise_ratio",
        lambda f: f(*_image_pair(), normalize_entries=False),
        "not normalized",
    ),
    Case(
        f"{UTILS}:peak_signal_to_noise_ratio",
        lambda f: f(*_image_pair()[:1], _image_pair()[1][0] * 1.1),
        "single image",
    ),
    Case(
        f"{UTILS}:kl_divergence",
        lambda f: f(varreg_counts(2), positive_tensor(16, 16, 2, 3)),
        "default",
    ),
    Case(
        f"{UTILS}:kl_divergence",
        lambda f: f(varreg_counts(2), positive_tensor(4, 16, 16, 2, 3)),
        "iteration axis",
    ),
    Case(
        f"{UTILS}:kl_divergence",
        lambda f: f(
            varreg_counts(2),
            positive_tensor(4, 16, 16, 2, 3),
            normalize_entries=False,
            eps=1e-6,
        ),
        "not normalized, epsilon given",
    ),
    Case(
        f"{UTILS}:kl_divergence",
        lambda f: f(varreg_counts(2), positive_tensor(4, 16, 16, 2, 3), full_KL=True),
        "full",
    ),
    Case(
        f"{UTILS}:kl_divergence",
        lambda f: f(varreg_counts(2), positive_tensor(4, 16, 16, 2, 3), k_loss=True),
        "per realization",
    ),
    Case(
        f"{UTILS}:kl_divergence",
        lambda f: _raised(
            lambda: f(positive_tensor(4, 16, 16, 2, 3), positive_tensor(16, 16, 2, 3))
        ),
        "fewer dimensions fail",
    ),
    Case(
        f"{UTILS}:autocorrelation2d",
        lambda f: (f(_sparse_float()), f(_sparse_float(), normalize=False)),
        "normalized and raw",
    ),
    Case(
        f"{REGULARIZERS}:compute_centroid",
        lambda f: [f(positive_tensor(2, 8, 8, 3), name) for name in ("l2", "l1", "KL")],
        "each regularizer",
    ),
    Case(
        f"{REGULARIZERS}:compute_centroid",
        lambda f: f(positive_tensor(2, 8, 8, 3), "kl", eps=0.7),
        "kl with epsilon",
    ),
    Case(
        f"{REGULARIZERS}:compute_centroid",
        lambda f: [
            _raised(lambda: f(torch.full((2, 3), -1.0), "kl")),
            _raised(lambda: f(torch.ones(2, 3, dtype=torch.complex64), "l1")),
            _raised(lambda: f(torch.ones(2, 3, dtype=torch.complex64), "kl")),
            _raised(lambda: f(torch.ones(2, 3), "l3")),
        ],
        "invalid input fails",
    ),
    Case(f"{REGULARIZERS}:grad_l2", lambda f: f(positive_tensor(2, 8, 8, 3)), ""),
    Case(f"{REGULARIZERS}:grad_l1", lambda f: f(positive_tensor(2, 8, 8, 3)), ""),
    Case(f"{REGULARIZERS}:grad_kl", lambda f: f(positive_tensor(2, 8, 8, 3)), ""),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            varreg_counts(), positive_tensor(16, 16, 3), varreg_psf(channels=None)
        ),
        "confocal",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(varreg_counts(2), positive_tensor(16, 16, 3), varreg_psf()),
        "MID",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            varreg_counts(2), positive_tensor(2, 16, 16, 3), varreg_psf(layers=2)
        ),
        "s2ISM",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            varreg_counts(2), positive_tensor(4, 2, 16, 16, 3), varreg_psf(layers=2)
        ),
        "reconstruction history",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            torch.from_numpy(varreg_counts(2)).float(),
            positive_tensor(2, 16, 16, 3),
            _otf(_package(f, "utils")),
            eps=1e-6,
        ),
        "otf, tensor data, epsilon given",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            varreg_counts(2),
            positive_tensor(2, 16, 16, 3),
            varreg_psf(layers=2),
            weight_map=positive_tensor(16, 16, seed=3),
            full_KL=True,
        ),
        "weight map, full",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            varreg_counts(2),
            positive_tensor(2, 16, 16, 3),
            varreg_psf(layers=2),
            weight_map=positive_tensor(16, 16, 3, seed=3),
            k_loss=False,
        ),
        "weight map per realization, summed",
    ),
    Case(
        f"{LOSS}:fidelity_loss",
        lambda f: f(
            varreg_counts(2),
            positive_tensor(2, 16, 16, 3),
            varreg_psf(layers=2),
            full_KL=True,
            k_loss=False,
        ),
        "full, summed",
    ),
    Case(
        f"{LOSS}:regularization_loss",
        lambda f: [
            f(positive_tensor(2, 16, 16, 3), name) for name in ("l2", "l1", "kl")
        ],
        "each regularizer",
    ),
    Case(
        f"{LOSS}:regularization_loss",
        lambda f: f(
            positive_tensor(3, 2, 16, 16, 3),
            weight_map=positive_tensor(16, 16, 1, seed=3),
            iteration_axis=True,
        ),
        "weight map, iteration axis",
    ),
    Case(
        f"{LOSS}:regularization_loss",
        lambda f: f(positive_tensor(2, 16, 16, 3), "kl", eps=0.1),
        "kl with epsilon",
    ),
    Case(
        f"{LOSS}:regularization_loss",
        lambda f: f(positive_tensor(2, 16, 16, 3), fourier=True),
        "fourier",
    ),
    Case(
        f"{LOSS}:regularization_loss",
        lambda f: f(
            positive_tensor(2, 16, 16, 3),
            fourier=True,
            otf=_otf(_package(f, "utils"))[..., :1],
            weight_map=positive_tensor(1, 16, 9, 3, seed=3),
        ),
        "fourier with otf",
    ),
    Case(
        f"{LOSS}:regularization_loss",
        lambda f: [
            _raised(lambda: f(positive_tensor(2, 4, 4, 3), eps=0.0)),
            _raised(lambda: f(positive_tensor(2, 4, 4, 3), "l3")),
            _raised(lambda: f(positive_tensor(2, 4, 4, 3), "l1", fourier=True)),
        ],
        "invalid input fails",
    ),
    Case(
        f"{INITIALIZERS}:flat",
        lambda f: f(
            (2, 16, 16, 3), varreg_counts().sum((0, 1)).reshape(1, 1, 1, 3), "cpu"
        ),
        "",
    ),
    Case(
        f"{INITIALIZERS}:data",
        lambda f: f(
            varreg_counts(), varreg_counts().sum((0, 1)).reshape(1, 1, 1, 3), "cpu"
        ),
        "confocal",
    ),
    Case(
        f"{INITIALIZERS}:data",
        lambda f: f(varreg_counts(2), varreg_counts(2).sum((0, 1, 2)), "cpu"),
        "MID",
    ),
    Case(
        f"{INITIALIZERS}:data",
        lambda f: _raised(lambda: f(varreg_counts(), torch.ones(2), "cpu")),
        "flux count mismatch fails",
    ),
    Case(
        f"{INITIALIZERS}:data",
        lambda f: f(np.zeros((16, 16, 3)), torch.ones(3), "cpu"),
        "empty data",
    ),
    Case(
        f"{INITIALIZERS}:rl_deconvolution",
        lambda f: f(varreg_counts(), varreg_psf(channels=None), max_iter=3),
        "confocal",
    ),
    Case(
        f"{INITIALIZERS}:rl_deconvolution",
        lambda f: f(varreg_counts(2), varreg_psf(layers=2), max_iter=3),
        "s2ISM",
    ),
    Case(
        f"{INITIALIZERS}:rl_deconvolution",
        lambda f: f(varreg_counts(2), varreg_psf(), max_iter=2, epsilon=1e-3),
        "MID, epsilon given",
    ),
    Case(
        f"{INITIALIZERS}:rl_deconvolution",
        lambda f: f(np.zeros((16, 16, 3)), varreg_psf(channels=None), max_iter=3),
        "empty data",
    ),
    Case(
        f"{INITIALIZERS}:rl_deconvolution",
        lambda f: f(varreg_counts(), varreg_psf(channels=None), max_iter=0),
        "no iterations",
    ),
    Case(
        f"{INITIALIZERS}:rl_deconvolution",
        lambda f: [
            _raised(lambda: f(varreg_counts(), varreg_psf(channels=None), max_iter=-1)),
            _raised(lambda: f(varreg_counts(), varreg_psf(channels=None), epsilon=0.0)),
            _raised(lambda: f(varreg_counts(), np.zeros((9, 9)))),
        ],
        "invalid input fails",
    ),
    Case(
        f"{INITIALIZERS}:initialize",
        lambda f: [f(*varreg_layout(name), "flat", "cpu") for name in _LAYOUTS],
        "flat, every layout",
    ),
    Case(
        f"{INITIALIZERS}:initialize",
        lambda f: [f(*varreg_layout(name), "DATA", "cpu") for name in _LAYOUTS],
        "data, every layout",
    ),
    Case(
        f"{INITIALIZERS}:initialize",
        lambda f: [
            f(*varreg_layout(name), "rl", "cpu", max_iter=3) for name in _LAYOUTS
        ],
        "rl, every layout",
    ),
    Case(
        f"{INITIALIZERS}:initialize",
        lambda f: _raised(lambda: f(*varreg_layout("confocal"), "other", "cpu")),
        "unknown method fails",
    ),
    Case(f"{PREPROCESSING}:divisors", lambda f: [f(n) for n in (1, 36, 37, 100)], ""),
    Case(
        f"{PREPROCESSING}:rebin",
        lambda f: f(np.arange(60.0).reshape(5, 12), 3, 4),
        "last axis",
    ),
    Case(
        f"{PREPROCESSING}:rebin",
        lambda f: f(np.arange(65.0).reshape(13, 5), 4, 3, channel=0),
        "first axis, remainder dropped",
    ),
    Case(
        f"{PREPROCESSING}:rebin",
        lambda f: _raised(lambda: f(np.ones((3, 4)), 5, 2)),
        "too many bins fails",
    ),
    Case(
        f"{PREPROCESSING}:binomial_split",
        lambda f: f(_split_counts(), device="cpu"),
        "two splits",
    ),
    Case(
        f"{PREPROCESSING}:binomial_split",
        lambda f: f(_split_counts(), n_splits=4, p=0.3, device="cpu"),
        "four splits, probability",
    ),
    Case(
        f"{PREPROCESSING}:binomial_split",
        lambda f: f(torch.from_numpy(_split_counts()), n_splits=1, device="cpu"),
        "single split, tensor",
    ),
    Case(
        f"{PREPROCESSING}:binomial_split",
        lambda f: [
            _raised(lambda: f(_split_counts(), n_splits=3, device="cpu")),
            _raised(lambda: f(_split_counts(), p=2.0, device="cpu")),
            _raised(lambda: f(-_split_counts(), device="cpu")),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PREPROCESSING}:multinomial_split",
        lambda f: f(_split_counts()),
        "default",
    ),
    Case(
        f"{PREPROCESSING}:multinomial_split",
        lambda f: f(_split_counts(), n_splits=4, dtype=torch.int32),
        "four splits, dtype",
    ),
    Case(
        f"{PREPROCESSING}:multinomial_split",
        lambda f: f(
            _split_counts(),
            probabilities=[0.2, 0.0, 0.8],
            generator=torch.Generator().manual_seed(5),
        ),
        "probabilities with a zero, generator",
    ),
    Case(
        f"{PREPROCESSING}:multinomial_split",
        lambda f: (
            f(_split_counts().astype(np.float64), n_splits=1),
            f(_split_counts().astype(np.uint64), n_splits=2),
            f(torch.from_numpy(_split_counts()).to(torch.uint64), n_splits=2),
            f([[1, 2], [3, 4]], probabilities=[1, 3], device="cpu"),
        ),
        "float, unsigned and list input",
    ),
    Case(
        f"{PREPROCESSING}:multinomial_split",
        lambda f: [
            _raised(lambda: f(-_split_counts())),
            _raised(lambda: f(_split_counts() + 0.5)),
            _raised(lambda: f(np.array([np.nan, 1.0]))),
            _raised(lambda: f(_split_counts(), n_splits=3, probabilities=[1, 1])),
            _raised(lambda: f(_split_counts(), probabilities=[0.0, 0.0])),
            _raised(lambda: f(_split_counts(), probabilities=[-1, 2])),
            _raised(lambda: f(_split_counts(), n_splits=0)),
            _raised(lambda: f(np.array([2**63], dtype=np.uint64))),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PREPROCESSING}:make_noisy_dataset",
        lambda f: f(_bins(), 3, 2, 3, seed=1),
        "without replacement",
    ),
    Case(
        f"{PREPROCESSING}:make_noisy_dataset",
        lambda f: f(_bins(), 4, 3, 2, seed=2, replace=True),
        "with replacement",
    ),
    Case(
        f"{PREPROCESSING}:make_noisy_dataset",
        lambda f: _with_default_seed(lambda: f(_bins(), 3, 2, 3)),
        "no seed, generator patched",
    ),
    Case(
        f"{PREPROCESSING}:make_noisy_dataset",
        lambda f: [
            _raised(lambda: f(np.ones((3, 3, 12)), 3, 2, 3)),
            _raised(lambda: f(_bins(), 7, 2, 1)),
            _raised(lambda: f(_bins(), 0, 2, 3)),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PHANTOM}:disk",
        lambda f: (f(5.0, (16, 20)), f(0, (5, 5))),
        "",
    ),
    Case(
        f"{PHANTOM}:disk",
        lambda f: [
            _raised(lambda: f(-1, (8, 8))),
            _raised(lambda: f(2, (8,))),
            _raised(lambda: f(2, (8, 0))),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PHANTOM}:sparse_random_image",
        lambda f: f((12, 12), 6, value_range=(0.2, 0.9), margin=2),
        "two dimensions",
    ),
    Case(
        f"{PHANTOM}:sparse_random_image",
        lambda f: f((5, 6, 7), 10, dtype=np.float64),
        "three dimensions",
    ),
    Case(
        f"{PHANTOM}:sparse_random_image",
        lambda f: [
            _raised(lambda: f((), 1)),
            _raised(lambda: f((4, 4), 1.5)),
            _raised(lambda: f((4, 4), -1)),
            _raised(lambda: f((4, 4), 1, margin=1.5)),
            _raised(lambda: f((4, 4), 1, margin=-1)),
            _raised(lambda: f((4, 4), 1, value_range=(1, 0))),
            _raised(lambda: f((4, 4), 1, margin=2)),
            _raised(lambda: f((4, 4), 17)),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PHANTOM}:generate_disk_phantom",
        lambda f: f((32, 32), 3, 3),
        "default",
    ),
    Case(
        f"{PHANTOM}:generate_disk_phantom",
        lambda f: f((32, 24), 2, 4.0, value_range=(0.5, 1.0), smooth=True),
        "smoothed, float radius",
    ),
    Case(
        f"{PHANTOM}:generate_disk_phantom",
        lambda f: _raised(lambda: f((32, 32), 3, 2.5)),
        "non-integer radius fails",
    ),
    Case(
        f"{PHANTOM}:Fit_gaussian2D",
        lambda f: (
            f(1.0, 2.0, 1.5, 2.5, 0.5, 1.0, 10.0),
            f(*np.meshgrid(np.arange(6.0), np.arange(5.0)), 1.5, 2.5, 2.0, 3.0, 7.0),
        ),
        "scalar and grid",
    ),
    Case(
        f"{PHANTOM}:Fit_gaussian2DDraw",
        lambda f: f(12, 10, 1.5, 2.5, 3.0, 4.0, 0.4, 9.0, 0.5),
        "",
    ),
    Case(
        f"{PHANTOM}:Nucleus",
        lambda f: f((32, 32), 4, (0.5, 1.0), (2.0, 3.0), (12, 8), seed=3),
        "seed",
    ),
    Case(
        f"{PHANTOM}:Nucleus",
        lambda f: f((24, 28), 3, (0.5, 1.0), (2.0, 3.0), (10, 9), background_noise=0.1),
        "global generator",
    ),
    Case(
        f"{PHANTOM}:Nucleus",
        lambda f: [
            _raised(lambda: f((16, 16), -1, (1, 2), (1, 2), (5, 5))),
            _raised(lambda: f((16, 16), 1, (1, 2), (1, 2), (0, 5))),
            _raised(lambda: f((16, 16), 1, (1, 2), (0, 2), (5, 5))),
            _raised(lambda: f((16, 16), 1, (1, 2), (3, 2), (5, 5))),
            _raised(lambda: f((15, 15), 1, (1, 2), (1, 2), (0.1, 0.1))),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PHANTOM}:generate_membrane_phantom",
        lambda f: f((32, 32), (12, 8), 5.0, (0.5, 1.0), seed=3),
        "seed",
    ),
    Case(
        f"{PHANTOM}:generate_membrane_phantom",
        lambda f: f(
            (24, 28), (10, 9), 8.0, (0.2, 1.0), noise_level=0.0, blur_sigma=0.0
        ),
        "global generator, no noise, no blur",
    ),
    Case(
        f"{PHANTOM}:generate_membrane_phantom",
        lambda f: [
            _raised(lambda: f((16, 16), (0, 5), 5.0, (1, 2))),
            _raised(lambda: f((16, 16), (5, 5), 0.0, (1, 2))),
            _raised(lambda: f((16, 16), (5, 5), 5.0, (1, 2), blur_sigma=-1.0)),
        ],
        "invalid input fails",
    ),
    Case(
        f"{PHANTOM}:generate_hollow_membrane_phantom",
        lambda f: f((32, 32), 4, (4, 9), (0.5, 1.0), 0.2, 1.0, 0.05, seed=3),
        "seed",
    ),
    Case(
        f"{PHANTOM}:generate_hollow_membrane_phantom",
        lambda f: f(shape=(24, 24), n_ellipses=3, radius_range=(3, 8)),
        "global generator",
    ),
    Case(
        f"{PHANTOM}:generate_hollow_membrane_phantom",
        lambda f: [
            _raised(lambda: f(n_ellipses=-1)),
            _raised(lambda: f(radius_range=(0, 5))),
            _raised(lambda: f(radius_range=(6, 5))),
            _raised(lambda: f(amplitude_range=(1.0, 0.5))),
            _raised(lambda: f(thickness=0.0)),
            _raised(lambda: f(noise_level=-1.0)),
        ],
        "invalid input fails",
    ),
]

SKIPPED: dict[str, str] = {}
