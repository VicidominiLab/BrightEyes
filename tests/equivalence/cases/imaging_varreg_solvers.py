"""Equivalence cases for the solver and lambda-selection modules of `varreg`."""

from importlib import import_module

import torch
from _capture import printed as _printed
from _inputs import (
    positive_tensor,
    varreg_canonical,
    varreg_confocal_image,
    varreg_counts,
    varreg_layout,
    varreg_psf,
)
from _registry import Case
from torch.fft import rfftn

OPTIMIZER = "varreg.optimizer"
ADAM = "varreg.adam_optimizer"
DISCREPANCY = "varreg.discrepancy_principle"
WHITENESS = "varreg.whiteness_principle"
_LAYOUTS = ("confocal", "MID", "s2ISM")


def _raised(call):
    try:
        call()
    except (ArithmeticError, NameError, RuntimeError, TypeError, ValueError) as error:
        return type(error).__name__, str(error)
    raise AssertionError("no exception raised")


def _package(obj, name):
    return import_module(obj.__module__.rpartition(".")[0] + "." + name)


def _otf(utils):
    return utils.compute_otf_from_psf(varreg_canonical()[1], (16, 16))


def _reconstruct(function, layout="confocal", **options):
    dset, psf = varreg_layout(layout)
    options = {"process": "cpu", "max_iter": 3, **options}
    return _printed(lambda: function(dset, psf, 0.05, **options))


def _iterate(function, **options):
    dset, psf = varreg_canonical()
    arguments = {
        "max_iter": 3,
        "lam": 0.05,
        "weight": 1.0,
        "stop_tol": None,
        "method": "sgp",
        "device": torch.device("cpu"),
        "regularizer_norm": "l2",
        "initialization": "flat",
        "normalize": True,
        "stop_criteria": "rel_diff",
        "rec_to_save": "last",
        "step_size_search": "fixed",
        "line_search_params": None,
        **options,
    }
    return _printed(lambda: function(dset, psf, **arguments))


def _update(function, **options):
    dset, _ = varreg_canonical()
    otf = _otf(_package(function, "utils"))
    arguments = {
        "lam": 0.05,
        "weight": 1.0,
        "eps": 1e-8,
        "m": 1,
        "method": "sgp",
        "regularizer": "l2",
        "normalize": True,
        "step_size_search": "fixed",
        "initial_step_size": torch.full((3,), 0.8),
        **options,
    }
    return _printed(
        lambda: function(
            dset, positive_tensor(2, 16, 16, 3), otf, torch.conj(otf), **arguments
        )
    )


def _line_search(function, **options):
    otf = _otf(_package(function, "utils"))
    obj = positive_tensor(2, 16, 16, 3)
    arguments = {
        "img": varreg_canonical()[0],
        "obj": obj,
        "obj_tmp": 0.4 * obj + 0.2 * positive_tensor(2, 16, 16, 3, seed=6),
        "psf_fft": otf,
        "grad": torch.zeros(2, 16, 16, 3),
        "lam": 0.05,
        "D_matrix_inv": obj,
        **options,
    }
    return _printed(lambda: function(**arguments))


def _batches(function, layout="confocal", **options):
    dset, psf = varreg_layout(layout)
    arguments = {"max_iter": 2, "process": "cpu", **options}
    return _printed(lambda: function(dset, psf, [10, 9], 2, 0.05, **arguments))


def _adam(function, k=1, **options):
    arguments = {
        "num_iter": 3,
        "process": "cpu",
        "lr": 0.5,
        "rec_to_save": "all",
        "verbose": False,
        **options,
    }
    return _printed(
        lambda: function(
            varreg_counts(k=k), varreg_psf(channels=None), 0.05, **arguments
        )
    )


def _warmup(module, losses, **options):
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=0
    )
    scheduler = module.WarmupWithPlateau(optimizer, plateau, warmup_steps=3, **options)
    rates = []
    for loss in losses:
        optimizer.step()
        scheduler.step(torch.tensor(loss))
        rates.append(optimizer.param_groups[0]["lr"])
    return rates, scheduler


def _solver(function, **options):
    dset, psf = varreg_layout("confocal")
    arguments = {"process": "cpu", "max_iter": 2, "stop_tol": 1e-5, **options}

    def solve(lam):
        return function(dset, psf, lam, **arguments)

    return solve


def _nedp(function, lambdas=(0.01, 0.1, 1.0), **options):
    solver = _solver(_package(function, "optimizer").variance_reconstruction)
    image, psf = varreg_confocal_image()
    arguments = {"Delta_GT": 150.0, "device": "cpu", **options}
    return _printed(
        lambda: function(solver, lambdas, image, psf, torch.zeros(1, 4), **arguments)
    )


def _discrepancy_otf(function):
    psf = varreg_confocal_image()[1]
    pad = import_module(function.__module__).centered_pad
    return rfftn(pad(psf, (16, 16)) / psf.sum(), dim=(0, 1))


def _bisection(function, delta=None, **options):
    solver = _solver(_package(function, "optimizer").variance_reconstruction)
    image, psf = varreg_confocal_image()
    if delta is None:
        compute = import_module(function.__module__).compute_D_lambda
        otf = _discrepancy_otf(function)
        ends = [compute(solver(lam)[0][-1], image, otf) for lam in (0.001, 100.0)]
        delta = sum(ends) / 2
    return _printed(
        lambda: function(
            solver,
            0.001,
            100.0,
            image,
            psf,
            torch.zeros(1, 4),
            Delta_GT=delta,
            **options,
        )
    )


def _fit(function, **options):
    y = positive_tensor(6, 2) * 3
    return _printed(
        lambda: function(
            y,
            positive_tensor(6, 2, seed=2),
            positive_tensor(6, 2, seed=3),
            **options,
        )
    )


def _whiteness_search(function, **options):
    maker = _package(function, "whiteness_principle").make_variance_reconstructor
    dset, psf = varreg_confocal_image()
    solver = maker(
        varreg_counts(), varreg_psf(channels=None), process="cpu", max_iter=2
    )
    arguments = {"device": "cpu", **options}
    return _printed(lambda: function(solver, (0.01, 0.1, 1.0), dset, psf, **arguments))


def _z_inputs(function):
    return (
        positive_tensor(2, 16, 16, 3),
        _otf(_package(function, "utils")),
        varreg_canonical()[0],
    )


def _reconstructor(function, **options):
    solver = function(
        varreg_counts(), varreg_psf(channels=None), process="cpu", max_iter=2, **options
    )
    return solver, solver(0.1)


CASES: list[Case] = [
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [
            _reconstruct(f, method=method)
            for method in ("osl", "abs", "sgm", "pgd", "sgp")
        ],
        "each method",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [_reconstruct(f, layout) for layout in _LAYOUTS],
        "each layout",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [
            _reconstruct(f, "s2ISM", step_size_search=search)
            for search in ("fixed", "line_search", "rl", 0.5)
        ],
        "each step size search, sgp",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [
            _reconstruct(f, method="pgd", step_size_search=search)
            for search in ("line_search", "rl", 0.7)
        ],
        "each step size search, pgd",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [
            _reconstruct(f, method=method, regularizer_norm=norm)
            for method in ("sgm", "sgp")
            for norm in ("l1", "kl")
        ],
        "each regularizer",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [
            _reconstruct(f, "s2ISM", initialization=start)
            for start in ("flat", "data", "rl")
        ],
        "each initialization",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: _reconstruct(f, rec_to_save="all", return_pointwise_sol=True),
        "full history, summed over realizations",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: _reconstruct(
            f,
            "MID",
            max_iter=20,
            stop_tol=0.7,
            stop_criteria="step_size",
            rec_to_save="all",
            step_size_search=0.6,
        ),
        "early stop on the step size",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: _reconstruct(
            f, max_iter=30, stop_tol=5.0, rec_to_save="all", idx_in_focus=0
        ),
        "early stop on the relative change",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: _reconstruct(
            f,
            weight_map=positive_tensor(16, 16, seed=3),
            normalize=False,
            line_search_params=(10, 2),
            verbose=True,
            step_size_search="line_search",
        ),
        "weight map, no normalization, verbose",
    ),
    Case(
        f"{OPTIMIZER}:variance_reconstruction",
        lambda f: [
            _raised(lambda: _reconstruct(f, method="other")),
            _raised(lambda: _reconstruct(f, step_size_search=-1.0)),
            _raised(lambda: _reconstruct(f, step_size_search="other")),
            _raised(lambda: _reconstruct(f, stop_tol=1.0, stop_criteria="other")),
        ],
        "invalid input fails",
    ),
    Case(
        f"{OPTIMIZER}:iterate",
        lambda f: _iterate(f),
        "defaults",
    ),
    Case(
        f"{OPTIMIZER}:iterate",
        lambda f: _iterate(
            f,
            method="sgm",
            regularizer_norm="kl",
            initialization="data",
            rec_to_save="all",
            return_pointwise_sol=True,
            idx_in_focus=1,
        ),
        "sgm, history summed over realizations",
    ),
    Case(
        f"{OPTIMIZER}:iterate",
        lambda f: _iterate(
            f,
            max_iter=15,
            stop_tol=0.6,
            stop_criteria="step_size",
            step_size_search=0.5,
        ),
        "early stop",
    ),
    Case(
        f"{OPTIMIZER}:iterate",
        lambda f: _iterate(f, step_size_search="line_search", verbose=True),
        "line search, verbose",
    ),
    Case(
        f"{OPTIMIZER}:iterate",
        lambda f: _raised(lambda: _iterate(f, step_size_search=0.0)),
        "zero step fails",
    ),
    Case(
        f"{OPTIMIZER}:update_fft",
        lambda f: [
            _update(f, method=method, initial_step_size=1.0)
            for method in ("osl", "abs", "sgm")
        ],
        "multiplicative methods",
    ),
    Case(
        f"{OPTIMIZER}:update_fft",
        lambda f: [_update(f, method="sgm", regularizer=norm) for norm in ("l1", "kl")],
        "sgm, each regularizer",
    ),
    Case(
        f"{OPTIMIZER}:update_fft",
        lambda f: [
            _update(f, method=method, step_size_search=search)
            for method in ("pgd", "sgp")
            for search in ("fixed", "rl", "line_search", 0.4)
        ],
        "gradient methods, each step size search",
    ),
    Case(
        f"{OPTIMIZER}:update_fft",
        lambda f: _update(f, normalize=False, line_search_params=None, lam=0.5, m=0),
        "no normalization, default truncation",
    ),
    Case(
        f"{OPTIMIZER}:update_fft",
        lambda f: [
            _raised(lambda: _update(f, method="other")),
            _raised(lambda: _update(f, regularizer="other", method="sgm")),
            _raised(lambda: _update(f, step_size_search="other")),
        ],
        "invalid input fails",
    ),
    Case(
        f"{OPTIMIZER}:compute_step_size_line_search",
        lambda f: _line_search(f),
        "accepted at once",
    ),
    Case(
        f"{OPTIMIZER}:compute_step_size_line_search",
        lambda f: _line_search(f, normalize=False, delta=1e-3, theta=0.5),
        "backtracks until accepted",
    ),
    Case(
        f"{OPTIMIZER}:compute_step_size_line_search",
        lambda f: _line_search(
            f, normalize=False, delta=1e-5, theta=0.5, max_iter=6, verbose=True
        ),
        "never accepted, verbose",
    ),
    Case(
        f"{OPTIMIZER}:compute_step_size_line_search",
        lambda f: _line_search(
            f,
            normalize=False,
            regularizer="l1",
            lam=3.0,
            weight=positive_tensor(16, 16, 1, seed=3),
            initial_step_size=torch.tensor([0.9, 0.5, 0.2]),
            delta=0.01,
            theta=0.7,
        ),
        "l1, weight map, initial step per realization",
    ),
    Case(
        f"{OPTIMIZER}:batch_reconstruction",
        lambda f: _batches(f),
        "confocal",
    ),
    Case(
        f"{OPTIMIZER}:batch_reconstruction",
        lambda f: _batches(
            f,
            "s2ISM",
            weight_map=positive_tensor(16, 16, seed=3),
            rec_to_save="all",
            step_size_search="line_search",
            verbose=True,
        ),
        "s2ISM, full-size weight map, verbose",
    ),
    Case(
        f"{OPTIMIZER}:batch_reconstruction",
        lambda f: _batches(f, "MID", method="sgm", return_pointwise_sol=True),
        "MID, summed over realizations",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: _adam(f),
        "l2, history",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: _adam(f, rec_to_save="last", verbose=True, num_iter=12),
        "last, verbose",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: [
            _adam(f, regularizer=norm, initialization=start)
            for norm in ("l1", "kl")
            for start in ("flat", "data")
        ],
        "each regularizer and initialization",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: _adam(f, k=3, sum_rep=True, full_KL=True, careful=False),
        "summed realizations, full, not careful",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: _adam(f, k=2, sum_rep=True, binomial=True),
        "binomial resampling",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: _adam(f, fourier=True),
        "fourier",
    ),
    Case(
        f"{ADAM}:adam_optimize_obj",
        lambda f: [
            _raised(lambda: _adam(f, rec_to_save="other")),
            _raised(lambda: _adam(f, k=3)),
        ],
        "unknown saving mode and several realizations fail",
    ),
    Case(
        f"{ADAM}:WarmupWithPlateau",
        lambda f: _warmup(_package(f, "adam_optimizer"), (5.0, 4.0, 3.0, 2.0, 1.0)),
        "warmup only",
    ),
    Case(
        f"{ADAM}:WarmupWithPlateau",
        lambda f: _warmup(
            _package(f, "adam_optimizer"),
            (5.0, 6.0, 6.5, 7.0, 7.5, 8.0),
            careful=True,
        ),
        "careful, loss rises",
    ),
    Case(
        f"{DISCREPANCY}:make_variance_reconstructor",
        lambda f: (
            f(varreg_counts(), varreg_psf(channels=None), max_iter=2, process="cpu"),
            _raised(
                lambda: f(varreg_counts(), varreg_psf(channels=None), process="cpu")(
                    0.1
                )
            ),
        ),
        "builds a callable, call fails",
    ),
    Case(
        f"{DISCREPANCY}:monte_carlo_deltaE",
        lambda f: f(positive_tensor(4, 3), S=300, device="cpu"),
        "tensor on the cpu",
    ),
    Case(
        f"{DISCREPANCY}:monte_carlo_deltaE",
        lambda f: f(torch.tensor([[0.05, 3.0]]), S=200, device=torch.device("cpu")),
        "small intensity",
    ),
    Case(
        f"{DISCREPANCY}:monte_carlo_deltaE",
        lambda f: _raised(lambda: f(torch.ones(3), S=10, device="cpu")),
        "one dimension fails",
    ),
    Case(
        f"{DISCREPANCY}:discrepancy_model",
        lambda f: f(positive_tensor(5, 3), positive_tensor(3, 4, seed=2)),
        "",
    ),
    Case(
        f"{DISCREPANCY}:discrepancy_model",
        lambda f: [
            _raised(lambda: f(torch.ones(3), torch.ones(1, 4))),
            _raised(lambda: f(torch.ones(2, 3), torch.ones(2, 4))),
            _raised(lambda: f(torch.ones(2, 3), torch.ones(3, 5))),
        ],
        "invalid shapes fail",
    ),
    Case(
        f"{DISCREPANCY}:fit_discrepancy_model_multi",
        lambda f: _fit(f, iters=8),
        "fitted parameters",
    ),
    Case(
        f"{DISCREPANCY}:fit_discrepancy_model_multi",
        lambda f: _fit(f, iters=8, check=True, lr=0.05, device="cpu"),
        "fitted parameters and losses",
    ),
    Case(
        f"{DISCREPANCY}:fit_c_from_simulated_y",
        lambda f: _printed(
            lambda: f(
                positive_tensor(5, 2) * 3,
                mc_S=200,
                device="cpu",
                fit_iters=6,
                verbose=True,
            )
        ),
        "verbose",
    ),
    Case(
        f"{DISCREPANCY}:fit_c_from_simulated_y",
        lambda f: _raised(lambda: f(torch.ones(4), device="cpu")),
        "one dimension fails",
    ),
    Case(
        f"{DISCREPANCY}:compute_D_lambda",
        lambda f: f(
            positive_tensor(1, 16, 16, 3),
            torch.from_numpy(varreg_counts()).float(),
            _discrepancy_otf(f),
        ),
        "default epsilon",
    ),
    Case(
        f"{DISCREPANCY}:compute_D_lambda",
        lambda f: f(
            positive_tensor(1, 16, 16, 3),
            torch.from_numpy(varreg_counts()).float(),
            _discrepancy_otf(f),
            eps=1e-3,
        ),
        "epsilon given",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP",
        lambda f: _nedp(f),
        "known target",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP",
        lambda f: _nedp(f, return_all=True, verbose=True, Delta_GT=900.0),
        "all candidates, verbose",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP",
        lambda f: _nedp(f, return_all_iters=True, Delta_GT=0.0),
        "all iterations",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP",
        lambda f: _raised(lambda: _nedp(f, Delta_GT=None)),
        "estimated target fails",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP",
        lambda f: _raised(lambda: _nedp(f, lambdas=())),
        "no candidates fail",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP_bisection",
        lambda f: _bisection(f),
        "bracket found",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP_bisection",
        lambda f: _bisection(
            f, return_all=True, verbose=True, rel_tol=1e-9, max_iters=3
        ),
        "not converged, all evaluations",
    ),
    Case(
        f"{DISCREPANCY}:choose_lambda_by_NEDP_bisection",
        lambda f: _raised(lambda: _bisection(f, delta=1e9, expand_tries=1)),
        "no bracket, fallback fails",
    ),
    Case(
        f"{WHITENESS}:VarianceReconstructor",
        lambda f: _reconstructor(f),
        "default parameters",
    ),
    Case(
        f"{WHITENESS}:VarianceReconstructor",
        lambda f: _reconstructor(
            f,
            method="sgm",
            regularizer_norm="l1",
            stop_criteria="rel_diff",
            stop_tol=1e-2,
        ),
        "sgm, l1, early stop",
    ),
    Case(
        f"{WHITENESS}:make_variance_reconstructor",
        lambda f: _reconstructor(f),
        "default parameters",
    ),
    Case(
        f"{WHITENESS}:make_variance_reconstructor",
        lambda f: _reconstructor(
            f,
            line_search_params=(5, 1),
            step_size_search="line_search",
            rec_to_save="all",
        ),
        "line search, history",
    ),
    Case(
        f"{WHITENESS}:T_func",
        lambda f: f(torch.tensor([0.1, 1.0, 5.0, 20.0])),
        "",
    ),
    Case(
        f"{WHITENESS}:V_func",
        lambda f: f(torch.tensor([0.1, 1.0, 5.0, 20.0])),
        "",
    ),
    Case(
        f"{WHITENESS}:compute_z_variable",
        lambda f: f(*_z_inputs(f)),
        "default",
    ),
    Case(
        f"{WHITENESS}:compute_z_variable",
        lambda f: f(*_z_inputs(f), masked=True, remove_border=True),
        "masked, border removed",
    ),
    Case(
        f"{WHITENESS}:compute_z_variable",
        lambda f: f(*_z_inputs(f), remove_border=2),
        "border width given",
    ),
    Case(
        f"{WHITENESS}:compute_z_variable",
        lambda f: _raised(lambda: f(*_z_inputs(f), remove_border=0)),
        "zero border fails",
    ),
    Case(
        f"{WHITENESS}:compute_z_variable",
        lambda f: _raised(lambda: f(*_z_inputs(f), remove_border="two")),
        "border of another type fails",
    ),
    Case(
        f"{WHITENESS}:compute_whiteness_measure",
        lambda f: f(positive_tensor(16, 16, 3) - 1.25),
        "",
    ),
    Case(
        f"{WHITENESS}:choose_lambda_by_WP",
        lambda f: _whiteness_search(f),
        "best lambda",
    ),
    Case(
        f"{WHITENESS}:choose_lambda_by_WP",
        lambda f: _whiteness_search(f, masked=True, remove_border=3, verbose=True),
        "masked, border, verbose",
    ),
    Case(
        f"{WHITENESS}:choose_lambda_by_WP",
        lambda f: _whiteness_search(f, return_whiteness_list=True),
        "whiteness values",
    ),
    Case(
        f"{WHITENESS}:choose_lambda_by_WP",
        lambda f: _whiteness_search(f, return_all_solutions=True),
        "all solutions",
    ),
    Case(
        f"{WHITENESS}:choose_lambda_by_WP",
        lambda f: _whiteness_search(f, return_all_iters=True, idx_in_focus=0),
        "all iterations",
    ),
    Case(
        f"{WHITENESS}:choose_lambda_by_WP",
        lambda f: _whiteness_search(f, return_z_variables=True),
        "residuals and predictions",
    ),
]

SKIPPED: dict[str, str] = {
    f"{DISCREPANCY}:compute_Delta_NE_lambda": "fails with the old code too: NameError, name 'forward' is not defined",
}
