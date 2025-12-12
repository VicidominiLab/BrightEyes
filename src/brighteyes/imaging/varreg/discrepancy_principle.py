import torch
import numpy as np
from typing import Callable, Optional, Tuple

from .loss import fidelity_loss
from .sgm_optimizer import variance_reconstruction
from .utils import forward
import torch.nn as nn
from torch.optim import Adam


def estimate_tau_montecarlo(dataset: torch.Tensor,
                            psf: torch.Tensor,
                            recon_obj: torch.Tensor,
                            n_mc: int = 30,
                            device: Optional[torch.device] = None,
                            eps: float = 1e-6) -> float:
    """
    Estimate expected KL discrepancy E[ KL(Y, mu) ] by Monte-Carlo sampling,
    using `recon_obj` as the surrogate for the true object whose forward model
    is mu = A(recon_obj).

    Parameters
    - dataset: observed data (torch.Tensor) shape like Nx x Ny x Ch x K
    - psf: PSF (will be passed to fidelity_loss / forward)
    - recon_obj: object estimate (Nx x Ny x K) used to compute mu = A(recon_obj)
    - n_mc: number of Monte-Carlo samples
    - device: torch device for sampling

    Returns
    - tau: scalar float, estimated expected KL for Poisson noise
    """
    if device is None:
        device = recon_obj.device

    # compute forward model mu = A(recon_obj)
    # `forward` returns the predicted data with same dims as dataset
    with torch.no_grad():
        mu = forward(psf, recon_obj) if not torch.is_complex(psf) else forward(psf, recon_obj)

    mu = mu.to(device)

    kappas = []
    for _ in range(n_mc):
        # sample Poisson observation Y ~ Poisson(mu)
        # torch.poisson expects rates (mu) and returns integer tensor
        Y = torch.poisson(mu)
        # compute KL(Y, mu) via fidelity_loss: fidelity_loss(Y, recon_obj, psf)
        kl = fidelity_loss(Y, recon_obj, psf, eps=eps).cpu().float()
        # fidelity_loss returns a tensor possibly with leading batch dimension; take mean if needed
        if kl.numel() > 1:
            kl_val = kl.mean().item()
        else:
            kl_val = float(kl.item())
        kappas.append(kl_val)

    return float(np.mean(kappas))


def find_lambda_by_discrepancy(dataset: torch.Tensor,
                               psf: torch.Tensor,
                               reconstructor: Callable[[float], torch.Tensor],
                               lambda_min: float = 1e-6,
                               lambda_max: float = 1.0,
                               tol: float = 0.05,
                               max_outer_iters: int = 12,
                               tau: Optional[float] = None,
                               tau_estimate_mode: str = 'montecarlo',
                               mc_samples: int = 40,
                               device: Optional[torch.device] = None,
                               verbose: bool = False) -> Tuple[float, torch.Tensor, float]:
    """
    Find regularization parameter `lambda` that satisfies the discrepancy principle
    for Poisson data using a bisection search.

    The function expects a `reconstructor(lambda_value)` callable that performs
    the full reconstruction (returns an object tensor) for the given lambda.

    Parameters
    - dataset, psf: observed data and PSF (torch tensors). Shapes should match
      what your reconstructor expects (e.g. Nx x Ny x Ch x K for dataset and
      Mx x My x Ch for psf).
    - reconstructor: callable lambda -> reconstructed object tensor (Nx x Ny x K)
    - lambda_min, lambda_max: search interval for lambda (positive)
    - tol: relative tolerance on fidelity: stop when |fidelity - tau|/tau < tol
    - max_outer_iters: max bisection steps
    - tau: optional user-provided target discrepancy. If None, it will be
      estimated according to `tau_estimate_mode`.
    - tau_estimate_mode: 'montecarlo' or 'manual'
    - mc_samples: Monte-Carlo draws when estimating tau

    Returns (lambda_star, obj_star, fidelity_star)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = dataset.to(device)
    psf = psf.to(device)

    # Estimate target discrepancy tau if not provided
    if tau is None:
        if tau_estimate_mode == 'montecarlo':
            if verbose:
                print('Estimating target discrepancy tau with Monte-Carlo...')
            # get an initial reconstruction with a small regularization to approximate mu
            lambda_init = lambda_min
            recon0 = reconstructor(lambda_init)
            recon0 = recon0.to(device)
            tau = estimate_tau_montecarlo(dataset, psf, recon0, n_mc=mc_samples, device=device)
            if verbose:
                print(f'Estimated tau = {tau:.4g}')
        else:
            raise ValueError('Unknown tau_estimate_mode. Provide manual tau or use montecarlo.')

    lo = float(lambda_min)
    hi = float(lambda_max)
    lambda_star = None
    obj_star = None
    fidelity_star = None

    for it in range(max_outer_iters):
        mid = 0.5 * (lo + hi)
        if verbose:
            print(f'Outer iter {it}: testing lambda = {mid:.4g} in [{lo:.4g}, {hi:.4g}]')

        # run reconstructor for mid
        obj_mid = reconstructor(mid)
        obj_mid = obj_mid.to(device)

        # compute fidelity KL(dataset, A(obj_mid)) using fidelity_loss
        fidelity = fidelity_loss(dataset, obj_mid, psf)
        # fidelity_loss may return tensor; reduce to scalar
        if isinstance(fidelity, torch.Tensor):
            fidelity_val = float(fidelity.mean().item())
        else:
            fidelity_val = float(fidelity)

        if verbose:
            print(f'  fidelity = {fidelity_val:.6g}, target tau = {tau:.6g}')

        # check convergence
        rel_err = abs(fidelity_val - tau) / (tau + 1e-12)
        if rel_err < tol:
            lambda_star = mid
            obj_star = obj_mid
            fidelity_star = fidelity_val
            if verbose:
                print(f'Converged: lambda = {lambda_star:.4g}, fidelity = {fidelity_star:.6g}')
            break

        # monotonic assumption: fidelity increases with lambda
        if fidelity_val > tau:
            # fidelity too large => decrease lambda
            hi = mid
        else:
            # fidelity too small => increase lambda
            lo = mid

        # keep last mid as fallback
        lambda_star = mid
        obj_star = obj_mid
        fidelity_star = fidelity_val

    return lambda_star, obj_star, fidelity_star


# Example reconstructor wrapper using the repository `variance_reconstruction`:
def make_variance_reconstructor(dset: np.ndarray,
                                psf: np.ndarray,
                                alpha: int = 1,
                                weight_map=1,
                                tol = 1e-5,
                                max_iter: int = 50,
                                method: str = 'sgp',
                                regularizer: str = 'l2',
                                initialization: str = 'flat',
                                normalize: bool = True,
                                process: str = 'gpu',
                                rec_to_save: str = 'last',
                                binomial: bool = False,
                                step_size_search: str = 'fixed',
                                line_search_params=None,
                                device: Optional[torch.device] = None) -> Callable[[float], torch.Tensor]:
    """
    Create a callable that takes `lambda_value` and returns the reconstructed object
    using the `variance_reconstruction` function in this repo.

    Note: inputs `dset` and `psf` are numpy arrays and will be converted inside
    `variance_reconstruction` to torch tensors. This wrapper avoids repeated
    preprocessing outside the reconstructor.
    """
    
    if line_search_params is None:
        line_search_params = (10, 2)

    def recon(lambda_value: float) -> torch.Tensor:
        obj = variance_reconstruction(dset, psf, lambda_value, alpha_value=alpha, weight_map=weight_map,
                                      max_iter=max_iter, tol=tol, method=method, process=process,
                                      regularizer=regularizer, initialization=initialization,
                                      normalize=normalize, rec_to_save=rec_to_save, binomial=binomial,
                                      step_size_search=step_size_search, line_search_params=line_search_params)
        return obj

    return recon


# ------------------------------------------------------------------
# Functions for estimating channel-wise parametric discrepancy model c
# ------------------------------------------------------------------


def monte_carlo_deltaE(y_values: torch.Tensor, S: int = 10000, device: Optional[torch.device] = None):
    """
    Monte-Carlo estimate of delta^E for candidate intensities. Expects
    `y_values` with shape (J, Ch): J simulated intensity-vectors, each with Ch channels.

    Returns (deltaE_mean, deltaE_var) of shape (J,).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    y_values = y_values.to(device)
    if y_values.ndim != 2:
        raise ValueError('y_values must be 2D with shape (J, Ch)')

    J, Ch = y_values.shape
    deltaE_mean = torch.zeros((J, Ch), device=device)
    deltaE_var = torch.zeros((J, Ch), device=device)

    # loop over J intensity-vectors; no loop over channels
    for j in range(J):
        y_vec = y_values[j]  # (Ch,)
        samples = torch.poisson(y_vec.unsqueeze(0).expand(S, Ch))
        k = samples.float()
        y_exp = y_vec.unsqueeze(0).expand(S, Ch).float()

        mask = k > 0
        contrib = torch.zeros_like(k, dtype=torch.float32)
        if mask.any():
            contrib[mask] = k[mask] * torch.log(k[mask] / y_exp[mask]) - k[mask] + y_exp[mask]
        if (~mask).any():
            contrib[~mask] = y_exp[~mask]

        fvals = contrib  # (S,Ch)
        deltaE_mean[j, :] = fvals.mean(dim=0)
        deltaE_var[j, :] = fvals.var(dim=0, unbiased=True)

    return deltaE_mean, deltaE_var


def discrepancy_model(y_mat: torch.Tensor, c_mat: torch.Tensor) -> torch.Tensor:
    """
    Vectorized discrepancy model that sums channel contributions internally.

    y_mat: (J, Ch) tensor of intensities
    c_mat: (Ch, 4) tensor of channel parameters

        Returns tensor (J, Ch) of per-sample per-channel contributions. The loss
        during fitting sums over both sample index `j` and channel `ch`.
    """
    # y_mat: (J,Ch), c_mat: (Ch,4)
    if y_mat.ndim != 2:
        raise ValueError('y_mat must be 2D (J,Ch)')
    J, Ch = y_mat.shape
    if c_mat.ndim != 2 or c_mat.shape[0] != Ch or c_mat.shape[1] != 4:
        raise ValueError('c_mat must have shape (Ch,4)')

    # Extract channel params and broadcast to shape (J,Ch)
    # c1..c4: (1,Ch)
    c1 = c_mat[:, 0].unsqueeze(0)
    c2 = c_mat[:, 1].unsqueeze(0)
    c3 = c_mat[:, 2].unsqueeze(0)
    c4 = c_mat[:, 3].unsqueeze(0)

    y = y_mat
    num = y ** 2 + c1 * y + c2
    den = 12 * (y ** 3) + c3 * (y ** 2) + c4 * y - 2 * c2
    fraction = num / torch.clamp(den, min=1e-12)  # (J,Ch)
    # per-channel contribution (J,Ch)
    return 0.5 + fraction


# Single-channel fitter removed: workflow uses multi-channel joint fitting only.


def fit_discrepancy_model_multi(y_grid: torch.Tensor,
                                deltaE_mean: torch.Tensor,
                                deltaE_var: torch.Tensor,
                                lr: float = 1e-2,
                                iters: int = 10000,
                                device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Fit channel-wise c parameters jointly from a grid of intensities.

    - y_grid: tensor shape (J, Ch)
    - deltaE_mean: tensor shape (J,)
    - deltaE_var: tensor shape (J,)

    Returns c_hats tensor shape (Ch, 4)
    """
    if device is None:
        device = y_grid.device
    y = y_grid.to(device)
    target = deltaE_mean.to(device)
    var = deltaE_var.to(device)

    J, Ch = y.shape
    # parameter tensor shape (Ch, 4)
    c = nn.Parameter(torch.zeros((Ch, 4), device=device))
    opt = Adam([c], lr=lr)
    w = 1.0 / torch.clamp(var, min=1e-12)

    for _ in range(iters):
        opt.zero_grad()
        # compute predicted f_j per-channel in vectorized form
        # y: (J,Ch), c: (Ch,4)
        preds = discrepancy_model(y, c)  # (J,Ch) per-sample per-channel
        loss = torch.sum(w * (preds - target) ** 2)
        loss.backward()
        opt.step()

    return c.detach()


# The previous helper that derived per-channel c from an initial reconstruction was
# removed: in the current workflow we estimate channel-wise parameters `c` from
# simulated intensity vectors directly using `fit_c_from_simulated_y` which
# performs a joint multi-channel fit. This avoids per-channel Python loops.


def compute_D_lambda(obj: torch.Tensor, img: torch.Tensor, psf_fft: torch.Tensor, eps: float = 1e-6) -> float:
    """Compute left-hand side D(λ; i) = sum_k KL(i_k, o_k * h)
    obj: reconstructed object (Nx x Ny x K)
    img: observed data (Nx x Ny x Ch x K) or (Nx x Ny x K) compatible with fidelity_loss
    psf_fft: PSF or OTF used by forward
    """
    # Use fidelity_loss to compute KL across whole image
    kl = fidelity_loss(img, obj, psf_fft, eps=eps)
    if isinstance(kl, torch.Tensor):
        return float(kl.sum().item())
    else:
        return float(kl)


def compute_Delta_NE_lambda(obj: torch.Tensor, psf_fft: torch.Tensor, c_hats: torch.Tensor) -> float:
    """Compute RHS ΔNE(λ) = sum_j δ^E(y_j) using channel-wise c_hats
    c_hats: tensor shape (Ch,4)
    """
    with torch.no_grad():
        y_pred = forward(psf_fft, obj)

    # y_pred shape handling: collapse spatial dims and repetition into N x Ch
    if y_pred.ndim == 4:
        Nx, Ny, Ch, K = y_pred.shape
        y_mat = y_pred.reshape(-1, Ch)
    elif y_pred.ndim == 3:
        Nx, Ny, Ch = y_pred.shape
        y_mat = y_pred.reshape(-1, Ch)
    else:
        raise ValueError('Unsupported y_pred ndim in compute_Delta_NE_lambda')

    # vectorized per-sample contributions (channels summed inside): (Npix,)
    per_sample = discrepancy_model(y_mat, c_hats.to(y_mat.device))
    total = float(per_sample.sum().item())
    return total


def choose_lambda_by_NEDP(solver_fn: Callable[[float], torch.Tensor],
                         lambdas,
                         img: torch.Tensor,
                         psf_fft: torch.Tensor,
                         c_hats: torch.Tensor,
                         device: Optional[torch.device] = None,
                         verbose: bool = False):
    """
    Search over candidate lambdas and pick the one minimizing |D(λ)-ΔNE(λ)|.
    solver_fn(λ) -> reconstructed object o(λ)
    lambdas: iterable of candidate λ values
    c_hats: tensor (Ch,4) estimated previously (kept fixed during search)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_diff = float('inf')
    best_lambda = None
    best_obj = None

    for lam in lambdas:
        if verbose:
            print(f'Testing lambda={lam}')
        obj_lam = solver_fn(lam)
        DL = compute_D_lambda(obj_lam, img, psf_fft)
        DR = compute_Delta_NE_lambda(obj_lam, psf_fft, c_hats)
        diff = abs(DL - DR)
        if verbose:
            print(f'  D_L={DL:.6g}, D_R={DR:.6g}, |diff|={diff:.6g}')
        if diff < best_diff:
            best_diff = diff
            best_lambda = lam
            best_obj = obj_lam

    if best_lambda is None:
        raise ValueError('No lambdas were provided or evaluation failed.')
    return float(best_lambda), best_obj, float(best_diff)


def choose_lambda_by_NEDP_bisection(solver_fn: Callable[[float], torch.Tensor],
                                     lambda_lo: float,
                                     lambda_hi: float,
                                     img: torch.Tensor,
                                     psf_fft: torch.Tensor,
                                     c_hats: torch.Tensor,
                                     max_iters: int = 20,
                                     rel_tol: float = 0.05,
                                     expand_tries: int = 5,
                                     verbose: bool = False) -> Tuple[float, torch.Tensor, float]:
    """
    Bisection search for lambda solving D(lambda) - DeltaNE(lambda) = 0.

    If no sign change is found in [lambda_lo, lambda_hi], the function will
    try to expand the interval geometrically for `expand_tries` times. If
    still no sign change, it falls back to a small grid search and returns the
    best lambda found.

    Returns (lambda_star, obj_star, final_abs_diff).
    """
    dev = img.device

    def f(lam: float):
        obj = solver_fn(lam)
        DL = compute_D_lambda(obj, img, psf_fft)
        DR = compute_Delta_NE_lambda(obj, psf_fft, c_hats)
        return (DL - DR, DL, DR, obj)

    lo = float(lambda_lo)
    hi = float(lambda_hi)

    # First, perform a coarse log-spaced grid to try to find a sign change bracket
    cache = {}
    def eval_cached(lam: float):
        lam = float(lam)
        if lam in cache:
            return cache[lam]
        val = f(lam)
        cache[lam] = val
        return val

    n_coarse = 9
    coarse_lams = np.logspace(np.log10(lo), np.log10(hi), num=n_coarse)
    fvals = [eval_cached(l)[0] for l in coarse_lams]

    bracket_found = False
    bracket_lo = None
    bracket_hi = None
    for k in range(len(coarse_lams) - 1):
        if fvals[k] == 0:
            bracket_lo = bracket_hi = coarse_lams[k]
            bracket_found = True
            break
        if fvals[k] * fvals[k + 1] < 0:
            bracket_lo = coarse_lams[k]
            bracket_hi = coarse_lams[k + 1]
            bracket_found = True
            break

    if not bracket_found:
        # Try expanding interval geometrically and repeat coarse search
        expand_factor = 2.0
        tries = 0
        while not bracket_found and tries < expand_tries:
            if verbose:
                print('No coarse bracket; expanding interval and re-checking...')
            lo = max(1e-12, lo / expand_factor)
            hi = hi * expand_factor
            coarse_lams = np.logspace(np.log10(lo), np.log10(hi), num=n_coarse)
            fvals = [eval_cached(l)[0] for l in coarse_lams]
            for k in range(len(coarse_lams) - 1):
                if fvals[k] == 0:
                    bracket_lo = bracket_hi = coarse_lams[k]
                    bracket_found = True
                    break
                if fvals[k] * fvals[k + 1] < 0:
                    bracket_lo = coarse_lams[k]
                    bracket_hi = coarse_lams[k + 1]
                    bracket_found = True
                    break
            tries += 1

    if not bracket_found:
        # Fall back to grid search in the supplied interval
        if verbose:
            print('No bracket found after expansions; falling back to grid search')
        lambdas = np.logspace(np.log10(lambda_lo), np.log10(lambda_hi), num=25)
        best_lambda, best_obj, best_diff = choose_lambda_by_NEDP(solver_fn, lambdas, img, psf_fft, c_hats, device=dev, verbose=verbose)
        return best_lambda, best_obj, best_diff

    # Use the found bracket for bisection
    assert bracket_found and (bracket_lo is not None) and (bracket_hi is not None)
    lo = float(bracket_lo)
    hi = float(bracket_hi)
    f_lo, DL_lo, DR_lo, obj_lo = eval_cached(lo)
    f_hi, DL_hi, DR_hi, obj_hi = eval_cached(hi)

    # Bisection loop
    mid = 0.5 * (lo + hi)
    f_mid, DL_mid, DR_mid, obj_mid = eval_cached(mid)
    for i in range(max_iters):
        mid = 0.5 * (lo + hi)
        f_mid, DL_mid, DR_mid, obj_mid = eval_cached(mid)
        if verbose:
            print(f'Iter {i}: lam_mid={mid:.6g}, f_mid={f_mid:.6g}, DL={DL_mid:.6g}, DR={DR_mid:.6g}')

        # stop when relative error wrt DR is small
        denom = max(abs(DR_mid), 1e-12)
        if abs(f_mid) / denom < rel_tol:
            return mid, obj_mid, abs(f_mid)

        # choose side
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    # return last mid if not converged
    return mid, obj_mid, abs(f_mid)


def fit_c_from_simulated_y(y_grid: torch.Tensor,
                           mc_S: int = 20000,
                           device: Optional[torch.device] = None,
                           fit_iters: int = 5000,
                           verbose: bool = False) -> torch.Tensor:
    """
    Fit c parameters from a simulated grid of intensities per channel.

    Parameters
    - y_grid: Tensor of shape (Ch, J) containing J candidate intensities per channel,
              or shape (J,) for a single channel.
    - mc_S: number of Monte-Carlo Poisson samples per intensity
    - device: torch device
    - fit_iters: iterations for fitting c via Adam

    Returns
    - c_hats: tensor shape (Ch, 4) or (4,) for single channel
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    y_grid = y_grid.to(device)

    # Expect y_grid to be shape (J, Ch): J simulated intensity-vectors
    if y_grid.ndim != 2:
        raise ValueError('y_grid must be 2D tensor with shape (J, Ch)')

    J, Ch = y_grid.shape
    if verbose:
        print(f'Fitting joint model for {Ch} channels using {J} intensity samples')
    # monte carlo returns deltaE per sample where KL is summed across channels
    delta_mean, delta_var = monte_carlo_deltaE(y_grid, S=mc_S, device=device)
    # joint fit of all channel c parameters (returns (Ch,4))
    c_hats = fit_discrepancy_model_multi(y_grid, delta_mean, delta_var, iters=fit_iters, device=device)
    return c_hats



