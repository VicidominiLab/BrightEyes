import numpy as np

import matplotlib.pyplot as plt

import sys

import torch

from scipy.stats import qmc

import brighteyes_ism.simulation.PSF_sim as sim

from brighteyes_ism.simulation.utils import partial_convolution

from s2ism.psf_estimator import GridFinder

from scipy.optimize import differential_evolution, dual_annealing

from scipy.special import rel_entr

import string

import warnings

#%%

FIXED_ZERNIKE = 3

iter_annealing = 1


def disk(grid, radius: float, fourier: bool = True, process: str = 'cpu'):
    """
    It calculates a binary mask with the shape of a disk.

    Parameters
    ----------
    grid : int
        Number of pixels of the simulation space, assumed squared.
    radius : float
        Radius of the disk in pixel units.
    fourier : bool
        If true, the function returns the fft of the disk.

    Returns
    -------
    disk : np.ndarray (nx x nx)
        Image of the binary mask.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() and process == 'gpu' else "cpu")

    x = (torch.arange(-(grid.Nx // 2), grid.Nx // 2 + np.mod(grid.Nx, 2)) * grid.pxsizex).float()
    z = (torch.arange(-(grid.Nz // 2), grid.Nz // 2 + np.mod(grid.Nz, 2)) * grid.pxsizez).float()

    zz, xx, yy = torch.meshgrid(z, x, x, indexing='ij')
    r = torch.sqrt(xx ** 2 + yy ** 2 + zz**2)

    sphere = torch.where(r < radius, 1, 0)

    if fourier is True:
        sphere = torch.fft.fftn(sphere)

    return torch.squeeze(sphere).to(device)


def fit_detector(data, grid, exPar, emPar):

    grid_fit = GridFinder(grid)
    
    grid_fit.estimate(data, exPar.wl, emPar.wl, exPar.na)
    
    grid_fit.Print()
    
    spad = sim.custom_detector(grid_fit, 'cpu')
    
    return spad


def forward(zer_coeff, expar, empar, grid, spad=None, obj=None, dark_noise=None, pxdwelltime=None, process='cpu'):

    nch = grid.Nch
    
    expar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), torch.as_tensor(zer_coeff)])
    
    empar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), torch.as_tensor(zer_coeff)])

    if grid.Nz > 1:
        psf, *_ = sim.SPAD_PSF_3D(grid, expar, empar, spad=spad, process=process, output = 'tensor')
    else:
        psf, *_ = sim.SPAD_PSF_2D(grid, expar, empar, spad=spad, process=process, output = 'tensor')

    if obj is not None:
        alphabet = list(string.ascii_lowercase)
        obj_axes = ''.join(alphabet[:obj.ndim])
        psf_axes = ''.join(alphabet[:psf.ndim])
        dset = partial_convolution(obj, psf, obj_axes, psf_axes, obj_axes, fourier=(1, 0))
    else:
        dset = psf
    
    if dark_noise is not None:
        for c in range(nch):
            dset[..., c] += dark_noise[c]*pxdwelltime
    
    dset /= dset.sum()
    
    return dset


def vectorized_forward(zer_coeff, expar, empar, grid, spad=None, obj=None, dark_noise=None, pxdwelltime=None,
                       process='cpu'):

    device = torch.device("cuda:0" if torch.cuda.is_available() and process == 'gpu' else "cpu")

    n_agents = zer_coeff.shape[0]
    datasets = []

    for n in range(n_agents):
        datasets.append(forward(zer_coeff[n], expar, empar, grid, spad, obj, process=process))

    return torch.stack(datasets, dim=0).to(device)


def loss_function(x0, data, expar, empar, grid, spad, obj=None, fit_mag: bool = False, loss: str = 'mse',
                  vectorized: bool = False, process: str = 'cpu'):

    if fit_mag is True:
        grid.M = x0[0]
        zer_coeff = x0[1:]
    else:
        zer_coeff = x0.clone()

    if vectorized is True:
        estimate = vectorized_forward(zer_coeff, expar, empar, grid, spad, obj, process=process)
    else:
        estimate = forward(zer_coeff, expar, empar, grid, spad, obj, process=process)

    axis = tuple(-np.arange(1, data.ndim+1))

    if loss == 'mse':
        loss = torch.sum(torch.abs(data - estimate)**2, axis=axis)
    elif loss == 'mae':
        loss = torch.sum(torch.abs(data - estimate), axis=axis)
    elif loss == 'kl':
        loss = torch.sum(rel_entr(data, estimate), axis=axis)
    else:
        raise Exception("Loss function unknown.")

    loss_value = loss.cpu().detach().numpy()

    return loss


def fit_psf(data, expar, empar, grid, spad, zer_max: int = 15, bead_radius: float = 0, loss: str = 'mae',
            num_starts: int = 10, max_iter: int = 200, bound: int = 2, initialization: str = 'sobol',
            verbose: str = True, process: str = 'cpu'):

    zer_to_fit = zer_max - FIXED_ZERNIKE  # neglect piston, tip, tilt

    excitation = expar.copy()
    emission = empar.copy()

    zernike_idx = np.arange(zer_max)
    excitation.abe_index = zernike_idx
    emission.abe_index = zernike_idx

    if bead_radius > 0:
        obj = disk(grid, bead_radius, process=process)
    else:
        obj = None

    kwargs = {'data': torch.as_tensor(data),
              'expar': excitation,
              'empar': emission,
              'grid': grid,
              'spad': spad,
              'obj': obj,
              'fit_mag': False,
              'loss': loss,
              'vectorized': False,
              'process': process
              }

    start_points = generate_starting_points(zer_to_fit, num_starts, bound, initialization)

    history = torch.zeros((num_starts, max_iter))
    result = torch.zeros((num_starts, zer_to_fit))

    for i, start in enumerate(start_points):
        if verbose is True:
            print(f"\nOptimization: {i + 1}/{len(start_points)}")

        x0, h0 = minimizer(start, max_iter, verbose=verbose, **kwargs)

        result[i] = x0.clone()
        history[i] = h0.clone()

    idx = torch.argmin(history[:, -1])
    best_result = result[idx]

    # Print final optimized parameters
    if verbose is True:
        print("\nOptimized parameters:\n", best_result.detach().numpy())

    return best_result, history


def generate_starting_points(ndim, num_starts, bound, initialization):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if initialization == 'lhs':
            lhs_sampler = qmc.LatinHypercube(d=ndim)
            samples = lhs_sampler.random(n=num_starts - 1)  # Generate LHS samples
        elif initialization == 'sobol':
                sobol_sampler = qmc.Sobol(d=ndim)
                samples = sobol_sampler.random(n=num_starts - 1)  # Generate Sobol samples
        else:
            raise Exception('Initialization method unknown.')

    bounds = np.ones((ndim, 2))
    bounds[:, 0] = -bound
    bounds[:, 1] = +bound

    scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])  # Scale to bounds
    scaled_samples = np.concatenate([np.zeros((1, ndim)), scaled_samples], axis=0)
    start_points = torch.tensor(scaled_samples, dtype=torch.float32)

    return start_points


def minimizer(params, max_iter, verbose: str = True, **kwargs):

    history = torch.zeros(max_iter)

    # Initialize trainable parameters
    x0 = params.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([x0], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    for step in range(max_iter):
        optimizer.zero_grad()  # Reset gradients

        # Compute loss using trainable and fixed parameters
        loss = loss_function(x0, **kwargs)

        history[step] = loss.item()

        # Compute gradients
        loss.backward()

        # Perform optimization step
        optimizer.step()
        scheduler.step(loss)

        # Print progress
        if verbose is True:
            iteration = f'Iteration: {step + 1}/{max_iter}'
            status = f'Loss: {loss.item():.2E}'
            sys.stdout.write('\r' + iteration + '\t' + status)
            sys.stdout.flush()

    return x0, history


def _old_fit_psf(data, exPar, emPar, grid, spad, zer_max: int = 15, bead_radius: float = 0, fit_mag: bool = False,
            show_hist: bool = False, minimizer: str = 'differential_evolution', loss: str = 'mse', process: str = 'cpu',
            **minimizer_params):

    zer_to_fit = zer_max - FIXED_ZERNIKE  # neglect piston, tip, tilt
    
    x0 = np.zeros(zer_to_fit)
    bounds = np.ones((zer_to_fit, 2))
    bounds[:, 0] = -4
    bounds[:, 1] = +4
    
    excitation = exPar.copy()
    emission = emPar.copy()
    
    zernike_idx = np.arange(zer_max)
    excitation.abe_index = zernike_idx
    emission.abe_index = zernike_idx

    if bead_radius > 0:
        obj = disk(grid, bead_radius, process=process)
    else:
        obj = None

    if minimizer == 'differential_evolution':

        if 'mutation' not in minimizer_params:
            minimizer_params.update({'mutation': (0.5, 1.9)})

        if 'disp' not in minimizer_params:
            minimizer_params.update({'disp': True})

        if show_hist == True:

            fig, ax = plt.subplots(1, 1)
            names = excitation.aberration

            ax.set_xticks(zernike_idx)
            ax.set_xticklabels(names, rotation=45, ha='right')

            rects = ax.bar(zernike_idx, np.zeros(zer_max), alpha=0.5, linewidth=1, edgecolor='black')
            plt.ion()
            callback = lambda intermediate_result: callback_evolution(intermediate_result, fig=fig, ax=ax, rects=rects)
        else:
            callback = None

        if 'vectorized' in minimizer_params:
            if minimizer_params['vectorized'] is True:
                vectorized = True
            else:
                vectorized = True
        else:
            vectorized = False

        result = differential_evolution(loss_function, x0=x0, bounds=bounds, callback=callback,
                                        args=(data, excitation, emission, grid, spad, obj, fit_mag, loss, vectorized, process),
                                        **minimizer_params)
    elif minimizer == 'dual_annealing':

        global iter_annealing
        iter_annealing = 1

        result = dual_annealing(loss_function, x0=x0, bounds=bounds, callback=callback_annealing,
                                args=(data, excitation, emission, grid, spad, obj, fit_mag, loss, process),
                                **minimizer_params)
    else:
        raise Exception('Minimizer unknown.')

    return result


def callback_evolution(intermediate_result, fig, ax, rects):

    y = intermediate_result.x

    for n, rect in enumerate(rects[FIXED_ZERNIKE:]):
        rect.set_height(intermediate_result.x[n])

    ax.set_ylim([y.min()-0.2, y.max()+0.2])

    fig.tight_layout()

    plt.pause(0.01)


def callback_annealing(x, f, context):
    global iter_annealing
    print(f'Iteration: {iter_annealing} \t f(x) = {f:.4E}')
    iter_annealing += 1