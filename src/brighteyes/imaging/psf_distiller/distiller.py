import numpy as np

import torch

import brighteyes_ism.simulation.PSF_sim as sim

from brighteyes_ism.simulation.utils import partial_convolution

from s2ism.psf_estimator import GridFinder

from scipy.optimize import differential_evolution, dual_annealing

from scipy.special import rel_entr

import string

#%%

FIXED_ZERNIKE = 3

iter_annealing = 1

def disk(nx: int, radius: float, fourier: bool = True, process: str = 'cpu'):
    """
    It calculates a binary mask with the shape of a disk.

    Parameters
    ----------
    nx : int
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

    x = torch.arange(-(nx // 2), nx // 2 + np.mod(nx,2))

    xx, yy = torch.meshgrid(x, x, indexing='ij')
    r = torch.sqrt(xx ** 2 + yy ** 2)

    circle = torch.where(r < radius, 1, 0)

    if fourier is True:
        circle = torch.fft.fftn(circle)

    return circle.to(device)


def fit_detector(data, grid, exPar, emPar):

    grid_fit = GridFinder(grid)
    
    grid_fit.estimate(data, exPar.wl, emPar.wl, exPar.na)
    
    grid_fit.Print()
    
    spad = sim.custom_detector(grid_fit, 'cpu')
    
    return spad


def forward(zer_coeff, expar, empar, grid, spad=None, obj=None, dark_noise=None, pxdwelltime=None, process='cpu'):
    
    nch = grid.Nch
    
    expar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), zer_coeff])
    
    empar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), zer_coeff])

    if grid.Nz > 1:
        psf, *_ = sim.SPAD_PSF_3D(grid, expar, empar, spad=spad, process=process, output = 'tensor')
    else:
        psf, *_ = sim.SPAD_PSF_2D(grid, expar, empar, spad=spad, process=process, output = 'tensor')

    alphabet = list(string.ascii_lowercase)
    obj_axes = ''.join(alphabet[:obj.ndim])
    psf_axes = ''.join(alphabet[:psf.ndim])

    if obj is not None:
        dset = partial_convolution(obj, psf, obj_axes, psf_axes, obj_axes, fourier=(1, 0))
    else:
        dset = psf
    
    if dark_noise is not None:
        for c in range(nch):
            dset[..., c] += dark_noise[c]*pxdwelltime
    
    dset /= dset.sum()
    
    return dset


def loss_function(x0, data, expar, empar, grid, spad, obj=None, fit_mag: bool = False, loss: str = 'mse', process: str = 'cpu'):

    if fit_mag is True:
        grid.M = x0[0]
        zer_coeff = x0[1:]
    else:
        zer_coeff = x0

    estimate = forward(zer_coeff, expar, empar, grid, spad, obj, process=process)
    
    if loss == 'mse':
        loss = torch.sum(torch.abs(data - estimate)**2)
    elif loss == 'kl':
        loss = torch.sum(rel_entr(data, estimate))
    else:
        raise Exception("Loss function unknown.")

    loss_value = loss.cpu().detach().numpy()

    return loss_value


def fit_psf(data, exPar, emPar, grid, spad, zer_max: int = 15, bead_radius: float = 0, fit_mag: bool = False,
            minimizer: str = 'differential_evolution', loss: str = 'mse', process: str = 'cpu', **minimizer_params):

    zer_to_fit = zer_max - FIXED_ZERNIKE  # neglect piston, tip, tilt
    
    x0 = np.zeros(zer_to_fit)
    bounds = np.ones((zer_to_fit, 2))
    bounds[:, 0] = -5
    bounds[:, 1] = +5
    
    excitation = exPar.copy()
    emission = emPar.copy()
    
    zernike_idx = np.arange(zer_max)
    excitation.abe_index = zernike_idx
    emission.abe_index = zernike_idx

    if bead_radius > 0:
        obj = disk(grid.Nx, bead_radius, process=process)
    else:
        obj = None

    if minimizer == 'differential_evolution':

        if 'mutation' not in minimizer_params:
            minimizer_params.update({'mutation': (0.5, 1.9)})

        if 'disp' not in minimizer_params:
            minimizer_params.update({'disp': True})

        result = differential_evolution(loss_function, x0=x0, bounds=bounds, # callback=callback_evolution,
                                        args=(data, excitation, emission, grid, spad, obj, fit_mag, loss, process),
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

def callback_evolution(*, intermediate_result):
    print(intermediate_result.x)


def callback_annealing(x, f, context):
    global iter_annealing
    print(f'Iteration: {iter_annealing} \t f(x) = {f:.4E}')
    iter_annealing += 1