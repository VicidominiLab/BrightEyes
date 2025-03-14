import numpy as np

import sys

import torch

from scipy.stats import qmc

import brighteyes_ism.simulation.PSF_sim as sim

from brighteyes_ism.simulation.utils import partial_convolution

from s2ism.psf_estimator import GridFinder

from scipy.special import rel_entr

import string

import warnings

#%%

FIXED_ZERNIKE = 3


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


def forward(zer_coeff: torch.tensor, expar: sim.simSettings, empar: sim.simSettings, grid: sim.GridParameters,
            spad: torch.tensor = None, obj: torch.tensor = None, delta_focus: bool = False, process: str = 'cpu'):
    """
    Forward model for the PSF.  It calculates the PSF for a given set of Zernike coefficients.
    It can also convolve the PSF with a sphere of given radius. The PSF is normalized to sum 1.

    Parameters
    ----------
    zer_coeff : torch.tensor
        Zernike coefficients.
    expar : sim.simSettings
        Excitation parameters.
    empar : sim.simSettings
        Emission parameters.
    grid : sim.GridParameters
        Grid parameters.
    spad : torch.tensor
        Array of pinholes.
    obj : torch.tensor
        Object to convolve with the PSF.
    delta_focus : bool
        If True, the defocus parameter can be different for excitation and emission.
    process : str
        Device running the computation. It can be 'cpu' or 'gpu'.

    Returns
    -------
    dset : torch.tensor
        PSF.
    """

    if delta_focus is True:
        zer_ex = torch.cat([zer_coeff[:1], zer_coeff[2:]])
        zer_em = torch.cat([zer_coeff[:2], zer_coeff[3:]])
    else:
        zer_ex = zer_coeff
        zer_em = zer_coeff

    expar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), torch.as_tensor(zer_ex)])
    
    empar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), torch.as_tensor(zer_em)])

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
    
    dset /= dset.sum()
    
    return dset


def vectorized_forward(zer_coeff, expar, empar, grid, spad=None, obj=None, delta_focus: bool = False, process='cpu'):

    device = torch.device("cuda:0" if torch.cuda.is_available() and process == 'gpu' else "cpu")

    n_agents = zer_coeff.shape[0]
    datasets = []

    for n in range(n_agents):
        datasets.append(forward(zer_coeff[n], expar, empar, grid, spad, obj, delta_focus=delta_focus, process=process))

    return torch.stack(datasets, dim=0).to(device)


def loss_function(x0: torch.tensor, data: torch.tensor, expar: sim.simSettings, empar: sim.simSettings,
                  grid: sim.GridParameters, spad: torch.tensor, obj=None, fit_mag: bool = False, loss: str = 'mse',
                  delta_focus: bool = False, vectorized: bool = False, process: str = 'cpu'):
    """
    Loss function for the optimization of the Zernike coefficients. It calculates the difference between the measured
    data and the forward model. The loss can be calculated using different metrics. The forward model can be calculated
    using the vectorized version. The function returns the loss, calculated for each agent.

    Parameters
    ----------
    x0 : torch.tensor
        Zernike coefficients.
    data : torch.tensor
        Measured data.
    expar : sim.simSettings
        Excitation parameters.
    empar : sim.simSettings
        Emission parameters.
    grid : sim.GridParameters
        Grid parameters.
    spad : torch.tensor
        Array of pinholes.
    obj : torch.tensor
        Object to convolve with the PSF.
    fit_mag : bool
        If True, the magnification of the microscope is also optimized.
    loss : str
        Metric to calculate the loss. It can be 'mse', 'mae' or 'kl'.
    delta_focus : bool
        If True, the defocus parameter can be different for excitation and emission.
    vectorized : bool
        If True, the forward model is calculated using the vectorized version.
    process : str
        Device running the computation. It can be 'cpu' or 'gpu'.

    Returns
    -------
    loss : torch.tensor
        Value of the loss at point x0. If vectorized is True, it contains the loss for each agent.
    """

    if fit_mag is True:
        grid.M = x0[0]
        zer_coeff = x0[1:]
    else:
        zer_coeff = x0.clone()

    if vectorized is True:
        estimate = vectorized_forward(zer_coeff, expar, empar, grid, spad, obj, delta_focus=delta_focus, process=process)
    else:
        estimate = forward(zer_coeff, expar, empar, grid, spad, obj, delta_focus=delta_focus, process=process)

    axis = tuple(-np.arange(1, data.ndim+1))

    if loss == 'mse':
        loss = torch.sum(torch.abs(data - estimate)**2, axis=axis)
    elif loss == 'mae':
        loss = torch.sum(torch.abs(data - estimate), axis=axis)
    elif loss == 'kl':
        loss = torch.sum(rel_entr(data, estimate), axis=axis)
    else:
        raise Exception("Loss function unknown.")

    return loss


def fit_psf(data: torch.tensor, expar: sim.simSettings, empar: sim.simSettings, grid: sim.GridParameters,
            spad: torch.tensor, zer_max: int = 15, bead_radius: float = 0, delta_focus: bool = False, loss: str = 'mae',
            num_starts: int = 10, max_iter: int = 200, bound: int = 2, initialization: str = 'sobol',
            verbose: str = True, process: str = 'cpu'):
    """
    It fits the Zernike coefficients to the measured data. The optimization is performed using the Adam optimizer with
    multiple starting points. The optimization can be performed using different metrics and different initialization
    methods (Sobol and Latin Hypercube sampling).

    Parameters
    ----------
    data : torch.tensor
        Measured data.
    expar : sim.simSettings
        Excitation parameters.
    empar : sim.simSettings
        Emission parameters.
    grid : sim.GridParameters
        Grid parameters.
    spad : torch.tensor
        Array of pinholes.
    zer_max : int
        Maximum number of Zernike coefficients to fit (excluding the first three: piston, tip, and tilt).
    bead_radius : float
        Radius of the bead to convolve with the PSF.
    delta_focus : bool
        If True, the defocus parameter can be different for excitation and emission.
    loss : str
        Metric to calculate the loss. It can be 'mse', 'mae' or 'kl'.
    num_starts : int
        Number of starting points for the optimization.
    max_iter : int
        Maximum number of iterations for the optimization.
    bound : int
        Bounds of the hypercube where to perform the initialization.
    initialization : str
        Initialization method. It can be 'lhs' or 'sobol'.
    verbose : str
        If True, it prints the progress of the optimization.
    process : str
        Device running the computation. It can be 'cpu' or 'gpu'.

    Returns
    -------
    best_result : torch.tensor
        Optimized Zernike coefficients.
    history : torch.tensor
        History of the loss during the optimization.
    """

    zer_to_fit = zer_max - FIXED_ZERNIKE  # neglect piston, tip, tilt
    if delta_focus is True:
        zer_to_fit = zer_to_fit + 1

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
              'delta_focus': delta_focus,
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


def generate_starting_points(ndim: int, num_starts: int, bound: float, initialization: str):
    """
    It generates starting points for the optimization. The points are generated using Latin Hypercube or Sobol sampling.

    Parameters
    ----------
    ndim : int
        Number of dimensions.
    num_starts : int
        Number of starting points.
    bound : float
        Bounds of the hypercube where to perform the initialization.
    initialization : str
        Initialization method. It can be 'lhs' or 'sobol'.

    Returns
    -------
    start_points : torch.tensor
        Starting points for the optimization.
    """

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


def minimizer(params: torch.tensor, max_iter: int, verbose: str = True, **kwargs):
    """
    It performs the optimization of the Zernike coefficients using the Adam optimizer.  The optimization is performed
    using the loss function defined in the kwargs. The function returns the optimized parameters and the history of the
    loss during the optimization.

    Parameters
    ----------
    params : torch.tensor
        Initial parameters.
    max_iter : int
        Maximum number of iterations.
    verbose : str
        If True, it prints the progress of the optimization.
    kwargs : dict
        Dictionary with the parameters for the loss function.

    Returns
    -------
    x0 : torch.tensor
        Optimized parameters.
    history : torch.tensor
        History of the loss during the optimization.
    """

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