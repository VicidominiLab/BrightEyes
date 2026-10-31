import numpy as np
import matplotlib.pyplot as plt

import brighteyes.imaging.simulation.PSF_sim as sim
import brighteyes.imaging.analysis.Graph_lib as gr

from brighteyes.imaging.s2ism.psf_estimator import GridFinder

import torch

import time

#%%

from brighteyes.imaging.psf_distiller.distiller import FIXED_ZERNIKE
import brighteyes.imaging.psf_distiller.distiller as dis

process = 'gpu'

#%%

plt.close('all')

#%%

exPar = sim.simSettings()
exPar.na = 1.4    # numerical aperture
exPar.wl = 640    # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90   # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 101 # 128

# exPar.mask = 'vortex'

exPar.abe_index = [5, 7, 9]
exPar.abe_ampli = [1, 1.2, 0.8]

zer_max = 15
zernike_idx = np.arange(zer_max)
zernike_input = np.zeros(zer_max)
zernike_input[exPar.abe_index] = exPar.abe_ampli

exPar.abe_index = zernike_idx
exPar.abe_ampli = zernike_input

names = exPar.aberration

emPar = exPar.copy()
emPar.wl = 660

grid = sim.GridParameters()
grid.pxsizex = 10
grid.Nx = 201

grid.pxsizez = 100
grid.Nz = 1

grid.pxpitch = 75e3
grid.pxdim = 75e3
grid.pinhole_shape = 'circle'
grid.N = 5
grid.M = 450

grid.rotation = np.pi/8
# grid.mirroring = -1

wavefront_gt = exPar.wavefront

#%%

radius = 40 # nm

obj = dis.disk(grid, radius, fourier = True, process = process)

#%%

device = 'cuda' if process == 'gpu' else 'cpu'

spad = sim.custom_detector(grid, device)

#%%

x0 = zernike_input[FIXED_ZERNIKE:]

data = dis.forward(x0, exPar, emPar, grid, spad = spad, process = process, obj=obj)

if grid.Nz > 1:
    fig = gr.StackSlider(data.cpu().detach().numpy().sum(-1), grid.pxsizex, grid.pxsizez)
else:
    fig = gr.ShowDataset(data.cpu())

#%%

signal = 1e6

data *= signal

data = torch.poisson(data)

if grid.Nz > 1:
    fig = gr.StackSlider(data.cpu().detach().numpy().sum(-1), grid.pxsizex, grid.pxsizez)
else:
    fig = gr.ShowDataset(data.cpu())

data = data/torch.sum(data)

#%%

zer_to_fit = zer_max - FIXED_ZERNIKE

x0 = np.zeros(zer_to_fit)
bounds = np.ones((zer_to_fit, 2))
bounds[:, 0] = -5
bounds[:, 1] = +5

excitation = exPar.copy()
emission = emPar.copy()

zernike_idx = np.arange(zer_max)
excitation.abe_index = zernike_idx
emission.abe_index = zernike_idx

import pyswarms as ps
from pyswarms.utils.plotters import plot_cost_history

options = {'c1': 0.5, 'c2': 0.8, 'w': 0.9}

# options = {'c1': 2, 'c2': 2, 'w': 1.5, 'k': 2, 'p': 2}

x_max = 5 * np.ones(zer_to_fit)
x_min = -1 * x_max
bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=zer_to_fit,
                                    bounds=bounds, options=options)


# optimizer = ps.single.LocalBestPSO(n_particles=10, dimensions=zer_to_fit,
#                                     bounds=bounds, options=options)


kwargs = {'data' : data,
          'expar': excitation,
          'empar': emission,
          'grid': grid,
          'spad': spad,
          'obj': obj,
          'fit_mag': False,
          'loss': 'mse',
          'vectorized': True,
          'process': process
          }

curr = time.time()

cost, pos = optimizer.optimize(dis.loss_function, iters = 500, **kwargs)

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%%

plot_cost_history(cost_history=optimizer.cost_history)

#%%

zernike_idx = np.arange(zer_max)

zernike_output = np.concatenate([np.zeros(FIXED_ZERNIKE), pos])

plt.figure()
plt.bar(zernike_idx, zernike_input, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'ground truth')
plt.bar(zernike_idx, zernike_output, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'fit')

plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45, ha = 'right')

plt.legend()

plt.tight_layout()

#%%

exPar.abe_index = zernike_idx
exPar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), pos])

emPar.abe_index = zernike_idx
emPar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), pos])

wavefront_fit = exPar.wavefront

#%%

fit, *_ = sim.SPAD_PSF_2D(grid, exPar, emPar, spad=spad, process = process)

fig = gr.ShowDataset(fit)

#%%

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

ax[0].imshow(wavefront_gt, cmap='seismic')
ax[0].axis('off')
ax[0].set_title('Ground truth')

ax[1].imshow(wavefront_fit, cmap='seismic')
ax[1].axis('off')
ax[1].set_title('Fit')
