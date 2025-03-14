import numpy as np
import matplotlib.pyplot as plt

import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Graph_lib as gr

from s2ism.psf_estimator import GridFinder

import torch

import time

#%%

# import sys
# sys.path.insert(0, '../src/')

from psf_ism_distiller.distiller import FIXED_ZERNIKE
import psf_ism_distiller.distiller as dis

process = 'gpu'

#%%

# plt.close('all')

#%%

exPar = sim.simSettings()
exPar.na = 1.4    # numerical aperture
exPar.wl = 640    # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90   # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 101 # 128

# exPar.mask = 'vortex'

exPar.abe_index = [5, 7, 9, 12]
exPar.abe_ampli = [1, 1.2, 0.8, 1.5]

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

# minimizer = 'differential_evolution'
# # minimizer = 'dual_annealing'
# loss = 'mae'

# kwargs = {'maxiter' : 1000,
#           'mutation' : (0.5, 1.99),
#           # 'recombination': 0.9,
#           # 'popsize': 15,
#           # 'strategy': "best1bin",
#           # 'init': 'sobol',
#           'disp' : True,
#           # 'vectorized': True,
#           # 'updating' : 'deferred',
#           # 'workers' :  -1,
#           }

# curr = time.time()

# result = dis.fit_psf(data, exPar, emPar, grid, spad, bead_radius=radius,
#                      zer_max=zer_max, loss = loss, minimizer = minimizer,
#                      process = process, show_hist = True, **kwargs)

# elapsed = time.time() - curr

# print(f'Time = {elapsed:.2f} s')

#%%

loss = 'mae'
initialization = 'sobol'
num_starts = 10
max_iter = 200
bounds = 2

curr = time.time()

result, history = dis.fit_psf(data, exPar, emPar, grid, spad, 
                              bead_radius=radius, zer_max=zer_max, loss = loss,
                              initialization=initialization,
                              num_starts=num_starts, max_iter=max_iter,
                              bound=bounds, process = process)

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%%

zernike_idx = np.arange(zer_max)

zernike_output = np.concatenate([np.zeros(FIXED_ZERNIKE), result.cpu().detach().numpy()])

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
exPar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), result])

emPar.abe_index = zernike_idx
emPar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), result])

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
