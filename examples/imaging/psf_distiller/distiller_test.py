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

from distiller import FIXED_ZERNIKE
import distiller as dis

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
grid.Nx = 200

grid.pxsizez = 50
grid.Nz = 5

grid.pxpitch = 75e3
grid.pxdim = 75e3
grid.pinhole_shape = 'circle'
grid.N = 5
grid.M = 450

grid.rotation = np.pi/8
# grid.mirroring = -1

wavefront_gt = exPar.wavefront

#%%

radius = 40 / grid.pxsizex

obj = dis.disk(grid.Nx, radius, fourier = True, process = process)

#%%

device = 'cuda' if process == 'gpu' else 'cpu'

spad = sim.custom_detector(grid, device)

#%%

x0 = zernike_input[FIXED_ZERNIKE:]

data = dis.forward(x0, exPar, emPar, grid, spad = spad, process = process, obj=obj)

fig = gr.ShowDataset(data.cpu())

#%%

signal = 1e6

data *= signal

data = torch.poisson(data)

fig = gr.ShowDataset(data.cpu())

data = data/torch.sum(data)

#%%

minimizer = 'differential_evolution'
loss = 'mse'

kwargs = {'maxiter' : 1000,
          'mutation' : (0.5, 1.9),
          'disp' : True}

curr = time.time()

result = dis.fit_psf(data, exPar, emPar, grid, spad, bead_radius=radius,
                     zer_max=zer_max, loss = loss, minimizer = minimizer,
                     process = process, **kwargs)

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%%

zernike_idx = np.arange(zer_max)

zernike_output = np.concatenate([np.zeros(FIXED_ZERNIKE), result.x])

plt.figure()
plt.bar(zernike_idx, zernike_input, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'ground truth')
plt.bar(zernike_idx, zernike_output, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'fit')

plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45)

plt.legend()

plt.tight_layout()

#%%

exPar.abe_index = zernike_idx
exPar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), result.x])

emPar.abe_index = zernike_idx
emPar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), result.x])

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
