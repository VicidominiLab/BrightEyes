import numpy as np
import matplotlib.pyplot as plt

import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Graph_lib as gr

from s2ism.psf_estimator import GridFinder

from torch import optim

import time

#%%

import sys
sys.path.insert(0, '../src/')

from distiller import *

#%%

plt.close('all')

#%%

exPar = sim.simSettings()
exPar.na = 1.4   # numerical aperture
exPar.wl = 640   # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90  # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 101 #128

# exPar.mask = 'VP'

exPar.abe_index = [5, 7, 9, 11]
exPar.abe_ampli = [1, 1.2, 0.8, 2]

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

#%%

wavefront_gt = exPar.wavefront

#%%

zer_max = 12
zernike_input = np.zeros(zer_max)
zernike_input[emPar.abe_index] = emPar.abe_ampli

data, *_ = sim.SPAD_PSF_2D(grid, exPar, emPar, process='cpu')

fig = gr.ShowDataset(data)

#%%

signal = 1e5

data *= signal

data = np.random.poisson(data)

fig = gr.ShowDataset(data)

data = data/np.sum(data)

#%%

grid_fit = GridFinder(grid)

grid_fit.estimate(data, exPar.wl, emPar.wl, exPar.na)

grid_fit.Print()

spad = sim.custom_detector(grid_fit, 'cpu')

#%%

curr = time.time()

result = fit_psf(data, exPar, emPar, grid, spad, loss = 'kl')

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%%

from poppy.zernike import zern_name

zernike_idx = np.arange(zer_max)
names = [zern_name(x) for x in zernike_idx]

zernike_output = np.concatenate([np.zeros(4), result.x])

plt.figure()
plt.bar(zernike_idx, zernike_input, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'ground truth')
plt.bar(zernike_idx, zernike_output, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'fit')

# plt.xlabel('Zernike coefficient')
plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45)

plt.legend()

plt.tight_layout()

#%%

exPar.abe_index = zernike_idx[1:]
exPar.abe_ampli = np.concatenate([np.zeros(3), result.x])

emPar.abe_index = zernike_idx[1:]
emPar.abe_ampli = np.concatenate([np.zeros(3), result.x])

wavefront_fit = exPar.wavefront

#%%

fit, *_ = sim.SPAD_PSF_2D(grid_fit, exPar, emPar, spad=spad, process='cpu')

fig = gr.ShowDataset(fit)

#%%

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

ax[0].imshow(wavefront_gt, cmap='seismic')
ax[0].axis('off')
ax[0].set_title('Ground truth')

ax[1].imshow(wavefront_fit, cmap='seismic')
ax[1].axis('off')
ax[1].set_title('Fit')
