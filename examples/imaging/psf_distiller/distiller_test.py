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

# plt.close('all')

#%% excitation

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

zernike_ex = np.zeros(zer_max)
zernike_ex[exPar.abe_index] = exPar.abe_ampli

exPar.abe_index = zernike_idx
exPar.abe_ampli = zernike_ex

names = exPar.aberration

wavefront_gt_ex = exPar.wavefront

#%% emission

emPar = exPar.copy()
emPar.wl = 660

emPar.abe_index = [6, 8, 10, 13]
emPar.abe_ampli = [0.6, 1.7, 1.1, 0.9]

zernike_em = np.zeros(zer_max)
zernike_em[emPar.abe_index] = emPar.abe_ampli

emPar.abe_index = zernike_idx
emPar.abe_ampli = zernike_em

wavefront_gt_em = emPar.wavefront

#%% geometrical parameters

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

#%% generate bead

radius = 40 # nm

obj = dis.disk(grid, radius, fourier = True, process = process)

#%% select device

device = 'cuda' if process == 'gpu' else 'cpu'

spad = sim.custom_detector(grid, device)

#%% simulate forward

x0 = torch.stack([torch.tensor(zernike_ex[FIXED_ZERNIKE:]), 
                  torch.tensor(zernike_em[FIXED_ZERNIKE:])], dim=0)

data = dis.forward(x0, exPar, emPar, grid, spad = spad, process = process, obj=obj, independent_wavefront = True)

if grid.Nz > 1:
    fig = gr.StackSlider(data.cpu().detach().numpy().sum(-1), grid.pxsizex, grid.pxsizez)
else:
    fig = gr.ShowDataset(data.cpu())

fig.suptitle('Clean data')
fig.tight_layout()
# fig.savefig('clean_data.png', dpi = 300)

#%% generate noisy data

signal = 1e6

data *= signal

data = torch.poisson(data)

if grid.Nz > 1:
    fig = gr.StackSlider(data.cpu().detach().numpy().sum(-1), grid.pxsizex, grid.pxsizez)
else:
    fig = gr.ShowDataset(data.cpu())

fig.suptitle('Noisy data')
fig.tight_layout()
# fig.savefig('noisy_data.png', dpi = 300)

data = data/torch.sum(data)

#%% solve inverse problem

independent_wavefront = True
loss = 'mae'
initialization = 'sobol'
num_starts = 17
max_iter = 200
bounds = 2

curr = time.time()

result, history = dis.fit_psf(data, exPar, emPar, grid, spad, 
                              bead_radius=radius, zer_max=zer_max, loss = loss,
                              initialization=initialization,
                              num_starts=num_starts, max_iter=max_iter,
                              bound=bounds, process = process,
                              independent_wavefront=independent_wavefront)

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%% show minimization history

plt.figure()
plt.semilogy(np.asarray(history).T)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Minimization history')

plt.tight_layout()
# plt.savefig('history.png', dpi = 300)

#%% excitation fit result

zernike_output_ex = result[0]

plt.figure()
plt.bar(zernike_idx, zernike_ex, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'ground truth')
plt.bar(zernike_idx, zernike_output_ex, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'fit')

plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45, ha = 'right')
plt.gca().set_title('Excitation')

plt.legend()

plt.tight_layout()
# plt.savefig('ex_hist.png', dpi = 300)

#%% emission fit result

zernike_output_em = result[1]

plt.figure()
plt.bar(zernike_idx, zernike_em, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'ground truth')
plt.bar(zernike_idx, zernike_output_em, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'fit')

plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45, ha = 'right')
plt.gca().set_title('Emission')

plt.legend()

plt.tight_layout()
# plt.savefig('em_hist.png', dpi = 300)

#%% calculate fit wavefront

exPar.abe_index = zernike_idx
exPar.abe_ampli = zernike_output_ex

wavefront_fit_ex = exPar.wavefront

emPar.abe_index = zernike_idx
emPar.abe_ampli = zernike_output_em

wavefront_fit_em = emPar.wavefront

#%% show excitation wavefront

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

ax[0].imshow(wavefront_gt_ex, cmap='seismic')
ax[0].axis('off')
ax[0].set_title('Ground truth')

ax[1].imshow(wavefront_fit_ex, cmap='seismic')
ax[1].axis('off')
ax[1].set_title('Excitation fit')

fig.tight_layout()
# plt.savefig('ex_wavefront.png', dpi = 300)

#%% show emission wavefront

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

ax[0].imshow(wavefront_gt_em, cmap='seismic')
ax[0].axis('off')
ax[0].set_title('Ground truth')

ax[1].imshow(wavefront_fit_em, cmap='seismic')
ax[1].axis('off')
ax[1].set_title('Emission fit')

fig.tight_layout()
# plt.savefig('em_wavefront.png', dpi = 300)

#%% show distilled psfs

fit, *_ = sim.SPAD_PSF_2D(grid, exPar, emPar, spad=spad, process = process)

fig = gr.ShowDataset(fit)

fig.suptitle('Distilled PSFs')

fig.tight_layout()
# plt.savefig('distilled_psfs.png', dpi = 300)

#%% compare fingerprints

fig, ax = plt.subplots(1,2)

gr.ShowFingerprint(data.cpu().detach().numpy(), fig = fig, ax = ax[0])
ax[0].set_title('Data')

gr.ShowFingerprint(fit, fig = fig, ax = ax[1])
ax[1].set_title('Fit')

fig.suptitle('Fingerprint')

fig.tight_layout()
# plt.savefig('fingerprints.png', dpi = 300)
