import numpy as np
import matplotlib.pyplot as plt
import os

import torch

from brighteyes_ism.dataio import mcs
import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Graph_lib as gr
import brighteyes_ism.analysis.Tools_lib as tool
import brighteyes_ism.analysis.APR_lib as apr

from s2ism.psf_estimator import GridFinder

import time

#%%

from psf_ism_distiller.distiller import FIXED_ZERNIKE
import psf_ism_distiller.distiller as dis

process = 'gpu'

#%%

plt.close('all')

#%%

path = r'\\iitfsvge101.iit.local\mms\Data MMS server\STED-ISM\AxialDeconvolution\Beads\Gold - 2'
file = r'data-07-12-2023-12-31-52.h5'

# path = r'\\iitfsvge101.iit.local\mms\Data MMS server\Other\PSF_distiller\2025-04-24_PSF_beads_distiller\100nm_gold'
# file = r'data-24-04-2025-17-05-28.h5'

fullpath = os.path.join(path, file)

dset, meta = mcs.load(fullpath)

#%%

# import napari

# viewer = napari.Viewer()
# viewer.add_image(dset)

#%% pre-process data

if meta.nz > 1:
    z0 = 38
    data_rep = dset[:,z0].sum(-2)
else:
    data_rep = dset.sum(axis=(1,-2))
    
data_rep = data_rep[:, 100:-100, 100:-100]

#%%

from skimage.filters import gaussian

# data = dset[:,38].sum(axis=(0,-2))
# data = data[100:-100, 100:-100]
# nx, ny = data.shape[:2]

nrep, nx, ny, nch = data_rep.shape

data = np.empty_like(data_rep)

for k in range(nrep):
    cx, cy, cc = tool.ArgMaxND(gaussian(data_rep[k], 4, channel_axis = -1))[0]

    sx, sy = nx//2 - cx, ny//2 - cy

    data[k] = np.roll(data_rep[k], [sx, sy], axis = (0, 1))

data = data.sum(0)

data = data/data.sum()

#%%

gr.ShowDataset(data)

plt.tight_layout()
plt.savefig('exp_data.png', dpi = 300)

#%%

shift_vectors, _ = apr.ShiftVectors(data, 10, 12)
gr.PlotShiftVectors(shift_vectors, meta.dx)

gr.ShowFingerprint(data)

#%%

exPar = sim.simSettings()
exPar.na = 1.4   # numerical aperture
exPar.wl = 640   # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90  # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 101 #128

emPar = exPar.copy()
emPar.wl = 660

grid = sim.GridParameters()
grid.pxsizex = meta.dx*1e3
grid.Nx = data.shape[0]

grid.pxpitch = 75e3
grid.pxdim = 50e3
grid.pinhole_shape = 'circle'
grid.N = 5
grid.M = 500 #450

#%%

grid_fit = GridFinder(grid)

grid_fit.estimate(data, exPar.wl, emPar.wl, exPar.na)

grid_fit.Print()

spad = sim.custom_detector(grid_fit, 'cpu')

plt.figure()
plt.imshow(spad.cpu().detach().numpy().sum(-1))

#%%

curr = time.time()

result, history = dis.fit_psf(data, exPar, emPar, grid, spad, bead_radius=40,
                              independent_wavefront=True, num_starts=17, max_iter=200)

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%%

plt.figure()
plt.semilogy(np.asarray(history).T)
plt.xlabel('Iteration')
plt.ylabel('Loss')

#%%

zer_max = 15
zernike_idx = np.arange(zer_max)

n_zer = zer_max - FIXED_ZERNIKE

zer_ex = result[:n_zer].cpu().detach()
exPar.abe_index = zernike_idx
exPar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), zer_ex])

zer_em = result[n_zer:].cpu().detach()
emPar.abe_index = zernike_idx
emPar.abe_ampli = torch.cat([torch.zeros(FIXED_ZERNIKE), zer_em])

names = exPar.aberration

#%%

zernike_output_ex = np.concatenate([np.zeros(FIXED_ZERNIKE), zer_ex])
zernike_output_em = np.concatenate([np.zeros(FIXED_ZERNIKE), zer_em])

plt.figure()
plt.bar(zernike_idx, zernike_output_ex, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'Excitation')
plt.bar(zernike_idx, zernike_output_em, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'Emission')

plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45, ha = 'right')

plt.legend()

plt.tight_layout()

plt.savefig('exp_hist.png', dpi = 300)

#%%

wavefront_fit_ex = exPar.wavefront
wavefront_fit_em = emPar.wavefront

fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

ax[0].imshow(wavefront_fit_ex, cmap='seismic')
ax[0].axis('off')
ax[0].set_title('Excitation')

ax[1].imshow(wavefront_fit_em, cmap='seismic')
ax[1].axis('off')
ax[1].set_title('Emission')

fig.tight_layout()
fig.savefig('exp_wavefront.png', dpi = 300)

#%%

fit, *_ = sim.SPAD_PSF_2D(grid_fit, exPar, emPar, spad=spad, process='cpu')

fig = gr.ShowDataset(fit)

fig.tight_layout()
fig.savefig('exp_fit.png', dpi = 300)

#%%

gr.ShowFingerprint(fit)