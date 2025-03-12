import numpy as np
import matplotlib.pyplot as plt
import os

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

fullpath = os.path.join(path, file)

dset, meta = mcs.load(fullpath)

#%%

data = dset[:,38].sum(axis=(0,-2))
data = data[100:-100, 100:-100]
data = data/data.sum()

nx, ny = data.shape[:2]
cx, cy, cc = tool.ArgMaxND(data)[0]

sx, sy = nx//2 - cx, ny//2 - cy

data = np.roll(data, [sx, sy], axis = (0, 1))

#%%

gr.ShowDataset(data)

shift_vectors, _ = apr.ShiftVectors(data, 10, 12)
gr.PlotShiftVectors(shift_vectors, meta.dx)

#%%

exPar = sim.simSettings()
exPar.na = 1.4   # numerical aperture
exPar.wl = 640   # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90  # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 101 #128

emPar = exPar.copy()
emPar.wl = 640

grid = sim.GridParameters()
grid.pxsizex = meta.dx*1e3
grid.Nx = data.shape[0]

grid.pxpitch = 75e3
grid.pxdim = 75e3
grid.pinhole_shape = 'circle'
grid.N = 5
grid.M = 450

#%%

grid_fit = GridFinder(grid)

grid_fit.estimate(data, exPar.wl, emPar.wl, exPar.na)

grid_fit.Print()

spad = sim.custom_detector(grid_fit, 'cpu')

plt.figure()
plt.imshow(spad.cpu().detach().numpy().sum(-1))

#%%

curr = time.time()

result, history = dis.fit_psf(data, exPar, emPar, grid, spad, bead_radius=40, num_starts=50, max_iter=500)

elapsed = time.time() - curr

print(f'Time = {elapsed:.2f} s')

#%%

plt.figure()
plt.semilogy(np.asarray(history).T)
plt.xlabel('Iteration')
plt.ylabel('Loss')

#%%

from poppy.zernike import zern_name

zer_max = 15

zernike_idx = np.arange(zer_max)
names = [zern_name(x) for x in zernike_idx]

zernike_output = np.concatenate([np.zeros(3), result.cpu().detach().numpy()])

plt.figure()
plt.bar(zernike_idx, zernike_output, alpha = 0.5, linewidth = 1, edgecolor = 'black', label = 'fit')

# plt.xlabel('Zernike coefficient')
plt.ylabel(r'Aberration amplitude ($OPD/\lambda)$')

plt.gca().set_xticks(zernike_idx)
plt.gca().set_xticklabels(names, rotation=45)

plt.legend()

plt.tight_layout()

#%%

exPar.abe_index = zernike_idx
exPar.abe_ampli = np.concatenate([np.zeros(3), result.cpu().detach().numpy()])

emPar.abe_index = zernike_idx
emPar.abe_ampli = np.concatenate([np.zeros(3), result.cpu().detach().numpy()])

wavefront_fit = exPar.wavefront

fit, *_ = sim.SPAD_PSF_2D(grid_fit, exPar, emPar, spad=spad, process='cpu')

fig = gr.ShowDataset(fit)
