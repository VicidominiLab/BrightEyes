import numpy as np
import matplotlib.pyplot as plt

import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Graph_lib as gr
from brighteyes_ism.analysis.Tools_lib import ArgMaxND

from s2ism.psf_estimator import GridFinder

import torch

import time

#%%

# import sys
# sys.path.insert(0, '../src/')

from psf_ism_distiller.distiller import FIXED_ZERNIKE
import psf_ism_distiller.distiller as dis

process = 'cpu'

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

#%% plot space
#abe_index = [5, 7, 9, 12]

n_points = 25

index_x, index_y = 6, 7

abe_x = np.linspace(-4, 4, num=n_points)
abe_y = np.linspace(-4, 4, num=n_points)

zer_to_fit = zer_max - FIXED_ZERNIKE

x0 = np.zeros((n_points, n_points, zer_to_fit))

for z in exPar.abe_index[FIXED_ZERNIKE:]:
    x0[:,:,z-FIXED_ZERNIKE] = np.ones((n_points, n_points)) * exPar.abe_ampli[z].cpu().detach().numpy()

for x in range(n_points):
    for y in range(n_points):
        x0[x,y,index_x-FIXED_ZERNIKE] = abe_x[x]
        x0[x,y,index_y-FIXED_ZERNIKE] = abe_y[y]

x0 = torch.as_tensor(x0.reshape( (n_points**2, zer_to_fit) ))

excitation = exPar.copy()
emission = emPar.copy()

zernike_idx = np.arange(zer_max)
excitation.abe_index = zernike_idx
emission.abe_index = zernike_idx

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

loss_map = dis.loss_function(x0, **kwargs)
loss_map = loss_map.reshape( (n_points, n_points) )

exPar.abe_index = zernike_idx
exPar.abe_ampli = zernike_input

#%%

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

truth = [exPar.abe_ampli[index_x], exPar.abe_ampli[index_y]]

fig, ax = plt.subplots(1,1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)

im = ax.imshow(loss_map, norm = LogNorm(), extent = (abe_x[0], abe_x[-1], abe_y[-1], abe_y[0]))
fig.colorbar(im, cax=cax, orientation='vertical')

ax.scatter(truth[1], truth[0], marker = 'x', color='red')

#%%

# argmin = ArgMaxND(-loss_map)[0]

# ax.scatter(argmin[0], argmin[1], marker = 'x', color='red')
