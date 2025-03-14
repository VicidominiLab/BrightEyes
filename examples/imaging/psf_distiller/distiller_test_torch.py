import numpy as np
import matplotlib.pyplot as plt

import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Graph_lib as gr

import torch

import time

#%%

from psf_ism_distiller.distiller import FIXED_ZERNIKE
import psf_ism_distiller.distiller as dis

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

exPar.abe_index = [6, 8, 10, 13]
exPar.abe_ampli = [1, 1.2, 0.8, 1.5]

zer_max = 15
zernike_idx = np.arange(zer_max)
zernike_input = torch.zeros(zer_max)
zernike_input[exPar.abe_index] = torch.Tensor(exPar.abe_ampli)

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

import sys
import torch
from scipy.stats import qmc

zer_to_fit = zer_max - FIXED_ZERNIKE

x0 = torch.zeros(zer_to_fit, dtype=torch.float32, requires_grad=True)

bounds = np.ones((zer_to_fit, 2))
bounds[:, 0] = -2
bounds[:, 1] = +2

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
          'loss': 'mae',
          'vectorized': False,
          'process': process
          }

# Number of optimization runs (starting points)
num_starts = 10
max_iter = 1000

sampling = 'sobol'

if sampling == 'lhs':
    lhs_sampler = qmc.LatinHypercube(d=zer_to_fit)
    samples = lhs_sampler.random(n=num_starts - 1)  # Generate LHS samples
elif sampling == 'sobol':
    sobol_sampler = qmc.Sobol(d=zer_to_fit)
    samples = sobol_sampler.random(n=num_starts-1)  # Generate Sobol samples

scaled_samples = qmc.scale(samples, bounds[:,0], bounds[:,1])  # Scale to bounds
scaled_samples = np.concatenate([np.zeros((1,zer_to_fit)), scaled_samples], axis=0)

# Convert to PyTorch tensor
start_points = torch.tensor(scaled_samples, dtype=torch.float32)

history = torch.zeros( (num_starts, max_iter) )
result = torch.zeros( (num_starts, zer_to_fit) )

curr = time.time()

for i, start in enumerate(start_points):
    
    print(f"\nOptimization: {i+1}/{len(start_points)}")

    # Initialize trainable parameters
    x0 = start.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([x0], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    for step in range(max_iter):
        optimizer.zero_grad()  # Reset gradients
    
        # Compute loss using trainable and fixed parameters
        loss = dis.loss_function(x0, **kwargs)
    
        history[i, step] = loss.item()
    
        # Compute gradients
        loss.backward()
    
        # Perform optimization step
        optimizer.step()
        scheduler.step(loss)
        lr = scheduler.get_last_lr()
        
        # Print progress
        iteration = f'Iteration: {step+1}/{max_iter}'
        status = f'Loss: {loss.item():.2E}'
        current_lr = f'Learning rate: {lr}'
        sys.stdout.write('\r' + iteration + '\t' + status+ '\t' + current_lr)
        sys.stdout.flush()
        
    result[i] = x0.clone()

idx = torch.argmin(history[:,-1])

# Print final optimized parameters
print("\nOptimized parameters:", result[idx].detach().numpy())

elapsed = time.time() - curr

print(f'\nTime = {elapsed:.2f} s')

#%%

plt.figure()
plt.semilogy(np.asarray(history).T)
plt.xlabel('Iteration')
plt.ylabel('Loss')

#%%

zernike_idx = np.arange(zer_max)

zernike_output = torch.cat([torch.zeros(FIXED_ZERNIKE), result[idx]]).cpu().detach().numpy()

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
exPar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), result[idx].detach().numpy()])

emPar.abe_index = zernike_idx
emPar.abe_ampli = np.concatenate([np.zeros(FIXED_ZERNIKE), result[idx].detach().numpy()])

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
