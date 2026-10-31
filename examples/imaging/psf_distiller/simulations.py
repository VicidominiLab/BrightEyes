# Origin: VicidominiLab/psf-distiller, demo/simulations.py at 4d927b1.
import numpy as np
import matplotlib.pyplot as plt

import brighteyes.imaging.simulation.PSF_sim as sim
import brighteyes.imaging.analysis.Graph_lib as gr

import torch

process = 'gpu'

#%%

# plt.close('all')

#%%
zer_max = 15

exPar = sim.simSettings()
exPar.na = 1.4    # numerical aperture
exPar.wl = 640    # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90   # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 101

exPar.abe_index = []
exPar.abe_ampli = []

zernike_idx = np.arange(zer_max)
zernike_input = np.zeros(zer_max)
zernike_input[exPar.abe_index] = exPar.abe_ampli

exPar.abe_index = zernike_idx
exPar.abe_ampli = zernike_input

names = exPar.aberration

#%%

emPar = exPar.copy()
emPar.wl = 660

emPar.abe_index = [4]
emPar.abe_ampli = [0]

zernike_idx = np.arange(zer_max)
zernike_input = np.zeros(zer_max)
zernike_input[exPar.abe_index] = exPar.abe_ampli

exPar.abe_index = zernike_idx
exPar.abe_ampli = zernike_input

#%%

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

#%%

psf, *_ = sim.SPAD_PSF_2D(grid, exPar, emPar)

#%%

gr.ShowDataset(psf)
gr.ShowFingerprint(psf)
