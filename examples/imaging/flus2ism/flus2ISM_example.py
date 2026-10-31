# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
import os
import string
from tqdm import trange

import brighteyes.imaging.simulation.PSF_sim as sim
import brighteyes.imaging.analysis.Graph_lib as gra
import brighteyes.imaging.analysis.APR_lib as apr

from skimage.filters import gaussian
import brighteyes.imaging.simulation.Tubulin_sim as simTub

from brighteyes.imaging.flus2ism.flus2ISM_lib import partial_convolution_rfft
import brighteyes.imaging.flus2ism.flus2ISM_lib as us2ism

# %%

plt.close('all')

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# %% psf

b = np.finfo(float).eps

exPar = sim.simSettings()
exPar.na = 1.4   # numerical aperture
exPar.wl = 640   # wavelength [nm]
exPar.gamma = 45  # parameter describing the light polarization
exPar.beta = 90  # parameter describing the light polarization
exPar.n = 1.5
exPar.mask_sampl = 151

emPar = exPar.copy()
emPar.wl = 660

grid = sim.GridParameters()
grid.Nz = 2
grid.pxsizex = 40
grid.pxsizez = 720
grid.pxpitch = 75e3
grid.pxdim = 50e3
grid.N = 5
grid.Nx = 101

file_psf = 'PSFs.npy'

if os.path.exists(file_psf):
    psf = np.load(file_psf)
else:
    psf = sim.SPAD_PSF_3D(grid, exPar, emPar)[0]
    np.save(file_psf, psf)

# %% phantom object

file_gt = 'simulated_tubulin.npz'

Nx = grid.Nx
Nstruct = 4
Nz = 2
N = grid.Nch

if os.path.exists(file_gt):
    phTub = np.load(file_gt)['phantom']
else:
    tubulin = simTub.tubSettings()
    tubulin.xy_pixel_size = grid.pxsizex
    tubulin.xy_dimension = grid.Nx
    tubulin.xz_dimension = 1
    tubulin.z_pixel = 1
    tubulin.n_filament = 1
    tubulin.radius_filament = 30
    tubulin.intensity_filament = [1, 1]

    phTub = np.empty((Nstruct, tubulin.xy_dimension, tubulin.xy_dimension))

    for n in range(Nstruct):
        phTub[n] = np.squeeze(simTub.functionPhTub(tubulin))
        phTub[n] += np.squeeze(simTub.functionPhTub(tubulin)).T
        phTub[n] += np.fliplr(np.squeeze(simTub.functionPhTub(tubulin)))
        

    np.savez(file_gt, phantom=phTub, tub_settings=tubulin)

# %%

fig, ax = plt.subplots(1, Nstruct)
for z in range(Nstruct):
    ax[z].imshow(phTub[z], cmap='magma')
    ax[z].axis('off')


# %%

def rect(t, w):
    r = np.where(abs(t) <= w/2, 1, 0)*1.
    r /= r.sum()
    return r

def exp_decay(tau, t):
    decay = np.exp(-t/tau)
    decay[t < 0] = 0
    return decay

def smooth_irf(t, s, w):
    irf_1 = rect(t, w)
    dt = t[1] - t[0]
    irf_2 = gaussian(irf_1, s/dt)
    irf_2 /= irf_2.sum()
    return irf_2

# %%
dt = 0.297619  # ns
nbin = 81
period = nbin * dt # ns
dfd_freq = 1e3/period # MHz

time = np.arange(3*nbin)*dt

t0 = time[len(time)//2]

par = np.array([0.3, 2]) # ns

irf_single = smooth_irf(time - t0, *par)
irf = np.repeat(irf_single, N).reshape((3*nbin, N))


#%%
flux = 100

ground_truth = phTub * flux

#%%

tau = np.asarray([3,6]) # ns    WE ARE IMPOSING TWO SPECIES WITH BOTH INFOCUS AND OOF EMITTERS

Ntau = len(tau)

decay = np.zeros((Ntau, 3*nbin))

for z in range(Ntau):
    for tc in [-1, 0, 1]:
        decay[z] += exp_decay(tau[z], time-nbin*dt-tc*period)
        
#%%

irf_fin = np.empty((Ntau,nbin*3,N))

# Update
str_irf = 'tc'
str_decay = 'mt'

irf_fin = partial_convolution_rfft(irf, decay, dim1=str_irf, dim2=str_decay, dim3='mtc', axis='t', fourier=(False, False))

#%%
    
plt.figure()
plt.plot(irf_fin[0].sum(-1))
plt.plot(irf_fin[1].sum(-1))

     
#%% combining PSFs with IRFs
        
psf_irf_focus = np.einsum('jkl , mnl -> mjknl', psf[1], irf_fin)   #(z=f tau x y t ch)
psf_irf_bkg = np.einsum('jkl , mnl -> mjknl', psf[0], irf_fin)   #(z=oof tau x y t ch)


#%% PSF-IRF normalizarion

for z in range(Nz):
    psf_irf_focus[z] = psf_irf_focus[z]/psf_irf_focus[z].sum()        

for z in range(Nz):
    psf_irf_bkg[z] = psf_irf_bkg[z]/psf_irf_bkg[z].sum()

#%% dataset

dataset_focus = np.empty([ 2, Nx, Nx, 3*nbin, N])
dataset_bkg = np.empty([ 2, Nx, Nx, 3*nbin, N])

ground_truth_bkg = ground_truth[:2]
ground_truth_focus = ground_truth[2:]

alphabet = string.ascii_lowercase
num_axis_to_conv = 2
str_psf = alphabet[:psf_irf_focus.ndim]
obj_str = alphabet[:ground_truth_focus.ndim]
axis_to_conv = alphabet[ground_truth_focus.ndim-num_axis_to_conv : ground_truth_focus.ndim]

#%% Partial Convolution
dataset_focus = partial_convolution_rfft(psf_irf_focus, ground_truth_focus, dim1=str_psf, dim2=obj_str, axis=axis_to_conv, fourier=(False, False))

# for c in range(N):
#         for j in range(2):
#             for l in range(3*nbin):
#                 dataset_focus[j, :, :, l, c] = convolve(ground_truth_focus[j], psf_irf_focus[ j, :, :,l, c], mode='same')
                
dataset_bkg = partial_convolution_rfft(psf_irf_bkg, ground_truth_bkg, dim1=str_psf, dim2=obj_str, axis=axis_to_conv, fourier=(False, False))              
                
# for c in range(N):
#         for j in range(2):
#             for l in range(3*nbin):
#                 dataset_bkg[j, :, :, l, c] = convolve(ground_truth_bkg[j], psf_irf_bkg[j, :, :,l, c], mode='same')

#%%
dataset = dataset_focus + dataset_bkg

dataset[dataset < 0] = 0
d = dataset.sum(axis=(0))

dset = np.random.poisson(d)

#%% DISPLAYING THE SIMULATED DATASET
plt.figure()
gra.ShowImg(dset.sum(axis=(-2,-1)), grid.pxsizex*1e-3, cmap='hot')

#%% PLOTTING THE RAW TEMPORAL DECAY

plt.figure()
plt.plot(dset.sum(axis=(0,1,3)))

#%%

psf_irf_focus = psf_irf_focus[..., nbin:-nbin, :]

psf_irf_bkg = psf_irf_bkg[..., nbin:-nbin, :]

dset = dset[..., nbin:-nbin, :]

#%%

dset_sum = dset.sum(-1)

intensity_sum = dset_sum.sum(-1)

gra.ShowImg(intensity_sum, grid.pxsizex*1e-3)

#%%

psf_irf_finale= np.stack((psf_irf_bkg, psf_irf_focus))

n_iter = 40

#%%
import importlib
importlib.reload(us2ism)

file_rec = 'fast_data' #recon_flus2ism.npy'

analyze = False

if os.path.exists(file_rec) and not analyze:
    reconstruction = np.load(file_rec)
else:
    reconstruction, *_ = us2ism.flus2ism(dset, psf_irf_finale, max_iter=n_iter, rep_to_save='all',process ='cpu')
    np.save(file_rec, reconstruction)
    
#%%
rec_focus= reconstruction[:,1]

rec_bkg = reconstruction[:,0]

#%% CHECKING FLUX CONSERVATION

print(dset.sum())
print(reconstruction[-1].sum())

#%%
chosen_iter = -1

#%%

clabel ='intensity (arb.units)'
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 20),layout="constrained")
axd =fig.subplot_mosaic(
    """
    AABCDE
    AAFGHI
   
    
    """
)

gra.ShowImg(dset.sum(axis=(-2,-1)), grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["A"], clabel =clabel)
axd["A"].set_title('open CLSM')

gra.ShowImg(ground_truth[0], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["B"], clabel =clabel)
axd["B"].set_title(f'g.t. oof, tau = {tau[0]} ns')

gra.ShowImg(ground_truth[1], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["C"], clabel =clabel)
axd["C"].set_title(f'g.t. oof, tau = {tau[1]} ns')


gra.ShowImg(ground_truth[2], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["D"], clabel =clabel)
axd["D"].set_title(f'g.t. focus, tau = {tau[0]} ns')

gra.ShowImg(ground_truth[3], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["E"], clabel =clabel)
axd["E"].set_title(f'g.t. focus, tau = {tau[1]} ns')


gra.ShowImg(rec_bkg[-1,0], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["F"], clabel =clabel)
axd["F"].set_title(f' flus2ISM oof, tau = {tau[0]} ns')

gra.ShowImg(rec_bkg[-1,1], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["G"], clabel =clabel)
axd["G"].set_title(f'flus2ISM oof, tau = {tau[1]} ns')


gra.ShowImg(rec_focus[-1,0], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["H"], clabel =clabel)
axd["H"].set_title(f'flus2ISM focus, tau = {tau[0]} ns')

gra.ShowImg(rec_focus[-1,1], grid.pxsizex*1e-3, cmap='hot', fig = fig, ax = axd["I"], clabel =clabel)
axd["I"].set_title(f'flus2ISM focus, tau = {tau[1]} ns')