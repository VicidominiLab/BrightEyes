import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from scipy.signal import convolve
import os

from tqdm import trange

import brighteyes_ism.simulation.PSF_sim as sim
import brighteyes_ism.analysis.Graph_lib as gra
import brighteyes_ism.analysis.APR_lib as apr

from skimage.filters import gaussian
import brighteyes_ism.simulation.Tubulin_sim as simTub

from s2ism import s2ism as amd
import s2ism.psf_estimator as est

import brighteyes_flim.flism as flim

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
Ntau = 4
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

    phTub = np.empty((Ntau, tubulin.xy_dimension, tubulin.xy_dimension))

    for n in range(Ntau):
        phTub[n] = np.squeeze(simTub.functionPhTub(tubulin))
        phTub[n] += np.squeeze(simTub.functionPhTub(tubulin)).T
        phTub[n] += np.fliplr(np.squeeze(simTub.functionPhTub(tubulin)))
        # phTub[n] += np.flipud(np.squeeze( simTub.functionPhTub(tubulin) ))
        # if n==0:
        # phTub[n] += np.fliplr(np.squeeze( simTub.functionPhTub(tubulin) ))
        # phTub[n] += np.flipud(np.squeeze( simTub.functionPhTub(tubulin) ))
        # phTub[n] += np.fliplr(np.squeeze( simTub.functionPhTub(tubulin) ))
        # phTub[n] += np.flipud(np.squeeze( simTub.functionPhTub(tubulin) ))
        # phTub[n] += np.fliplr(np.squeeze( simTub.functionPhTub(tubulin) ))
        # phTub[n] += np.flipud(np.squeeze( simTub.functionPhTub(tubulin) ))

    np.savez(file_gt, phantom=phTub, tub_settings=tubulin)

# %%

fig, ax = plt.subplots(1, Ntau)
for z in range(Ntau):
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

# ground_truth = np.repeat(phTub, 3*nbin).reshape((4, Nx, Nx, 3*nbin))

ground_truth = phTub * flux


#%%

decay = np.zeros((Ntau, 3*nbin))

tau = np.asarray([3,6,1,9]) # ns


for z in range(Ntau):
    for tc in [-1, 0, 1]:
        decay[z] += exp_decay(tau[z], time-nbin*dt-tc*period)
        
#%%

irf_fin = np.empty((Ntau,nbin*3,N))

for i in range(Ntau):
    for c in range(N):
        irf_fin[i,...,c] = convolve(decay[i], irf[...,c], mode='same')
    
plt.figure()
plt.plot(irf_fin[0].sum(-1))
plt.plot(irf_fin[1].sum(-1))
plt.plot(irf_fin[2].sum(-1))
plt.plot(irf_fin[3].sum(-1))
     
    #%%
    
irf_fin_focus = irf_fin[0:2]
irf_fin_bkg = irf_fin[2:]

        
psf_irf_focus = np.einsum('jkl , mnl -> mjknl', psf[1], irf_fin_focus)   #(z=f tau x y t ch)
psf_irf_bkg = np.einsum('jkl , mnl -> mjknl', psf[0], irf_fin_bkg)   #(z=oof tau x y t ch)

# psf_irf = est.combine_psf_irf(psf, irf)

for z in range(Nz):
    psf_irf_focus[z] = psf_irf_focus[z]/psf_irf_focus[z].sum()
        

for z in range(Nz):
    psf_irf_bkg[z] = psf_irf_bkg[z]/psf_irf_bkg[z].sum()

#%% dataset

dataset_focus = np.empty([ 2, Nx, Nx, 3*nbin, N])
dataset_bkg = np.empty([ 2, Nx, Nx, 3*nbin, N])

ground_truth_bkg = ground_truth[:2]
ground_truth_focus = ground_truth[2:]


for c in range(N):
        for j in range(2):
            for l in range(3*nbin):
                dataset_focus[j, :, :, l, c] = convolve(ground_truth_focus[j], psf_irf_focus[ j, :, :,l, c], mode='same')
                
                
for c in range(N):
        for j in range(2):
            for l in range(3*nbin):
                dataset_bkg[j, :, :, l, c] = convolve(ground_truth_bkg[j], psf_irf_bkg[j, :, :,l, c], mode='same')

#%%
dataset = dataset_focus +dataset_bkg
dataset[dataset < 0] = 0
d = dataset.sum(axis=((0)))

dset = np.random.poisson(d)

#%%


psf_irf_focus = psf_irf_focus[..., nbin:-nbin, :]

psf_irf_bkg = psf_irf_bkg[..., nbin:-nbin, :]


dset = dset[..., nbin:-nbin, :]

#%%

dset_sum = dset.sum(-1)

intensity_sum = dset_sum.sum(-1)

gra.ShowImg(intensity_sum, grid.pxsizex*1e-3)

#%% apr

sv, _ = apr.ShiftVectors(dset.sum(-2), 50, 12)

dset_apr = np.empty_like(dset)

for t in trange(nbin):
    dset_apr[:,:,t] = apr.Reassignment(sv, dset[:,:,t])
    
dset_apr = dset_apr.sum(-1)

intensity_apr = dset_apr.sum(-1)

#%%
import s2ism_unmixing_lib as us2ism

psf_irf_finale= np.stack((psf_irf_bkg, psf_irf_focus))

n_iter = 40

#%%

file_rec = 'recon.npy'
analyze = True

if os.path.exists(file_rec) and not analyze:
    reconstruction = np.load(file_rec)
else:
    reconstruction, *_ = us2ism.s2ism_unmixing(dset, psf_irf_finale, max_iter=n_iter, rep_to_save='all',process ='cpu')
    np.save(file_rec, reconstruction)
    
    #%%

rec_focus= reconstruction[:,1]

rec_bkg = reconstruction[:,0]


#%%
chosen_iter = -1

intensity_focus_1 = reconstruction[chosen_iter, 1,0]
intensity_focus_2 = reconstruction[chosen_iter, 1,1]

intensity_bkg_1 = reconstruction[chosen_iter, 0,0]
intensity_bkg_2 = reconstruction[chosen_iter, 0,1]

#%%

gra.ShowImg(dset.sum(-1).sum(-1),1)

fig, ax = plt.subplots(1, Ntau)
for z in range(Ntau):
    ax[z].imshow(phTub[z], cmap='magma')
    ax[z].axis('off')

gra.ShowImg(rec_focus[-1,0],1)
gra.ShowImg(rec_focus[-1,1],1)

gra.ShowImg(rec_bkg[-1,0],1)
gra.ShowImg(rec_bkg[-1,1],1)

#%%
