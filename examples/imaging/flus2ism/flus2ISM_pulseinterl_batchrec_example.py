# Origin: VicidominiLab/flus2ISM, flus2ISM_pulseinterl_batchrec_example.py at 164a45b.
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
import os
import sys

from tqdm import trange

import brighteyes.imaging.simulation.PSF_sim as sim
import brighteyes.imaging.analysis.Graph_lib as gra
import brighteyes.imaging.analysis.APR_lib as apr

from skimage.filters import gaussian
import brighteyes.imaging.simulation.Tubulin_sim as simTub

from scipy.ndimage import fourier_shift, shift

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import brighteyes.imaging.flus2ism.flus2ISM_lib as us2ism

# %%

plt.close('all')

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# %% psf

Nz = 1

Nx = 31

N = 5

exc_wl = [488]   #nm

em_wl = [510]  #nm

wls_exc = len(exc_wl)

wls_em = len(em_wl)


file_psf = 'PSFs_comp_batchtest.npy'

if os.path.exists(file_psf):
    psf_comp = np.load(file_psf)
    
else:
    sz = [Nz] + [wls_exc] + [wls_em] +  [Nx] + [Nx] + [N**2]
    psf_comp = np.zeros((sz))

    
    for i in range(wls_exc):
        for j in range(wls_em):

        
            b = np.finfo(float).eps
            
            exPar = sim.simSettings()
            exPar.na = 1.4   # numerical aperture
            exPar.wl = exc_wl[i]   # wavelength [nm]
            exPar.gamma = 45  # parameter describing the light polarization
            exPar.beta = 90  # parameter describing the light polarization
            exPar.n = 1.5
            exPar.mask_sampl = 151
            
            emPar = exPar.copy()
            emPar.wl = em_wl[j]
            
            grid = sim.GridParameters()
            grid.Nz = Nz
            grid.pxsizex = 40
            grid.pxsizez = 720
            grid.pxpitch = 75e3
            grid.pxdim = 50e3
            grid.N = N
            grid.Nx = Nx
            
           
            psf_comp[:,i,j] = sim.SPAD_PSF_3D(grid, exPar, emPar)[0]
        
    np.save(file_psf, psf_comp)
       


# %% phantom object

file_gt = 'simulated_tubulin_pi_batchtest.npz'

Nx =201
Nstruct = 2
N = 5

if os.path.exists(file_gt):
    phTub = np.load(file_gt)['phantom']
else:
    tubulin = simTub.tubSettings()
    tubulin.xy_pixel_size = grid.pxsizex
    tubulin.xy_dimension = grid.Nx
    tubulin.xz_dimension = 1
    tubulin.z_pixel = 1
    tubulin.n_filament = 3
    tubulin.radius_filament = 25
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

def exp_decay(tau, t, a=1):
    decay = a*np.exp(-t/tau)
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

# t0 = time[len(time)//2]
t0 =time[101]

par = np.array([0.3, 2]) # ns

irf_single = smooth_irf(time - t0, *par)
irf = np.repeat(irf_single, N**2).reshape((3*nbin, N**2))


#%%
flux = 100

ground_truth = phTub * flux

#%%

tau_values = np.asarray([3,6]) # ns    WE ARE IMPOSING TWO SPECIES WITH BOTH INFOCUS AND OOF EMITTERS

Ntau = len(tau_values)


decay = np.zeros((Ntau, 3*nbin))

for z in range(Ntau):
    for tc in [-1, 0, 1]:
        decay[z] += exp_decay(tau_values[z], time-nbin*dt-tc*period)
        
#%%
Np = 1  #number of Pulses

T_ex = 3*nbin*dt  # [ns]


irf_pulse= np.empty((Np,nbin*3,N**2))

irf_fin = np.empty((Np,Ntau,nbin*3,N**2))


for k in range(Ntau):
    for c in range(N**2):
            irf_fin[0,k,...,c] = convolve(decay[k], irf[...,c], mode='same')
        
  

#%%adjusting manually excitation pulse and starting of the decay on temporal axis




#%%
    
plt.figure()
plt.plot(irf_fin[0,1,nbin:-nbin].sum(-1))

plt.plot(irf_fin[0,0,nbin:-nbin].sum(-1))



     
#%% combining PSFs with IRFs

irf_t = np.flip(irf_fin,2)  
        
psf_irf = np.einsum('ijklmo , jkno ->ijklmno', psf_comp, irf_t)      #(z p tau x y ch)  + (p tau t ch)   --> (z p tau x y t ch)

psf_irf = np.swapaxes(psf_irf, 1, 2)    #(z tau p x y t ch)




#%%

for m in range(Ntau):
    psf_irf[:,m] = psf_irf[:,m]/psf_irf[:,m].sum()  
        

#%%

                
dataset_t = np.empty([Np, 2, Nx, Nx, 3*nbin, N**2])

for p in range(Np):            
    for c in range(N**2):
            for j in range(2):
                for l in range(3*nbin):
                    dataset_t[p,j, :, :, l, c] = convolve(ground_truth[j], psf_irf[0,j,p, :, :,l, c], mode='same') 

#%%
dataset = dataset_t

dataset[dataset < 0] = 0

d = dataset.sum(axis=(0,1))

del dataset


dset = np.random.poisson(d)

del d



#%%

dset = dset[..., nbin:-nbin, :]

psf_irf = psf_irf[...,nbin:-nbin,:]

#%% PLOTTING THE RAW TEMPORAL DECAY

plt.figure()
plt.plot(dset.sum(axis=(0,1,3)))

#%%

dset_sum = dset.sum(-1)

intensity_sum = dset_sum.sum(-1)

gra.ShowImg(intensity_sum, 0.04)

del dset_sum

#%%

n_iter = 20

#%%

dset_crop = dset[1:,1:]
batch_lenght = 54
overlap = 4


#%%

file_rec = 'recon_flus2ism_pulseinterl_batchtest.npy'

analyze = True

if os.path.exists(file_rec) and not analyze:
    reconstruction = np.load(file_rec)
else:
    reconstruction, *_ = us2ism.batch_reconstruction_flus2ism_pulse_inter(dset_crop, psf_irf, batch_size=(batch_lenght,batch_lenght), overlap=overlap, max_iter = 20)
    np.save(file_rec, reconstruction)

    
    #%%

rec_focus= reconstruction[1]

rec_bkg = reconstruction[0]

#%% CHECKING FLUX CONSERVATION

print(dset.sum())
print(reconstruction.sum())

#%%
chosen_iter = -1

#%%

clabel ='intensity (arb.units)'
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 20),layout="constrained")
axd =fig.subplot_mosaic(
    """
    AABC
    AADE
   
    
    """
)

gra.ShowImg(dset.sum(axis=(-2,-1)), 0.04, cmap='hot', fig = fig, ax = axd["A"], clabel =clabel)
axd["A"].set_title('open CLSM')

gra.ShowImg(ground_truth[0],  0.04, cmap='hot', fig = fig, ax = axd["B"], clabel =clabel)
axd["B"].set_title(f'g.t. oof, tau = {tau_values[0]} ns')

gra.ShowImg(ground_truth[1],  0.04, cmap='hot', fig = fig, ax = axd["C"], clabel =clabel)
axd["C"].set_title(f'g.t. oof, tau = {tau_values[1]} ns')


gra.ShowImg(rec_focus,  0.04, cmap='hot', fig = fig, ax = axd["D"], clabel =clabel)
axd["D"].set_title(f' flus2ISM oof, tau = {tau_values[0]} ns')


gra.ShowImg(rec_bkg,  0.04, cmap='hot', fig = fig, ax = axd["E"], clabel =clabel)
axd["E"].set_title(f'flus2ISM focus, tau = {tau_values[0]} ns')

