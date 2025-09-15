import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from birfi.birfi import Birfi
from birfi.utils import plot_dataset, partial_convolution, generate_truncated_exponential

#%%

plt.close('all')

#%% --- Parameters ---

T = 200                # number of time points
C = 9                 # number of channels
dt = 0.01              # time step
true_k = 0.1           # shared decay rate
noise_level = 0.00     # additive noise
bkg_level = 0.10        # background strength

time = torch.arange(T, dtype=torch.float32) * dt

#%% --- Simulate multi-Gaussian IRFs ---

torch.manual_seed(0)  # reproducibility

irfs = torch.zeros(T, C)

for c in range(C):
    peak_idx_1 = torch.randint(T//2 - 0, T//2 + 20, (1,)).item()   # random peak location
    sigma = 4
    t = torch.arange(T)
    irf = torch.exp(-0.5 * ((t - peak_idx_1)/sigma)**2)
    peak_idx_2 = peak_idx_1 + torch.randint(-15, 15, (1,)).item()   # random peak location
    irf +=0.7 * torch.exp(-0.5 * ((t - peak_idx_2)/sigma)**2)
    irf /= irf.sum()  # normalize
    irfs[:, c] = irf

#%%

_ = plot_dataset(time.numpy(), irfs)

#%% --- Generate truncated exponential decays ---

params = {'A' : 1,
          'k' : true_k/dt,
          'C' : bkg_level,
          't0': 0, #dt*(T//4),
    }

exp_decay = generate_truncated_exponential(time, params)

plt.figure()
plt.plot(time, exp_decay)

#%%

data = partial_convolution(exp_decay, irfs, dim1 = 'x', dim2 = 'xc', axis = 'x', fourier = (0,0)) # convolve exponential with IRFs
data += bkg_level*torch.rand((1,C,)) # add background
data += noise_level * torch.rand((T,C)) # add noise
    
#%%

_ = plot_dataset(time, data)

#%% --- Initialize Birfi and run pipeline ---

b = Birfi(data, dt=dt)
retrieved_irf = b.run(lr=5e-2, steps=2000, rl_iterations=50)

#%%

_ = b.plot_raw_and_fit()

#%%

fig, ax = plot_dataset(b.time, irfs / irfs.max(dim = 0, keepdim=True).values)

fig, ax = plot_dataset(b.time, retrieved_irf / retrieved_irf.max(dim = 0, keepdim=True).values, fig = fig, ax = ax, color = 'C1', linestyle='dashed')

#%%

plt.figure();
plt.plot(b.kernel)

#%%

b.plot_forward_model()