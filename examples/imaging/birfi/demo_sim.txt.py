import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from birfi.birfi import Birfi

#%%

plt.close('all')

#%% --- Parameters ---
T = 200                # number of time points
C = 9                 # number of channels
dt = 0.01              # time step
true_k = 0.1           # shared decay rate
noise_level = 0.00     # additive noise

time = torch.arange(T, dtype=torch.float32) * dt

#%% --- Simulate narrow Gaussian IRFs ---

torch.manual_seed(0)  # reproducibility
irfs = torch.zeros(T, C)
for c in range(C):
    peak_idx = torch.randint(80, 120, (1,)).item()   # random peak location
    sigma = 3
    t = torch.arange(T)
    irf = torch.exp(-0.5 * ((t - peak_idx)/sigma)**2)
    irf /= irf.sum()  # normalize
    irfs[:, c] = irf


#%% --- Generate truncated exponential decays starting from IRF peak ---

data = np.zeros((T, C))
for c in range(C):
    peak_idx =  T//2
    x_exp = np.zeros(T)
    x_local = np.arange(T - peak_idx)
    x_exp[peak_idx:] = np.exp(-true_k * x_local)  # truncated exponential
    # Convolve using fftconvolve
    conv = fftconvolve(irfs[:, c], x_exp, mode='same')  # take first T points
    data[:, c] = conv
    data[:, c] += noise_level * np.random.randn(T)          # add noise

data = torch.tensor(data)

#%%

plt.figure()

plt.plot(x_exp)

plt.figure()

plt.plot(data)


#%% --- Initialize Birfi and run pipeline ---

b = Birfi(data, dt=dt)
b.run(lr=5e-2, steps=2000, rl_iterations=100)

retrieved_irf  = b.irf

#%%

b.plot_raw_and_fit()

#%% --- Plot example channels ---

fig, ax = plt.subplots(3,3, figsize=(12,10))
ax = np.array(ax).reshape(-1)

for c in range(C):
    ax[c].plot(time.numpy(), b.data_fit[:, c].numpy(), '-', color='C0')
    ax[c].set_title(f'Channel {c}')
    ax[c].set_xlabel('Time')
    ax[c].set_ylabel('Intensity')

for c in range(C, len(ax)):
    ax[c].axis('off')

fig.tight_layout()
plt.show()

#%% --- Plot example channels ---

fig, ax = plt.subplots(3, 3, figsize=(12,10))
ax = np.array(ax).reshape(-1)

for c in range(C):
    # ax[c].plot(time.numpy(), irfs[:, c].numpy(), '-', color='C0', label='true IRF')
    ax[c].plot(time.numpy(), retrieved_irf[:, c].numpy(), '-', color='C1', label='Retrieved IRF')
    ax[c].set_title(f'Channel {c}')
    ax[c].set_xlabel('Time')
    ax[c].set_ylabel('Intensity')

for c in range(C, len(ax)):
    ax[c].axis('off')

fig.tight_layout()
plt.show()


#%%

b.plot_forward_model()