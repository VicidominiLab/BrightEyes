import torch

import matplotlib.pyplot as plt
import numpy as np
import math

from. utils import median_filter

class Birfi:

    def __init__(self, data: torch.tensor, dt: float = 1.0):
        data = torch.as_tensor(data)
        if data.dim() == 1:
            self.data = data.unsqueeze(1).clone()
        elif data.dim() == 2:
            self.data = data.clone()
        elif data.dim() > 2:
            raise ValueError("data must be 1D or 2D tensor")

        self.dt = dt
        self.time = torch.arange(self.data.shape[0]) * dt

        self.t0 = None   # shape (channel,)
        self.t1 = None   # shape (channel,)
        self.params = None  # dict with A, k, C


    def find_t0_t1(self, median_window: int = 5):
        """
        Find t0, t1 for each channel separately.
        """
        T, C = self.data.shape
        t0s, t1s = [], []
        for c in range(C):
            d = torch.diff(self.data[:, c])
            d = median_filter(d, median_window)

            t0 = torch.argmin(d).item()
            post = d[t0+1:]
            nonneg = torch.where(post >= 0)[0]
            if len(nonneg) > 0:
                t1 = t0 + 1 + nonneg[0].item()
            else:
                t1 = T - 1
            t0s.append(t0)
            t1s.append(t1)

        self.t0 = torch.tensor(t0s)
        self.t1 = torch.tensor(t1s)
        return self.t0, self.t1


    def fit_exponential(self, lr=1e-2, steps=1000):
        """
        Fit exponential decays for all channels.
        Shared k, per-channel A and C,
        each channel restricted to [t0_c, t1_c].
        """
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

        T, C = self.data.shape
        device, dtype = self.data.device, self.data.dtype

        # Parameters: per-channel A, C ; shared k
        A = torch.tensor(self.data.max(dim=0).values - self.data.min(dim=0).values,
                         device=device, dtype=dtype, requires_grad=True)
        Cparam = torch.tensor(self.data.min(dim=0).values,
                              device=device, dtype=dtype, requires_grad=True)
        k = torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)

        opt = torch.optim.Adam([A, Cparam, k], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            loss = 0.0
            for c in range(C):
                y = self.data[self.t0[c]:self.t1[c]+1, c]
                x = torch.arange(len(y), device=device, dtype=dtype)
                y_pred = A[c] * torch.exp(-k * x) + Cparam[c]
                loss = loss + torch.mean((y - y_pred) ** 2)
            loss = loss / C
            loss.backward()
            opt.step()

        self.params = {
            "A": A.detach(),
            "C": Cparam.detach(),
            "k": k.detach().item()
        }
        return self.params


    def generate_truncated_exponential(self):
        """
        Generate exponential decay curves for each channel,
        truncated at t0_c (zero before t0_c) and aligned with the peak.
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        A, C, k = self.params["A"], self.params["C"], self.params["k"]
        T, Cdim = self.data.shape
        device, dtype = self.data.device, self.data.dtype
        x = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T, 1)

        exp_curves = torch.zeros_like(self.data)
        for c in range(Cdim):
            # Create shifted exponential starting at t0[c]
            start = self.t0[c].item()
            x_local = torch.arange(T - start, device=device, dtype=dtype)
            exp_local = A[c] * torch.exp(-k * x_local) + C[c]
            exp_curves[start:, c] = exp_local  # zero before t0

        return exp_curves


    def plot_raw_and_fit(self, time=None):
        """
        Plot raw data and fitted exponential for each channel in an adaptive grid.
        Raw data: points, Fit: continuous line.
        """

        if self.params is None or not hasattr(self, 'data_fit'):
            raise RuntimeError("Fit the data first and generate fit_curve for all channels.")

        if time is None:
            time = np.arange(self.data.shape[0])

        num_channels = self.data.shape[1]
        ncols = math.ceil(math.sqrt(num_channels))
        nrows = math.ceil(num_channels / ncols)

        fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.5*nrows), sharex=True, sharey=True)
        ax = np.array(ax).reshape(-1)  # flatten for easy indexing

        for n in range(num_channels):
            # Plot raw as points
            ax[n].plot(time, self.data[:, n], 'o', markersize=3, color='C0', label='Raw')
            # Plot fit as line
            ax[n].plot(time, self.data_fit[:, n], '-', color='C1', label='Fit')

            ax[n].text(0.5, 0.8, f'channel {n}', transform=ax[n].transAxes, fontsize=8)
            ax[n].set_xlabel('time (ns)')
            ax[n].set_ylabel('Intensity')

        # Turn off empty subplots
        for n in range(num_channels, len(ax)):
            ax[n].axis('off')

        # Single legend
        fig.legend(['Raw', 'Fit'], loc='upper right', bbox_to_anchor=(1.05, 0.99))
        fig.tight_layout()
        plt.show()
