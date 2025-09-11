import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
import numpy as np
import math

from. utils import median_filter

class Birfi:

    def __init__(self, data: torch.Tensor, dt: float = 1.0):
        data = torch.as_tensor(data)
        if data.dim() == 1:
            self.data = data.unsqueeze(1).clone()
        elif data.dim() == 2:
            self.data = data.clone()
        elif data.dim() > 2:
            raise ValueError("data must be 1D or 2D tensor")

        self.dt = dt
        self.time = torch.arange(self.data.shape[0], device=self.data.device) * dt

        self.T, self.C = self.data.shape
        self.t0 = None   # shape (channel,)
        self.t1 = None   # shape (channel,)
        self.params = None  # dict with A, k, C
        self.data_fit = None # shape (time, channel)
        self.irf = None  # shape (time, channel)


    def find_t0_t1(self, median_window: int = 5):
        t0s, t1s = [], []
        for c in range(self.C):
            d = torch.diff(self.data[:, c])
            d = median_filter(d, median_window)
            t0 = torch.argmin(d).item()
            post = d[t0+1:]
            nonneg = torch.where(post >= 0)[0]
            t1 = t0 + 1 + nonneg[0].item() if len(nonneg) > 0 else self.T - 1
            t0s.append(t0)
            t1s.append(t1)
        self.t0 = torch.tensor(t0s)
        self.t1 = torch.tensor(t1s)
        return self.t0, self.t1


    def fit_exponential(self, offset = 0, lr=1e-2, steps=1000):
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

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
            for c in range(self.C):
                y = self.data[self.t0[c]+offset:self.t1[c]+1, c]
                x = torch.arange(len(y), device=device, dtype=dtype)
                y_pred = A[c] * torch.exp(-k * x) + Cparam[c]
                loss = loss + torch.mean((y - y_pred) ** 2)
            loss = loss / self.C
            loss.backward()
            opt.step()

        self.params = {"A": A.detach(), "C": Cparam.detach(), "k": k.detach().item()}
        return self.params


    def generate_truncated_exponential(self):
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")
        A, C, k = self.params["A"], self.params["C"], self.params["k"]
        exp_curves = torch.zeros_like(self.data)
        for c in range(self.C):
            start = self.t0[c].item()
            x_local = torch.arange(self.T - start, device=self.data.device, dtype=self.data.dtype)
            exp_local = A[c] * torch.exp(-k * x_local) + C[c]
            exp_curves[start:, c] = exp_local
        self.data_fit = exp_curves
        return exp_curves


    def plot_raw_and_fit(self):
        """
        Plot raw data (points) and fitted exponential (line only in [t0, t1])
        for each channel in an adaptive grid.
        """
        if self.params is None or not hasattr(self, "data_fit"):
            raise RuntimeError("Run fit_exponential() and generate_truncated_exponential() first.")

        num_channels = self.data.shape[1]
        ncols = math.ceil(math.sqrt(num_channels))
        nrows = math.ceil(num_channels / ncols)

        fig, ax = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows), sharex=True, sharey=True)
        ax = np.array(ax).reshape(-1)

        time = self.time.numpy()

        for n in range(num_channels):
            # Raw data as points
            ax[n].plot(time, self.data[:, n].numpy(), 'o', markersize=3, color='k', label='Raw')

            # Fit curve only in [t0, t1]
            t0, t1 = int(self.t0[n]), int(self.t1[n])
            ax[n].plot(time[t0:t1 + 1], self.data_fit[t0:t1 + 1, n].numpy(), '-', color='r', label='Fit')

            # Title = channel index
            ax[n].set_title(f'Channel {n}', fontsize=9)

            # Labels only on edges
            if n % ncols == 0:
                ax[n].set_ylabel('Intensity')
            if n // ncols == nrows - 1:
                ax[n].set_xlabel('Time (ns)')

        # Hide empty subplots
        for n in range(num_channels, len(ax)):
            ax[n].axis('off')

        # Shared legend
        fig.legend(['Raw', 'Fit'], loc='upper right', bbox_to_anchor=(0.95, 0.95))
        fig.tight_layout()


    def richardson_lucy_deconvolution(self, iterations=50, eps=1e-8):
        """
        Perform Richardson-Lucy deconvolution on each channel of self.data
        using a truncated exponential (starting at zero, no offset) as PSF.

        Returns:
            irf: torch.Tensor of shape (time, channel)
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential() first.")

        A, C, k = self.params["A"], self.params["C"], self.params["k"]
        irf = torch.zeros_like(self.data)

        for c in range(self.C):
            y = self.data[:, c].cpu().numpy().astype(np.float64) - C[c].item()
            y = np.clip(y, 0, None)

            # --- PSF: truncated exponential starting at 0 ---
            #x = np.arange(self.T, dtype=np.float64)
            #psf = np.exp(-k * x) * A[c].item()

            psf = self.data_fit[:, c].cpu().numpy().astype(np.float64) - C[c].item()
            psf = np.clip(psf, 0, None)
            psf /= psf.sum()  # normalize PSF

            # Initialize estimate
            x_est = np.ones_like(y)

            for _ in range(iterations):
                conv = fftconvolve(x_est, psf, mode="same")
                conv = np.clip(conv, eps, None)  # avoid div by 0
                relative_blur = y / conv
                correction = fftconvolve(relative_blur, psf[::-1], mode="same")
                x_est *= correction
                x_est = np.clip(x_est, 0, None)  # enforce positivity

            irf[:, c] = torch.from_numpy(x_est.astype(np.float32)).to(self.data.device)

        self.irf = irf
        return self.irf


    def run(self, lr=1e-2, steps=1000, rl_iterations=200):
        """
        Complete pipeline to generate IRF:
        1. Find t0, t1 per channel
        2. Fit truncated exponential
        3. Generate truncated exponential
        4. Perform Richardson-Lucy deconvolution

        Returns:
            irf: torch.Tensor of shape (time, channel)
        """

        self.find_t0_t1()
        self.fit_exponential(lr=lr, steps=steps)
        self.generate_truncated_exponential()
        self.richardson_lucy_deconvolution(iterations=rl_iterations)


    def plot_forward_model(self):
        """
        Convolve estimated IRFs with fitted truncated exponential
        and plot them against the input data, per channel.
        """

        if self.irf is None:
            raise RuntimeError("Run richardson_lucy_deconvolution() first.")
        if self.data_fit is None:
            raise RuntimeError("Run generate_truncated_exponential() first.")

        from scipy.signal import fftconvolve

        num_channels = self.C
        ncols = math.ceil(math.sqrt(num_channels))
        nrows = math.ceil(num_channels / ncols)

        fig, ax = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))#, sharex=True, sharey=True)
        ax = np.array(ax).reshape(-1)

        time = self.time.cpu().numpy()

        for c in range(num_channels):
            y_true = self.data[:, c].cpu().numpy()
            irf = self.irf[:, c].cpu().numpy()
            psf = self.data_fit[:, c].cpu().numpy()
            psf /= psf.sum() + 1e-12

            # Forward convolution (same length as data)
            y_recon = fftconvolve(irf, psf, mode="same")[: self.T] + self.params["C"][c].item()

            # Plot measured data (points)
            ax[c].plot(time, y_true, "o", markersize=3, color="k", label="Measured")

            # Plot forward model (line)
            ax[c].plot(time, y_recon, "-", color="g", label="IRF ⊗ Exp")

            ax[c].set_title(f"Channel {c}", fontsize=9)
            if c % ncols == 0:
                ax[c].set_ylabel("Intensity")
            if c // ncols == nrows - 1:
                ax[c].set_xlabel("Time (ns)")

        # Hide unused subplots
        for c in range(num_channels, len(ax)):
            ax[c].axis("off")

        # Shared legend
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95))
        fig.tight_layout()
        plt.show()
