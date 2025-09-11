import torch

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
        self.data_fit = None


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


    def fit_exponential(self, lr=1e-2, steps=1000):
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
                y = self.data[self.t0[c]:self.t1[c]+1, c]
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
        if self.data_fit is None:
            raise RuntimeError("Generate the fit first using generate_truncated_exponential().")

        num_channels = self.C
        ncols = math.ceil(math.sqrt(num_channels))
        nrows = math.ceil(num_channels / ncols)
        fig, ax = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.5*nrows), sharex=True, sharey=True)
        ax = np.array(ax).reshape(-1)

        time_np = self.time.cpu().numpy() if isinstance(self.time, torch.Tensor) else self.time

        for n in range(num_channels):
            ax[n].plot(time_np, self.data[:, n].cpu().numpy(), 'o', markersize=3, color='C0', label='Raw')
            ax[n].plot(time_np, self.data_fit[:, n].cpu().numpy(), '-', color='C1', label='Fit')
            ax[n].text(0.5, 0.8, f'channel {n}', transform=ax[n].transAxes, fontsize=8)
            ax[n].set_xlabel('time (ns)')
            ax[n].set_ylabel('Intensity')

        for n in range(num_channels, len(ax)):
            ax[n].axis('off')

        fig.legend(['Raw', 'Fit'], loc='upper right', bbox_to_anchor=(1.05, 0.99))
        fig.tight_layout()
        plt.show()

    def richardson_lucy_deconvolution(self, iterations=50, eps=1e-8):
        """
        Perform Richardson-Lucy deconvolution on each channel of self.data
        using the fitted truncated exponential as the PSF.
        Returns:
            irf: torch.Tensor of shape (time, channel)
        """
        if self.data_fit is None:
            raise RuntimeError("Run generate_truncated_exponential() first.")

        irf = torch.zeros_like(self.data)

        for c in range(self.C):
            y = self.data[:, c].clone()
            psf = self.data_fit[:, c].clone()
            psf = psf / psf.sum()  # normalize PSF

            # Initialize estimate with uniform or small positive values
            x_est = torch.ones_like(y)

            psf_flip = torch.flip(psf, dims=[0])

            for _ in range(iterations):
                # Convolve current estimate with PSF
                conv = F.conv1d(x_est.view(1, 1, -1), psf.view(1, 1, -1), padding=0).view(-1)
                conv = torch.clamp(conv, min=eps)
                relative_blur = y / conv
                # Convolve relative_blur with flipped PSF
                correction = F.conv1d(relative_blur.view(1, 1, -1), psf_flip.view(1, 1, -1), padding=0).view(-1)
                x_est = x_est * correction
                x_est = torch.clamp(x_est, min=0.0)

            irf[:, c] = x_est

        return irf

    def run(self, lr=1e-2, steps=1000, rl_iterations=50):
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
        irf = self.richardson_lucy_deconvolution(iterations=rl_iterations)
        return irf