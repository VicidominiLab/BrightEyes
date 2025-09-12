import torch
from scipy.signal import fftconvolve

import matplotlib.pyplot as plt
import numpy as np
import math

from. utils import median_filter, generate_truncated_exponential, plot_dataset

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
        self.kernel = None  # shape (time, channel)
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


    def fit_exponential(self, offset = 0, lr=1e-2, steps=1000):
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

        device, dtype = self.data.device, self.data.dtype
        # Parameters: per-channel A, C ; shared k
        A = (self.data.max(dim=0).values - self.data.min(dim=0).values).clone().detach().requires_grad_(True)

        Cparam = self.data.min(dim=0).values.clone().detach().requires_grad_(True)

        k = torch.tensor(0.1 / self.dt, device=device, dtype=dtype, requires_grad=True)

        opt = torch.optim.Adam([A, Cparam, k], lr=lr)


        for _ in range(steps):
            opt.zero_grad()
            loss = 0.0
            for c in range(self.C):
                y = self.data[self.t0[c]+offset:self.t1[c]+1, c]
                x = torch.arange(len(y), device=device, dtype=dtype) * self.dt
                y_pred = A[c] * torch.exp(-k * x ) + Cparam[c]
                loss = loss + torch.mean((y - y_pred) ** 2)
            loss = loss / self.C
            loss.backward()
            opt.step()

        self.params = {"A": A.detach(), "C": Cparam.detach(), "k": k.detach().item()}


    def generate_data_fit(self):
        """
        Generate truncated exponential fits for all channels using fitted parameters.
        Stores result in self.data_fit.
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        exp_curves = torch.zeros_like(self.data)

        for c in range(self.C):
            params = {
                "A": self.params["A"][c],
                "C": self.params["C"][c],
                "k": self.params["k"],
                "t0": int(self.t0[c])*self.dt,
            }
            exp_curves[:, c] = generate_truncated_exponential(self.time, params)

        self.data_fit = exp_curves


    def generate_kernel(self):
        """
        Generate exponential decays to be used for deconvolution.
        Stores result in self.kernel.
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        params = {
            "A": 1,
            "C": 0,
            "k": self.params["k"],
            "t0": 0,  #(self.T // 2) * self.dt,
        }

        exp_curve = generate_truncated_exponential(self.time, params)
        exp_curves = torch.clamp(exp_curve, min=0) # enforce positivity
        exp_curve /= exp_curve.sum()  # normalize kernel

        self.kernel = exp_curves


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
        psf = self.kernel.cpu().numpy().astype(np.float64)

        for c in range(self.C):
            y = self.data[:, c].cpu().numpy().astype(np.float64) - C[c].item()
            y = np.clip(y, 0, None)

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
        self.generate_data_fit()
        self.generate_kernel()
        self.richardson_lucy_deconvolution(iterations=rl_iterations)


    def plot_raw_and_fit(self):
        """
        Plot raw data (points) and fitted exponential (line only in [t0, t1])
        for each channel.
        """

        if self.params is None or not hasattr(self, "data_fit"):
            raise RuntimeError("Run fit_exponential() and generate_data_fit() first.")

        time = self.time.cpu().numpy()
        raw = self.data.cpu().numpy()
        fit = self.data_fit.cpu().numpy()

        # First, plot raw data as scatter (points)
        fig, ax = plot_dataset(time, raw, color="k", linestyle="none", marker='.')
        fig, ax = plot_dataset(time, fit, color="r", linestyle="-", fig = fig, ax=ax)


        fig.legend(["Raw", "Fit"], loc="upper right", bbox_to_anchor=(0.95, 0.95))


    def plot_forward_model(self):
        """
        Convolve estimated IRFs with fitted truncated exponential
        and plot them against the input data, per channel.
        """

        if self.irf is None:
            raise RuntimeError("Run richardson_lucy_deconvolution() first.")
        if self.kernel is None:
            raise RuntimeError("Run generate_kernel() first.")

        time = self.time.cpu().numpy()
        raw = self.data.cpu().numpy()
        forward = np.zeros_like(raw)

        psf = self.kernel.cpu().numpy()

        for c in range(self.C):
            irf = self.irf[:, c].cpu().numpy()
            forward[:, c] = fftconvolve(irf, psf, mode="same")[: self.T] + self.params["C"][c].item()
            forward[:, c] += self.params["C"][c].item()

        # First, plot raw data as scatter
        fig, ax = plot_dataset(time, raw, color="k", linestyle="none", marker='.')
        # Then add forward model as line
        fig, ax = plot_dataset(time, forward, color="g", linestyle="-", fig =fig, ax=ax)


        fig.legend(["Measured", "IRF ⊗ Exp"], loc="upper right", bbox_to_anchor=(0.95, 0.95))
