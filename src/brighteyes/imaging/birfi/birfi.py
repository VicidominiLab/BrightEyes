import torch
from torch.fft import fftn
from scipy.signal import savgol_filter

from. utils import median_filter, generate_truncated_exponential, plot_dataset, partial_convolution, estimate_lifetime

class Birfi:
    """
    Estimate instrument response functions (IRFs) from time-series data using truncated exponential fitting and Richardson-Lucy deconvolution.

    Attributes:
        data (torch.Tensor): Input time-series data (shape: [num_samples, num_channels]).
        dt (float): Time step between samples.
        time (torch.Tensor): Time vector.
        num_samples (int): Number of time points.
        num_channels (int): Number of channels.
        t0 (torch.Tensor|None): Per-channel start index of decay.
        t1 (torch.Tensor|None): Per-channel end index of decay.
        params (dict|None): Fitted parameters ("A", "C", "k").
        data_fit (torch.Tensor|None): Fitted exponential curves.
        kernel (torch.Tensor|None): Kernel for deconvolution.
        irf (torch.Tensor|None): Deconvolved IRFs.
    """
    def __init__(self, data: torch.Tensor, dt: float = 1.0, device = None):
        """
        Initialize Birfi with time-series data and time step.

        Args:
            data (torch.Tensor): 1D or 2D tensor of time-series data. If 1D, promoted to shape (num_samples, 1).
            dt (float): Sampling interval (default: 1.0).

        Raises:
            ValueError: If data has more than 2 dimensions.
        """

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        data = torch.as_tensor(data, dtype=torch.float32)
        if data.dim() == 1:
            self.data = data.unsqueeze(1).clone().to(self.device)
        elif data.dim() == 2:
            self.data = data.clone().to(self.device)
        elif data.dim() > 2:
            raise ValueError("data must be 1D or 2D tensor")


        self.dt = dt
        self.time = torch.arange(self.data.shape[0], device=self.device) * dt

        self.num_samples, self.num_channels = self.data.shape
        self.t0 = None   # shape (channel,)
        self.t1 = None   # shape (channel,)
        self.params = None  # dict with A, k, C
        self.data_fit = None # shape (time, channel)
        self.kernel = None  # shape (time,)
        self.irf = None  # shape (time, channel)


    def find_t0_t1(self, window_length: int = 11, polyorder: int = 3, persistence: int = 5, threshold: float = 0.05):
        """
        Estimate per-channel start (t0) and end (t1) indices of decay using Savitzky-Golay derivative filtering.

        Args:
            window_length (int): Length of the Savitzky-Golay filter window (must be odd).
            polyorder (int): Polynomial order for Savitzky-Golay filter.
            persistence (int): Number of consecutive positive derivative samples for t1 detection.
            threshold (float): Minimum amplitude threshold for t1 detection (fraction of channel range).

        Side effects:
            Sets self.t0 and self.t1 as torch.Tensors of shape (num_channels,).
        """

        if window_length % 2 == 0:
            raise ValueError("window_length must be odd.")

        t0s, t1s = [], []
        y_range = self.data.max(dim=0).values - self.data.min(dim=0).values

        for c in range(self.num_channels):
            # Move data to CPU for scipy (savgol_filter), then back to device
            y = self.data[:, c].detach().cpu().numpy()
            dy = savgol_filter(y, window_length=window_length, polyorder=polyorder, deriv=1, delta=self.dt)
            dy = torch.tensor(dy, dtype=self.data.dtype, device=self.device)

            # t0: global minimum of derivative
            t0 = int(torch.argmin(dy))

            # t1: first point after t0 with persistent positive derivative
            t1 = len(dy) - 1  # fallback to end
            for i in range(t0 + 1, len(dy) - persistence):
                avg_diff = dy[i:i + persistence].mean()
                amplitude = torch.clamp(self.data[i + persistence, c] - self.data[..., c].min(), 0).item()
                if avg_diff > 0 and amplitude > threshold * y_range[c].item():
                    t1 = i
                    break

            t0s.append(t0)
            t1s.append(t1)

        self.t0 = torch.tensor(t0s, device=self.device)
        self.t1 = torch.tensor(t1s, device=self.device)


    def fit_exponential(self, offset = 0, lr=1e-2, steps=1000):
        """
        Fit per-channel truncated exponential curves to the data between t0 and t1 using gradient-based optimization.

        Args:
            offset (int): Shift applied to t0 when selecting data for fitting.
            lr (float): Learning rate for Adam optimizer (default: 1e-2).
            steps (int): Number of optimization steps (default: 1000).

        Side effects:
            Sets self.params to a dict with keys: "A", "C", "k".

        Raises:
            RuntimeError: If t0 or t1 have not been computed.
        """

        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

        device, dtype = self.device, self.data.dtype
        A = (self.data.max(dim=0).values - self.data.min(dim=0).values).clone().detach().to(device).requires_grad_(True)
        Cparam = self.data.min(dim=0).values.clone().detach().to(device).requires_grad_(True)

        cc = self.num_channels // 2  # central channel index
        tau = estimate_lifetime(self.time, self.data[..., cc], self.t0[cc], self.t1[cc])
        k = (1.0 / tau).clone().detach().to(device).requires_grad_(True)

        opt = torch.optim.Adam([A, Cparam, k], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            loss = 0.0
            for c in range(self.num_channels):
                y = self.data[self.t0[c]+offset:self.t1[c]+1, c]
                x = torch.arange(len(y), device=device, dtype=dtype) * self.dt
                y_pred = A[c] * torch.exp(-k * x) + Cparam[c]
                loss = loss + torch.mean((y - y_pred) ** 2)
            loss = loss / self.num_channels
            loss.backward()
            opt.step()

        self.params = {"A": A.detach(), "C": Cparam.detach(), "k": k.detach().item()}


    def generate_data_fit(self):
        """
        Generate fitted truncated exponential curves for all channels using parameters in self.params.

        Side effects:
            Sets self.data_fit to tensor of shape (num_samples, num_channels).

        Raises:
            RuntimeError: If self.params is None.
        """

        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        exp_curves = torch.zeros_like(self.data, device=self.device)

        for c in range(self.num_channels):
            params = {
                "A": self.params["A"][c],
                "C": self.params["C"][c],
                "k": self.params["k"],
                "t0": int(self.t0[c])*self.dt,
            }
            exp_curves[:, c] = generate_truncated_exponential(self.time, params).to(self.device)

        self.data_fit = exp_curves


    def generate_kernel(self):
        """
        Build a normalized, positive kernel from a truncated exponential with unit amplitude and zero offset.

        Side effects:
            Sets self.kernel to tensor of shape (num_samples,).

        Raises:
            RuntimeError: If self.params is None.
        """
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        params = {
            "A": 1,
            "C": 0,
            "k": self.params["k"],
            "t0": 0,
        }

        exp_curve = generate_truncated_exponential(self.time, params).to(self.device)
        exp_curve = torch.clamp(exp_curve, min=0)  # enforce positivity
        exp_curve /= exp_curve.sum()  # normalize kernel
        self.kernel = exp_curve


    def richardson_lucy_deconvolution(self, iterations=30, eps=1e-4, regularization = 3):
        """
        Perform Richardson-Lucy deconvolution channel-wise using FFT-based convolutions and a precomputed kernel.

        Args:
            iterations (int): Number of RL iterations (default: 30).
            eps (float): Small value to avoid division by zero (default: 1e-4).
            regularization (int): Median filter window size for regularization (default: 3).

        Side effects:
            Sets self.irf to the deconvolved estimate (shape: [num_samples, num_channels]).

        Raises:
            RuntimeError: If self.kernel is None.
        """

        if self.kernel is None:
            raise RuntimeError("Run generate_kernel() first or provide a convolution kernel manually.")

        # initialize output tensor
        x_est = torch.ones_like(self.data, device=self.device) # shape (time, channel)

        # load deconvolution kernel
        kernel = self.kernel.clone().to(self.device)  # shape (time,)
        kernel_t = self.kernel.clone().flip(0).to(self.device)  # time-reversed kernel, shape (time,)

        kernel = fftn(kernel, dim=0) # FT of kernel, shape (time,)
        kernel_t = fftn(kernel_t, dim=0)  # FT of time-reversed kernel, shape (time,)

        # Subtract offset
        y = self.data.detach().clone().to(dtype=torch.float32, device=self.device) - self.params['C'].unsqueeze(0).to(self.device)
        y = torch.clamp(y, min=0)

        # RL deconvolution
        for _ in range(iterations):
            conv = partial_convolution(x_est, kernel, dim1='xc', dim2='x', axis='x', fourier=(0,1))
            conv = torch.clamp(conv, min=eps)  # avoid div by 0
            relative_blur = y / conv
            correction = partial_convolution(relative_blur, kernel_t, dim1='xc', dim2='x', axis='x', fourier=(0,1))
            x_est = x_est * correction
            x_est = torch.clamp(x_est, min=0)  # enforce positivity
            if regularization > 1:
                x_est = median_filter(x_est, window_size=regularization, dims=[0], mode='replicate')  # temporal median filter

        self.irf = x_est


    def run(self, lr=1e-2, steps=1000, rl_iterations=30, regularization=3,
            window_length=11, polyorder=3, persistence=5, threshold=0.05):
        """
        Execute the full IRF estimation pipeline:
            1. Find t0, t1 per channel using Savitzky-Golay filtering.
            2. Fit truncated exponential curves.
            3. Generate fitted exponential curves.
            4. Generate deconvolution kernel.
            5. Perform Richardson-Lucy deconvolution.

        Args:
            lr (float): Learning rate for exponential fit.
            steps (int): Number of optimization steps for exponential fit.
            rl_iterations (int): Number of RL deconvolution iterations.
            regularization (int): Median filter size for RL deconvolution.
            window_length (int): Length of Savitzky-Golay filter window (odd).
            polyorder (int): Polynomial order for SG filter.
            persistence (int): Number of consecutive positive derivative samples for t1.
            threshold (float): Minimum amplitude threshold for t1.

        Returns:
            torch.Tensor: Estimated IRFs (shape: [num_samples, num_channels]).
        """

        self.find_t0_t1(window_length=window_length, polyorder=polyorder, persistence=persistence, threshold=threshold)
        self.fit_exponential(lr=lr, steps=steps)
        self.generate_data_fit()
        self.generate_kernel()
        self.richardson_lucy_deconvolution(iterations=rl_iterations, regularization=regularization)

        return self.irf


    def plot_raw_and_fit(self):
        """
        Plot raw data points and fitted exponential curves for each channel.
        Also draws vertical dashed lines at t0 and t1 for each channel.

        Side effects:
            Uses plot_dataset to create matplotlib figures and axes.
            Adds a legend labeling 'Raw', 'Fit', and 'Fitting interval'.

        Raises:
            RuntimeError: If self.data_fit is None.
        """

        if self.data_fit is None:
            raise RuntimeError("Run generate_data_fit() first.")

        # Move tensors to CPU for plotting
        time = self.time.cpu().numpy()
        raw = self.data.cpu().numpy()
        fit = self.data_fit.cpu().numpy()

        # First, plot raw data as scatter (points)
        fig, ax = plot_dataset(time, raw, color="k", linestyle="none", marker='.')

        # Overlay full fit curve
        fig, ax = plot_dataset(time, fit, color="r", linestyle="-", fig=fig, ax=ax)

        # Draw vertical dashed lines at t0 and t1
        for c in range(self.num_channels):
            t0_val = self.time[int(self.t0[c])].item()
            t1_val = self.time[int(self.t1[c])].item()
            ax[c].axvline(t0_val, color='grey', linestyle='--', alpha=0.5)
            ax[c].axvline(t1_val, color='grey', linestyle='--', alpha=0.5)


        fig.legend(["Raw", "Fit", "Fitting interval"], loc="upper right", bbox_to_anchor=(0.98, 0.95))


    def plot_forward_model(self):
        """
        Convolve estimated IRFs with the fitted exponential kernel and plot the forward model against the measured data for each channel.

        Side effects:
            Uses partial_convolution and plot_dataset to produce a matplotlib figure.
            Adds a legend labeling 'Measured' and 'IRF ⊗ Exp'.

        Raises:
            RuntimeError: If self.irf is None.
        """

        if self.irf is None:
            raise RuntimeError("Run richardson_lucy_deconvolution() first.")

        time = self.time.cpu().numpy()
        raw = self.data.cpu().numpy()

        forward = partial_convolution(self.irf, self.kernel, dim1='xc', dim2='x', axis='x', fourier=(0,0))
        forward += self.params['C'].unsqueeze(0).to(forward.device)  # Add offset on same device
        forward = forward.cpu().numpy()  # Move to CPU only for plotting

        # First, plot raw data as scatter
        fig, ax = plot_dataset(time, raw, color="k", linestyle="none", marker='.')
        # Then add forward model as line
        fig, ax = plot_dataset(time, forward, color="g", linestyle="-", fig=fig, ax=ax)

        fig.legend(["Measured", "IRF ⊗ Exp"], loc="upper right", bbox_to_anchor=(0.95, 0.95))
