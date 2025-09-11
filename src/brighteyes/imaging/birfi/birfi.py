import torch
import torch.nn.functional as F

from. utils import median_filter

class Birfi:

    def __init__(self, data):
        self.data = torch.squeeze(data)
        self.t0 = None
        self.t1 = None

    def find_t0_t1(self, median_window: int = 5):
        # Derivative
        derivative = torch.diff(self.data)
        derivative = median_filter(derivative, median_window)

        # Global minimum index
        t0 = torch.argmin(derivative).item()

        # Search for first non-negative derivative after t0
        post = derivative[t0+1:]
        nonneg = torch.where(post >= 0)[0]

        if len(nonneg) > 0:
            t1 = (t0 + 1 + nonneg[0].item())
        else:
            t1 = len(self.data) - 1  # last index

        # Store
        self.t0, self.t1 = t0, t1
        return t0, t1

    def fit_exponential(self, lr=1e-2, steps=1000):
        if self.t0 is None or self.t1 is None:
            raise RuntimeError("Run find_t0_t1 first.")

        y = self.data[self.t0:self.t1 + 1]
        x = torch.arange(len(y), device=y.device, dtype=y.dtype)

        # Parameters to optimize: A, k, C
        A = torch.tensor(y.max() - y.min(), device=y.device, dtype=y.dtype, requires_grad=True)
        k = torch.tensor(0.1, device=y.device, dtype=y.dtype, requires_grad=True)
        C = torch.tensor(y.min(), device=y.device, dtype=y.dtype, requires_grad=True)

        opt = torch.optim.Adam([A, k, C], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            y_pred = A * torch.exp(-k * x) + C
            loss = torch.mean((y - y_pred) ** 2)
            loss.backward()
            opt.step()

        self.params = {
            "A": A.detach().item(),
            "k": k.detach().item(),
            "C": C.detach().item()
        }
        return self.params


    def generate_truncated_exponential(self):
        if self.params is None:
            raise RuntimeError("Run fit_exponential first.")

        A, k, C = self.params["A"], self.params["k"], self.params["C"]

        # Peak index of real data
        peak_idx = torch.argmax(self.data).item()

        # Time axis for full signal
        x = torch.arange(len(self.data), device=self.data.device, dtype=self.data.dtype)

        # Shift exponential so that its max is at peak_idx
        shifted_x = x - peak_idx
        exp_curve = A * torch.exp(-k * torch.clamp(shifted_x, min=0)) + C

        return exp_curve