# Origin: VicidominiLab/Regularization-by-Noise, src/varreg/adam_optimizer.py at bb6d0c6.
import torch

import numpy as np
import torch.optim as optim
import math

from scipy.ndimage import gaussian_filter

from torch.nn import functional
from torch.fft import rfftn, irfftn, ifftshift
from torch import real, einsum

from .utils import centered_pad
from .loss import fidelity_loss, regularization_loss

from .preprocessing import binomial_split

from . import initializers as init


def adam_optimize_obj(dset, psf, lambda_value=0.01, alpha_value=1, weight_map=1, num_iter=500, lr=1e-3,
                      careful: bool = True,
                      regularizer: str = 'l2', initialization: str = 'flat', process: str = 'gpu',
                      full_KL: bool = False, rec_to_save: str = 'last', verbose: bool = True, sum_rep: bool = False,
                      binomial: bool = False, fourier: bool = False):
    """
    Ottimizza obj_set minimizzando fidelity_loss + reg_weight * regularization_loss con Adam.

    Parameters:
        dset (Tensor): dataset tensor of shape (Nx, Ny) o (Nx, Ny, K).
        psf (Tensor): Point Spread Function tensor with dimension (Mx, My) o (Mx, My, K).
        lambda_value (float): Weight for the regularization term.
        alpha_value (float): Weight for the fidelity term.
        W (float or tensor): Regularization weight a scalar or tensor of shape Nx x Ny.
        num_iter (int): Number of iterations.
        lr (float): Learning rate of Adam.
        regularizer (str): type of regularization ('l1', 'l2', 'kl').
        initialization (str): Initialization method for the object estimate ('flat' or 'data').
        process (str): Processing mode ('cpu' or 'gpu').
        full_KL (bool): If True, use the full form of KL.
        rec_to_save (str): 'last' to return last iteration, 'all' to return all iterations.
        verbose (bool): If True, print progress.

    Returns:
        obj_set (Tensor): Optimized tensor of shape (Nx, Ny, K).
    """

    # Ensure tensors are on the correct device
    if process == 'gpu' and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    dset = torch.from_numpy(dset).float().to(device)
    psf = torch.from_numpy(psf).float().to(device)

    if sum_rep is True:
        dset = dset.sum(dim=-1, keepdim=True)
        n_rep = 1
    else:
        n_rep = dset.shape[-1]

    if binomial is True:
        img = dset.sum(dim=-1)

    num_axis_to_conv = 2

    init_obj = init.initialize(dset, psf, method=initialization, device=None)

    # Repeat the psf for each repetition
    psf = psf.unsqueeze(-1).repeat(*([1] * psf.ndim), n_rep)

    # Axis to sum for flux conservation
    axis_to_sum = tuple(range(dset.ndim - 1))
    flux = dset.sum(axis_to_sum, keepdim=True)

    sz = dset.shape[:num_axis_to_conv]
    psf_pad = centered_pad(psf, target_shape=sz)  # Centered padding of PSF to match dataset shape
    psf_fft = rfftn(psf_pad, dim=tuple(range(num_axis_to_conv)))  # FFT of the PSF

    # Initialization
    size = list(dset.shape)
    if rec_to_save == 'last':
        size = [1] + size
    elif rec_to_save == 'all':
        size = [num_iter + 1] + size
    obj_all = torch.empty(size).to(device)

    obj_all[0] = init_obj
    obj_set = init_obj
    obj_set.requires_grad = True

    # Dictionary with losses
    losses = {
        'total': [],
        'fidelity': [],
        'regularization': [],
        'learning_rate': []
    }

    optimizer = optim.Adam([obj_set], lr=lr, amsgrad=True)

    # ---- Cosine Warmup ----

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)

    scheduler = WarmupWithPlateau(optimizer, plateau_scheduler, warmup_steps=200, careful=careful)

    # ---- Minimizer ----
    for m in range(num_iter):
        optimizer.zero_grad()

        # Compute loss values
        loss_fidelity = alpha_value * fidelity_loss(dset, obj_set, psf_fft, weight_map=weight_map, full_KL=full_KL)

        loss_reg = lambda_value * regularization_loss(obj_set, weight_map=1, regularizer=regularizer, fourier=fourier,
                                                      otf=psf_fft)
        loss = loss_fidelity + loss_reg

        # Store loss values
        losses['total'].append(loss.item())
        losses['fidelity'].append(loss_fidelity.item())
        losses['regularization'].append(loss_reg.item())
        losses['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Backpropagation + update
        loss.backward()
        optimizer.step()

        scheduler.step(loss)

        # Non-negativity constraint
        with torch.no_grad():
            obj_set.clamp_(min=1e-8)

        # Save solution
        if rec_to_save == 'all':
            obj_all[m + 1] = obj_set.detach()
        elif rec_to_save == 'last':
            obj_all[0] = obj_set.detach()
        else:
            raise ValueError("rec_to_save must be 'last' or 'all'.")

        if verbose and (m % max(1, num_iter // 10) == 0 or m == num_iter - 1):
            print(f"Iter {m + 1}/{num_iter} | Loss: {loss.item():.6f} "
                  f"| Fidelity: {loss_fidelity.item():.9f} | Reg: {loss_reg.item():.9f}"
                  f" | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if binomial is True:
            dset = binomial_split(img, n_splits=n_rep, p=0.5, device=device)

    return obj_all.detach(), losses


class WarmupWithPlateau(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, plateau_scheduler, warmup_steps=100, last_epoch=-1, careful=False):
        self.plateau_scheduler = plateau_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_active = True
        self.prev_loss = None
        self.careful = careful
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]  # store initial LR
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.warmup_active:
            step = max(1, self.last_epoch + 1)
            warmup_factor = 0.5 * (1 - math.cos(math.pi * min(step, self.warmup_steps) / self.warmup_steps))
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # After warmup, return whatever LR is set by plateau scheduler
            return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, loss=None, epoch=None):
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()

            # Stop warmup if loss increases
            if self.prev_loss is not None and loss > self.prev_loss and self.careful is True:
                self.warmup_active = False

            self.prev_loss = loss

            # Step plateau scheduler only after warmup ends
            if not self.warmup_active:
                self.plateau_scheduler.step(loss, epoch)

        # Update optimizer LRs according to warmup or leave as is
        new_lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, new_lrs):
            group['lr'] = lr

        return super().step(epoch)