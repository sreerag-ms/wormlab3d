from typing import Tuple, Optional, Sequence

import torch
import torch.nn as nn


class ConvergenceDetector(nn.Module):
    tau_fast: torch.Tensor
    tau_slow: torch.Tensor
    val: torch.Tensor
    mu_fast: torch.Tensor
    mu_slow: torch.Tensor
    T_upper: torch.Tensor
    T_lower: torch.Tensor
    convergence_count: torch.Tensor
    converged: torch.Tensor
    initialized: torch.Tensor
    late_start_mask: torch.Tensor
    threshold: float
    patience: int
    eps: float

    continuous_state_vars = [
        'tau_fast', 'tau_slow', 'val', 'mu_fast', 'mu_slow',
        'T_upper', 'T_lower', 'convergence_count'
    ]
    binary_state_vars = ['converged', 'initialized', 'late_start_mask']

    def __init__(
        self,
        shape: Tuple[int],
        tau_fast: int = 10,
        tau_slow: int = 100,
        threshold: float = 0.1,
        patience: int = 100,
        eps: float = 0.0,
        late_start_indices: Optional[Sequence[int]] = None,  # e.g. [-1] for head/tail at the end
    ):
        super().__init__()
        for x_name in self.continuous_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.float32))
        for x_name in self.binary_state_vars:
            # store as bool buffers
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.bool))

        self.tau_fast.fill_(tau_fast)
        self.tau_slow.fill_(tau_slow)
        self.threshold = threshold
        self.patience = patience
        self.eps = eps

        # mark which channels are late-starting
        if late_start_indices is None:
            self.late_start_mask.zero_()
        else:
            mask = torch.zeros_like(self.late_start_mask)
            for i in late_start_indices:
                mask[i] = True
            self.late_start_mask = mask

        # For non-late channels, treat as initialized from the start
        self.initialized = ~self.late_start_mask

    @torch.jit.export
    def set_late_start_indices(self, idxs: torch.Tensor):
        """
        Optionally update late-start mask at runtime.
        idxs: 1D long tensor of indices (same length as shape).
        """
        self.late_start_mask.zero_()
        self.late_start_mask[idxs] = True
        self.initialized = ~self.late_start_mask

    def forward(self, val: torch.Tensor, first_val: bool = False):
        # keep device/dtype consistent
        if val.dtype != self.mu_fast.dtype:
            val = val.to(self.mu_fast.dtype)
        if val.device != self.mu_fast.device:
            val = val.to(self.mu_fast.device)

        self.val = val.clone()

        # First-call seeding for every channel
        if first_val:
            self.mu_fast = val.clone()
            self.mu_slow = val.clone()

        # Init only for late-start channels
        needs_init = self.late_start_mask & (~self.initialized) & (val.abs() > self.eps)
        if needs_init.any():
            self.mu_fast = torch.where(needs_init, val, self.mu_fast)
            self.mu_slow = torch.where(needs_init, val, self.mu_slow)
            self.convergence_count = torch.where(
                needs_init, torch.zeros_like(self.convergence_count), self.convergence_count
            )
            self.converged = torch.where(needs_init, torch.zeros_like(self.converged), self.converged)
            self.initialized = self.initialized | needs_init

        mask = self.initialized
        self._update_estimates(mask)
        self._update_state(mask)

    @torch.jit.export
    def reset_counters(self):
        self.convergence_count.zero_()
        self.converged.zero_()
        self.initialized = ~self.late_start_mask

    def _update_state(self, mask: torch.Tensor):
        in_bounds_core = (self.mu_slow > self.T_lower) & (self.mu_slow < self.T_upper)

        in_bounds_nonlate_bonus = (~self.late_start_mask) & (self.mu_fast == 0)

        in_bounds = mask & (in_bounds_core | in_bounds_nonlate_bonus)

        self.convergence_count = torch.where(
            in_bounds,
            self.convergence_count + 1,
            torch.where(mask, torch.zeros_like(self.convergence_count), self.convergence_count)
        )

        base_converged = self.convergence_count >= self.patience
        # Uninitialized channels are treated as converged
        self.converged = torch.where(mask, base_converged, torch.ones_like(base_converged))

    def _update_estimates(self, mask: torch.Tensor):
        alpha_fast = 1.0 / self.tau_fast
        diff_fast = self.val - self.mu_fast
        mu_fast_new = self.mu_fast + alpha_fast * diff_fast
        self.mu_fast = torch.where(mask, mu_fast_new, self.mu_fast)

        alpha_slow = 1.0 / self.tau_slow
        diff_slow = self.val - self.mu_slow
        mu_slow_new = self.mu_slow + alpha_slow * diff_slow
        self.mu_slow = torch.where(mask, mu_slow_new, self.mu_slow)

        band = self.threshold * torch.abs(self.mu_fast)
        T_upper_new = self.mu_fast + band
        T_lower_new = self.mu_fast - band
        self.T_upper = torch.where(mask, T_upper_new, self.T_upper)
        self.T_lower = torch.where(mask, T_lower_new, self.T_lower)
