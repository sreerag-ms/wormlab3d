import torch
import torch.nn as nn

eps = 1e-10


class Detector(nn.Module):
    tau: torch.Tensor
    val: torch.Tensor
    mu: torch.Tensor
    sigma: torch.Tensor
    T_upper: torch.Tensor
    T_lower: torch.Tensor
    detected: torch.Tensor
    detecting: torch.Tensor
    lost: torch.Tensor
    idle: torch.Tensor

    continuous_state_vars = ['tau', 'val', 'mu', 'sigma', 'T_upper', 'T_lower']
    binary_state_vars = ['detected', 'detecting', 'lost', 'idle']

    def __init__(self, tau_init, shape):
        super().__init__()
        self.step = 0
        self.tau_init = tau_init
        for x_name in self.continuous_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.float32))
        for x_name in self.binary_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.bool))

    def configure(self):
        self.idle.fill_(1)
        self.tau.fill_(self.tau_init)

    def forward(self, val):
        self.val = val
        # if self.step == 0:
        #     self.
        self.update_state()
        self.update_estimates()

    def update_state(self):
        is_anomalous = self.detect()
        self.detected = is_anomalous & self.idle
        self.lost = (is_anomalous == 0) & self.detecting
        self.detecting = is_anomalous
        self.idle = self.detecting == 0

    def detect(self):
        return (self.val > self.T_upper) | (self.val < self.T_lower)

    def update_estimates(self):
        alpha = 1 / self.tau
        diff = self.val - self.mu
        incr = alpha * diff
        self.mu += incr
        self.sigma = (1 - alpha) * (self.sigma + diff * incr)
        self.sigma = self.sigma.clamp(min=eps**2)
        # self.T_upper = self.mu + 2 * torch.sqrt(self.sigma)
        # self.T_lower = self.mu - 2 * torch.sqrt(self.sigma)
        self.T_upper = self.mu + torch.sqrt(self.sigma)
        self.T_lower = self.mu - torch.sqrt(self.sigma)



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
    threshold: float
    patience: int

    continuous_state_vars = ['tau_fast', 'tau_slow', 'val', 'mu_fast', 'mu_slow', 'T_upper', 'T_lower', 'convergence_count']
    binary_state_vars = ['converged']

    def __init__(self, shape, tau_fast=10,tau_slow=100, threshold=0.1, patience=100):
        super().__init__()
        for x_name in self.continuous_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.float32))
        for x_name in self.binary_state_vars:
            self.register_buffer(x_name, torch.zeros(shape, dtype=torch.bool))

        self.tau_fast.fill_(tau_fast)
        self.tau_slow.fill_(tau_slow)
        self.threshold = threshold
        self.patience =patience

    def forward(self, val):
        self.val = val
        self.update_state()
        self.update_estimates()

    def update_state(self):
        in_bounds = self.check_bounds()
        self.convergence_count = torch.where(
            in_bounds,
            self.convergence_count + 1,
            torch.zeros_like(self.convergence_count)
        )
        self.converged = self.convergence_count >= self.patience

    def check_bounds(self):
        return (self.mu_slow > self.T_lower) & (self.mu_slow < self.T_upper)

    def update_estimates(self):
        # Update fast estimate
        alpha_fast = 1 / self.tau_fast
        diff_fast = self.val - self.mu_fast
        self.mu_fast += alpha_fast * diff_fast

        self.T_upper = self.mu_fast + self.threshold * torch.abs(self.mu_fast)
        self.T_lower = self.mu_fast - self.threshold * torch.abs(self.mu_fast)

        # Update slow estimate
        alpha_slow = 1 / self.tau_slow
        diff_slow = self.val - self.mu_slow
        self.mu_slow += alpha_slow * diff_slow

