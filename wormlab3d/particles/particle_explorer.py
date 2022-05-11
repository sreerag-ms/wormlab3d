import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Dict, Tuple, Optional, Any

import torch
from progress.bar import Bar
from pyro.distributions import Stable
from torch import nn
from torch.distributions import Distribution, LogNormal, Cauchy, Normal

from wormlab3d import logger
from wormlab3d.midlines3d.mf_methods import normalise

PARTICLE_PARAMETER_KEYS = ['speeds', 'planar_angles', 'nonplanar_angles']


def orthogonalise(source: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return source - (torch.einsum('bs,bs->b', source, ref) / ref.norm(dim=-1, keepdim=False, p=2))[:, None] * ref


def init_dist(params: Dict[str, Any]) -> Distribution:
    """
    Initialise a Torch Distribution instance.
    """
    if params['type'] == 'norm':
        mu, sigma = params['params']
        return Normal(loc=mu, scale=sigma)
    elif params['type'] == 'lognorm':
        mu, sigma = params['params']
        return LogNormal(loc=mu, scale=sigma)
    elif params['type'] == 'cauchy':
        mu, sigma = params['params']
        return Cauchy(loc=mu, scale=sigma)
    elif params['type'] == 'levy_stable':
        alpha, beta, loc, scale = params['params']
        return Stable(stability=alpha, skew=beta, loc=loc, scale=scale)
    else:
        raise RuntimeError(f'Unsupported distribution "{params["type"]}"!')


class ParticleExplorer(nn.Module, ABC):
    state: torch.Tensor
    x: torch.Tensor
    e0: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor
    speeds: torch.Tensor
    planar_angles: torch.Tensor
    nonplanar_angles: torch.Tensor

    def __init__(
            self,
            batch_size: int = 20,
            x0: torch.Tensor = None,
            state0: torch.Tensor = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self._init_state(state0)
        self._init_particle(x0)

    @abstractmethod
    def _init_state(
            self,
            state0: Optional[torch.Tensor] = None
    ):
        """
        Initialise the state and state parameters.
        """
        pass

    def _init_particle(self, x0: Optional[torch.Tensor] = None):
        """
        Initialise the particle position and orientation.
        """

        # Starting position
        if x0 is None:
            x0 = torch.zeros((self.batch_size, 3), dtype=torch.float32)
        else:
            assert x0.shape == (self.batch_size, 3)
            x0 = x0.to(torch.float32)
        self.register_buffer('x', x0)

        # Heading
        e0 = normalise(torch.rand(self.batch_size, 3))
        self.register_buffer('e0', e0)

        # Orientation
        e1 = normalise(orthogonalise(torch.rand(self.batch_size, 3), self.e0))
        self.register_buffer('e1', e1)
        e2 = normalise(torch.cross(e0, e1))
        self.register_buffer('e2', e2)

    def forward(
            self,
            T: int,
            dt: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simulate a batch of particle explorers.
        """
        n_steps = int(T / dt)
        ts = torch.arange(n_steps) * dt

        start_time = time.time()

        X = torch.zeros((self.batch_size, n_steps, 3))
        states = torch.zeros((self.batch_size, n_steps))
        speeds = torch.zeros((self.batch_size, n_steps,))
        planar_angles = torch.zeros((self.batch_size, n_steps,))
        nonplanar_angles = torch.zeros((self.batch_size, n_steps,))

        logger.info('Simulating particle exploration.')
        bar = Bar('Simulating', max=n_steps)
        bar.check_tty = False

        for i in range(n_steps):
            self.step(dt)
            X[:, i] = self.x
            states[:, i] = self.state
            speeds[:, i] = self.speeds
            planar_angles[:, i] = self.planar_angles
            nonplanar_angles[:, i] = self.nonplanar_angles
            bar.next()
        bar.finish()
        sim_time = time.time() - start_time
        logger.info(f'Time: {timedelta(seconds=sim_time)}')

        return ts, X, states, speeds, planar_angles, nonplanar_angles

    def step(self, dt: float = 1.):
        """
        Take a single time step forward.
        """
        self._update_state(dt)
        self._update_particle(dt)

    @abstractmethod
    def _update_state(self, dt: float = 1.):
        """
        Update the state.
        """
        pass

    def _update_particle(self, dt: float = 1.):
        """
        Update the particle position and orientation.
        """

        # Sample new parameters
        self._sample_parameters(dt)

        # Update the frame
        self._rotate_frames(self.planar_angles, 'planar')
        self._rotate_frames(self.nonplanar_angles, 'nonplanar')

        # Take a step
        self.x += self.speeds[:, None] * self.e0

    @abstractmethod
    def _sample_parameters(self, dt: float = 1.):
        """
        Sample parameters from the distributions associated with the current states.
        """
        pass

    def _rotate_frames(self, angles: torch.Tensor, which: str):
        """
        Rotate the frames in a planar or non-planar direction.
        """
        if which == 'planar':
            u = self.e2
        else:
            u = self.e1

        # Convert rotation axis and angle to 3x3 rotation matrix
        # (See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
        I = torch.eye(3)[None, ...].repeat(self.batch_size, 1, 1)
        cosA = torch.cos(angles)[:, None, None]
        sinA = torch.sin(angles)[:, None, None]
        outer = torch.einsum('bi,bj->bij', u, u)
        cross = torch.cross(u[..., None].repeat(1, 1, 3), I)

        R = cosA * I \
            + sinA * cross \
            + (1 - cosA) * outer

        # Rotate frame vectors
        self.e0 = normalise(torch.einsum('bij,bj->bi', R, self.e0))
        if which == 'planar':
            self.e1 = normalise(torch.einsum('bij,bj->bi', R, self.e1))

        # Recalculate frame
        self.e1 = normalise(orthogonalise(self.e1, self.e0))
        self.e2 = normalise(torch.cross(self.e0, self.e1))
