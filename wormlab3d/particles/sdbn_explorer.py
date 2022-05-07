import time
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
from progress.bar import Bar
from torch import nn
from torch.distributions import Distribution, LogNormal, Cauchy, Normal

from wormlab3d import logger
from wormlab3d.midlines3d.mf_methods import normalise

PARTICLE_PARAMETER_KEYS = ['speeds', 'planar_angles', 'nonplanar_angles']


def orthogonalise(source: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return source - (torch.einsum('bs,bs->b', source, ref) / ref.norm(dim=-1, keepdim=False, p=2))[:, None] * ref


class SDBNExplorer(nn.Module):
    state: torch.Tensor
    x: torch.Tensor
    e0: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor
    speeds: torch.Tensor
    planar_angles: torch.Tensor
    nonplanar_angles: torch.Tensor
    dists: Dict[str, Dict[str, Distribution]]
    state_parameters: Dict[str, Dict[str, Dict[str, Union[str, float]]]]

    def __init__(
            self,
            depth: int = 3,
            batch_size: int = 1,
            transition_rates: List[torch.Tensor] = None,
            x0: torch.Tensor = None,
            state0: torch.Tensor = None,
            state_parameters: Dict[str, Dict[str, Dict[str, Union[str, float]]]] = None
    ):
        super().__init__()
        if transition_rates is not None:
            depth = len(transition_rates)
        self.depth = depth
        self.batch_size = batch_size
        self._init_transition_rates(transition_rates)
        self._init_particle(x0)
        self._init_state(state0)
        self._init_distributions(state_parameters)

    def _init_transition_rates(self, transition_rates: Optional[List[Union[torch.Tensor, np.ndarray]]] = None):
        """
        Initialise the transition rates.
        """
        if transition_rates is None:
            transition_rates = []
            for d in range(self.depth):
                transition_rates.append(torch.zeros((2,) * d, dtype=torch.float32))
        else:
            assert len(transition_rates) == self.depth
            for d, tr in enumerate(transition_rates):
                assert tr.shape == (2,) * (d + 1)
                if type(tr) == np.ndarray:
                    transition_rates[d] = torch.from_numpy(tr)
                transition_rates[d] = transition_rates[d].to(torch.float32)

        self.transition_rates: List[torch.Tensor] = transition_rates

    def _init_state(
            self,
            state0: Optional[torch.Tensor] = None
    ):
        """
        Initialise the state and state parameters.
        """
        if state0 is None:
            state0 = torch.zeros((self.batch_size, self.depth), dtype=torch.uint8)
        else:
            assert state0.shape == (self.batch_size, self.depth)
            state0 = state0.to(torch.uint8)
        self.register_buffer('state', state0)

        for pk in PARTICLE_PARAMETER_KEYS:
            self.register_buffer(pk, torch.zeros(self.batch_size))

    def _init_distributions(
            self,
            state_parameters: Optional[Dict[str, Dict[str, Dict[str, Union[str, float]]]]] = None
    ):
        """
        Initialise the distributions from which values are sampled.
        """
        if state_parameters is None:
            state_parameters = {}
        dists = {}
        for decimal_state in range(2**self.depth):
            sk = np.binary_repr(decimal_state, self.depth)
            dists[sk] = {}
            assert sk in state_parameters
            for pk in PARTICLE_PARAMETER_KEYS:
                assert pk in state_parameters[sk]
                params = state_parameters[sk][pk]
                assert 'dist' in params
                assert 'mu' in params
                assert 'sigma' in params
                mu, sigma = params['mu'], params['sigma']
                if params['dist'] == 'norm':
                    dists[sk][pk] = Normal(loc=mu, scale=sigma)
                elif params['dist'] == 'lognorm':
                    dists[sk][pk] = LogNormal(loc=mu, scale=sigma)
                elif params['dist'] == 'cauchy':
                    dists[sk][pk] = Cauchy(loc=mu, scale=sigma)
                else:
                    raise RuntimeError(f'Unsupported distribution "{params["dist"]}"!')

        self.state_parameters = state_parameters
        self.dists = dists

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

        # Previously sampled parameter values
        for k in PARTICLE_PARAMETER_KEYS:
            self.register_buffer(k, torch.zeros(self.batch_size))

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
        X = torch.zeros((self.batch_size, n_steps, 3))
        states = torch.zeros((self.batch_size, n_steps, 3))
        speeds = torch.zeros((self.batch_size, n_steps,))
        planar_angles = torch.zeros((self.batch_size, n_steps,))
        nonplanar_angles = torch.zeros((self.batch_size, n_steps,))

        logger.info('Simulating particle exploration.')
        bar = Bar('Simulating', max=n_steps)
        bar.check_tty = False
        start_time = time.time()

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

    def _update_state(self, dt: float = 1.):
        """
        Update the state.
        """
        for d in range(self.depth):
            # Get the transition rates active for the state at this depth
            tr = torch.zeros((self.batch_size, 2), dtype=torch.float32)
            for i in range(self.batch_size):
                tri = self.transition_rates[d]
                for d2 in range(d):
                    tri = tri[self.state[i, d2].to(torch.long)]
                tr[i] = tri

            # Update states
            rand = torch.rand(self.batch_size)
            self.state[:, d] = torch.where(
                self.state[:, d].to(torch.bool),
                rand > tr[:, 1] * dt,
                rand < tr[:, 0] * dt,
            ).to(torch.uint8)

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

    def _sample_parameters(self, dt: float = 1.):
        """
        Sample parameters from the distributions associated with the current states.
        """
        for pk in PARTICLE_PARAMETER_KEYS:
            for i in range(self.batch_size):
                dist = self.dists[''.join([str(si.item()) for si in self.state[i]])][pk]
                val = dist.sample() * dt
                if pk in ['planar_angles', 'nonplanar_angles']:
                    val = torch.fmod(val, torch.pi)
                self.get_buffer(pk)[i] = val

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
