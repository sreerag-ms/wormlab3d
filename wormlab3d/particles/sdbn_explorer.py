import time
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import torch
from progress.bar import Bar
from torch.distributions import Distribution, LogNormal, Cauchy, Normal

from wormlab3d import logger
from wormlab3d.particles.particle_explorer import ParticleExplorer, PARTICLE_PARAMETER_KEYS


class SDBNExplorer(ParticleExplorer):
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
        super().__init__(batch_size, x0, state0)
        if transition_rates is not None:
            depth = len(transition_rates)
        self.depth = depth
        self._init_transition_rates(transition_rates)
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
