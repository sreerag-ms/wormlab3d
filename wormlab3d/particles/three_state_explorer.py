import time
from datetime import timedelta
from typing import Dict, Optional, Any
from typing import Tuple

import torch
from progress.bar import Bar
from scipy.signal import find_peaks
from torch import nn

from wormlab3d import logger
from wormlab3d.midlines3d.mf_methods import normalise
from wormlab3d.particles.particle_explorer import orthogonalise, init_dist

PARTICLE_PARAMETER_KEYS = ['planar_angles', 'nonplanar_angles']


class ThreeStateExplorer(nn.Module):

    def __init__(
            self,
            batch_size: int = 20,
            rate_01: float = 0.001,
            rate_10: float = 0.001,
            rate_02: float = 0.001,
            rate_20: float = 0.001,
            speed_0: float = 0.001,
            speed_1: float = 0.005,
            planar_angle_dist_params: Optional[Dict[str, Any]] = None,
            nonplanar_angle_dist_params: Optional[Dict[str, Any]] = None,
            x0: torch.Tensor = None,
            state0: torch.Tensor = None,
    ):
        super().__init__()
        self.batch_size = batch_size

        # Transition rates
        self.rate_01 = rate_01
        self.rate_10 = rate_10
        self.rate_02 = rate_02
        self.rate_20 = rate_20

        # Speeds
        self.speed_0 = speed_0
        self.speed_1 = speed_1

        self._init_particle(x0)
        self._init_state(state0)
        self._init_distributions(planar_angle_dist_params, nonplanar_angle_dist_params)

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

    def _init_state(
            self,
            state0: Optional[torch.Tensor] = None
    ):
        """
        Initialise the state and state parameters.
        """
        if state0 is None:
            state0 = torch.zeros((self.batch_size,), dtype=torch.uint8)
        else:
            assert state0.shape == (self.batch_size,)
            state0 = state0.to(torch.uint8)
        self.register_buffer('state', state0)

        for pk in PARTICLE_PARAMETER_KEYS:
            self.register_buffer(pk, torch.zeros(self.batch_size))

    def _init_distributions(
            self,
            planar_angle_dist_params: Optional[Dict[str, Any]] = None,
            nonplanar_angle_dist_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise the angles distributions from which values are sampled.
        """
        if planar_angle_dist_params is None:
            planar_angle_dist_params = {
                'type': 'cauchy',
                'mu': 0,
                'sigma': 0.1,
            }
        if nonplanar_angle_dist_params is None:
            nonplanar_angle_dist_params = {
                'type': 'cauchy',
                'mu': 0,
                'sigma': 0.5,
            }

        self.planar_angles_dist_params = planar_angle_dist_params
        self.nonplanar_angles_dist_params = nonplanar_angle_dist_params
        self.planar_angles_dist = init_dist(planar_angle_dist_params)
        self.nonplanar_angles_dist = init_dist(nonplanar_angle_dist_params)

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

        # Generate the state transitions
        logger.info('Generating state transitions.')
        states = torch.zeros((self.batch_size, n_steps))
        bar = Bar('Simulating', max=n_steps)
        bar.check_tty = False
        for i in range(n_steps):
            self._update_state()
            states[:, i] = self.state
            bar.next()
        bar.finish()

        # Calculate the sequences
        sequences = []
        for i in range(self.batch_size):
            sequences.append(torch.unique_consecutive(states[i]))

        # Calculate the run durations
        run_starts = {s: [[] for _ in range(self.batch_size)] for s in [0, 1]}
        durations = {s: [[] for _ in range(self.batch_size)] for s in [0, 1]}
        for s in [0, 1]:
            cond_state = torch.cat([
                torch.zeros(self.batch_size, 1),
                states == s,
                torch.zeros(self.batch_size, 1)
            ], dim=1)
            for i in range(self.batch_size):
                centre_idxs, section_props = find_peaks(cond_state[i], width=1)
                for j in range(len(centre_idxs)):
                    start = section_props['left_bases'][j]
                    w = section_props['widths'][j]
                    run_starts[s][i].append(int(start))
                    durations[s][i].append(int(w))

        # Generate the tumbles
        logger.info('Generating tumbles.')
        tumble_idxs = (states == 2).nonzero()
        tumble_ts = [tumble_idxs[tumble_idxs[:, 0] == i][:, 1] * dt for i in range(self.batch_size)]
        max_tumbles = max((states == 2).sum(dim=1))
        e0s = [self.e0, ]
        planar_angles = []
        nonplanar_angles = []
        bar = Bar('Simulating', max=max_tumbles)
        bar.check_tty = False
        for _ in range(max_tumbles):
            self._sample_parameters()
            self._rotate_frames(self.planar_angles, 'planar')
            self._rotate_frames(self.nonplanar_angles, 'nonplanar')
            planar_angles.append(self.planar_angles.clone())
            nonplanar_angles.append(self.nonplanar_angles.clone())
            e0s.append(self.e0)
            bar.next()
        bar.finish()
        e0s = torch.stack(e0s, dim=1)
        planar_angles = torch.stack(planar_angles, dim=1)
        nonplanar_angles = torch.stack(nonplanar_angles, dim=1)

        # Simulate the particle exploration
        e0_exp = torch.zeros((self.batch_size, n_steps, 3))
        logger.info('Simulating particle exploration.')
        bar = Bar('Simulating', max=self.batch_size)
        bar.check_tty = False
        for i in range(self.batch_size):
            step = 1
            j = [0, 0]
            k = 0
            e0 = e0s[i, 0]
            e0_exp[i, 0] = self.x[i]

            while step < n_steps:
                s = int(states[i, step])

                # Tumble - change heading but stay in the same place
                if s == 2:
                    e0 = e0s[i, k]
                    k += 1
                    run_steps = 1

                # Run
                else:
                    run_steps = durations[s][i][j[s]]
                    speed = [self.speed_0, self.speed_1][s]
                    e0_exp[i, step:step + run_steps] = e0[None, :] * speed
                    j[s] += 1

                step += run_steps

            X = torch.cumsum(e0_exp, dim=1)

            bar.next()
        bar.finish()

        # Convert durations into seconds
        for s in [0, 1]:
            for i in range(self.batch_size):
                durations[s][i] = torch.tensor(durations[s][i], dtype=torch.float32) * dt

        sim_time = time.time() - start_time
        logger.info(f'Time: {timedelta(seconds=sim_time)}')

        # Calculate intervals between turns
        intervals = []
        for i in range(self.batch_size):
            intervals.append(tumble_ts[i][1:] - tumble_ts[i][:-1])

        return ts, tumble_ts, X, states, durations, planar_angles, nonplanar_angles, intervals

    def _update_state(self):
        """
        Update the state.
        """
        r0 = torch.rand(self.batch_size)
        r1 = torch.rand(self.batch_size)
        self.state = torch.where(

            # --- State 0 ---
            self.state == 0,
            torch.where(
                # Transition from 0->1
                r0 < self.rate_01,
                torch.ones_like(self.state),

                torch.where(
                    # Transition from 0->2
                    r0 > (1 - self.rate_02),
                    torch.ones_like(self.state) * 2,

                    # Stay in state 0
                    torch.zeros_like(self.state)
                )
            ),

            torch.where(
                # --- State 1 ---
                self.state == 1,
                torch.where(
                    # Transition from 1->0
                    r1 < self.rate_10,
                    torch.zeros_like(self.state),

                    # Stay in state 1
                    torch.ones_like(self.state)
                ),

                # --- State 2 ---
                torch.where(
                    # Transition from 2->0
                    r1 < self.rate_20,
                    torch.zeros_like(self.state),

                    # Transition from 2->1
                    torch.ones_like(self.state)

                    # Stay in state 2
                    # torch.ones_like(self.state) * 2
                ),
            ),
        ).to(torch.uint8)

    def _sample_parameters(self):
        """
        Sample parameters from the distributions associated with the current states.
        """
        for pk in PARTICLE_PARAMETER_KEYS:
            if pk == 'planar_angles':
                dist = self.planar_angles_dist
            elif pk == 'nonplanar_angles':
                dist = self.nonplanar_angles_dist
            else:
                continue
            val = dist.sample((self.batch_size,))

            if pk in ['planar_angles', 'nonplanar_angles']:
                val[val.isnan() | val.isinf()] = 0
                val = torch.fmod(val, torch.pi)
            else:
                val = torch.abs(val)

            self.get_buffer(pk)[:] = val

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
