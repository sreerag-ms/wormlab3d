import time
from datetime import timedelta
from typing import Dict, Optional, Any, Tuple

import torch
from progress.bar import Bar
from torch import nn

from wormlab3d import logger
from wormlab3d.midlines3d.mf_methods import normalise
from wormlab3d.particles.particle_explorer import orthogonalise, init_dist

PARTICLE_PARAMETER_KEYS = ['run_durations', 'planar_angles', 'nonplanar_angles']


class TwoStateExplorer(nn.Module):

    def __init__(
            self,
            batch_size: int = 20,
            speed: float = 0.005,
            run_duration_dist_params: Optional[Dict[str, Any]] = None,
            planar_angle_dist_params: Optional[Dict[str, Any]] = None,
            nonplanar_angle_dist_params: Optional[Dict[str, Any]] = None,
            x0: torch.Tensor = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.speed = speed
        self._init_particle(x0)
        self._init_distributions(run_duration_dist_params, planar_angle_dist_params, nonplanar_angle_dist_params)

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

        # Tumble state
        for pk in PARTICLE_PARAMETER_KEYS:
            self.register_buffer(pk, torch.zeros(self.batch_size))

    def _init_distributions(
            self,
            run_duration_dist_params: Optional[Dict[str, Any]] = None,
            planar_angle_dist_params: Optional[Dict[str, Any]] = None,
            nonplanar_angle_dist_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise the movement and angles distributions from which values are sampled.
        """
        if run_duration_dist_params is None:
            run_duration_dist_params = {
                'type': 'levy_stable',
                'mu': 0,
                'sigma': 0.1,
            }
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

        self.run_duration_dist_params = run_duration_dist_params
        self.planar_angles_dist_params = planar_angle_dist_params
        self.nonplanar_angles_dist_params = nonplanar_angle_dist_params
        self.run_duration_dist = init_dist(run_duration_dist_params)
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

        logger.info('Generating run durations and tumble angles.')
        e0s = []
        run_durations = []
        sim_times = torch.zeros(self.batch_size, dtype=torch.float32)
        planar_angles = []
        nonplanar_angles = []
        while min(sim_times) < T:
            self._sample_parameters()
            self._rotate_frames(self.planar_angles, 'planar')
            self._rotate_frames(self.nonplanar_angles, 'nonplanar')
            planar_angles.append(self.planar_angles.clone())
            nonplanar_angles.append(self.nonplanar_angles.clone())
            run_durations.append(self.run_durations.clone())
            e0s.append(self.e0)
            sim_times += self.run_durations

        e0s = torch.stack(e0s, dim=1)
        planar_angles = torch.stack(planar_angles, dim=1)
        nonplanar_angles = torch.stack(nonplanar_angles, dim=1)
        run_durations = torch.stack(run_durations, dim=1)
        tumble_ts = run_durations.cumsum(dim=1)

        X = torch.zeros((self.batch_size, n_steps, 3))
        logger.info('Simulating particle exploration.')
        bar = Bar('Simulating', max=self.batch_size)
        bar.check_tty = False
        for i in range(self.batch_size):
            step = 0
            j = 0
            x = self.x[i]
            X[i, 0] = self.x[i]
            while step < n_steps:
                if j > run_durations.shape[1]:
                    run_t = torch.tensor((n_steps - step) * dt)
                else:
                    run_t = min(torch.tensor((n_steps - step) * dt), run_durations[i, j])

                run_steps = int((run_t / dt).round())
                e0 = e0s[i, j]

                run_start = x[None, :]
                run_end = (x + self.speed * e0 * run_t)[None, :]
                y = torch.linspace(0, 1, run_steps)[:, None]
                X[i, step:step + run_steps] = (1 - y) * run_start + y * run_end

                j += 1
                step += run_steps
                x = run_end

            bar.next()
        bar.finish()

        sim_time = time.time() - start_time
        logger.info(f'Time: {timedelta(seconds=sim_time)}')
        return ts, tumble_ts, X, run_durations, planar_angles, nonplanar_angles

    def _sample_parameters(self):
        """
        Sample parameters from the distributions
        """
        for pk in PARTICLE_PARAMETER_KEYS:
            if pk == 'planar_angles':
                dist = self.planar_angles_dist
            elif pk == 'nonplanar_angles':
                dist = self.nonplanar_angles_dist
            else:
                dist = self.run_duration_dist
            val = dist.sample((self.batch_size,))

            if pk in ['planar_angles', 'nonplanar_angles']:
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
