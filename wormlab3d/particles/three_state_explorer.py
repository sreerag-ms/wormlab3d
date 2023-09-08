import gc
import time
from datetime import timedelta
from multiprocessing import Pool
from typing import Dict, Optional, Any, Union, Tuple, List

import numpy as np
import torch
from progress.bar import Bar
from scipy.signal import find_peaks
from torch import nn

from wormlab3d import logger, N_WORKERS
from wormlab3d.midlines3d.mf_methods import normalise
from wormlab3d.particles.particle_explorer import orthogonalise, init_dist

PARTICLE_PARAMETER_KEYS = ['thetas', 'phis']


def calculate_durations(
        cond_state: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    """
    Calculate the durations stayed in state.
    """
    run_starts = []
    durations = []
    centre_idxs, section_props = find_peaks(cond_state, width=1)
    for j in range(len(centre_idxs)):
        start = section_props['left_bases'][j]
        w = section_props['widths'][j]
        run_starts.append(int(start))
        durations.append(int(w))
    return run_starts, durations


def calculate_durations_parallel(
        cond_state: torch.Tensor,
) -> Tuple[List[int], List[int]]:
    """
    Calculate the durations in parallel.
    """
    bs = len(cond_state)
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_durations,
            [cond_state[i] for i in range(bs)]
        )
    run_starts = []
    durations = []
    for i in range(bs):
        run_starts.append(res[i][0])
        durations.append(res[i][1])
    return run_starts, durations


def simulate_particle_trajectory(
        states: torch.Tensor,
        e0s: torch.Tensor,
        durations_0: torch.Tensor,
        durations_1: torch.Tensor,
        pauses: torch.Tensor,
        speed_0: float,
        speed_1: float
) -> torch.Tensor:
    """
    Simulate an individual particle trajectory.
    """
    n_steps = states.shape[0]
    step = 0
    j = [0, 0]
    k = 1
    e0 = e0s[0]
    e0_exp = torch.zeros((n_steps, 3))
    pause_steps = 0

    while step < n_steps:
        s = int(states[step - pause_steps])

        # Tumble - change heading but stay in the same place
        if s == 2:
            e0 = e0s[k]
            pause = pauses[k - 1]
            pause_steps += pause
            run_steps = 1 + pause
            k += 1

        # Run
        else:
            if s == 0:
                run_steps = durations_0[j[s]]
            else:
                run_steps = durations_1[j[s]]
            speed = [speed_0, speed_1][s]
            e0_exp[step:step + run_steps] = e0 * speed
            j[s] += 1

        step += run_steps

    return e0_exp


def simulate_particle_trajectory_wrapper(args):
    return simulate_particle_trajectory(*args)


class ThreeStateExplorer(nn.Module):
    thetas: torch.Tensor
    phis: torch.Tensor

    def __init__(
            self,
            batch_size: int = 20,
            rate_01: float = 0.001,
            rate_10: float = 0.001,
            rate_02: float = 0.001,
            rate_20: float = 0.001,
            speed_0: Union[float, np.ndarray] = 0.001,
            speed_1: Union[float, np.ndarray] = 0.005,
            theta_dist_params: Optional[Dict[str, Any]] = None,
            phi_dist_params: Optional[Dict[str, Any]] = None,
            x0: torch.Tensor = None,
            state0: torch.Tensor = None,
            nonp_pause_type: Optional[str] = None,
            nonp_pause_max: float = 0.,
            quiet: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.quiet = quiet

        # Transition rates
        self.rate_01 = rate_01
        self.rate_10 = rate_10
        self.rate_02 = rate_02
        self.rate_20 = rate_20

        # Speeds
        if type(speed_0) == np.ndarray:
            assert speed_0.shape == (batch_size,)
            speed_0 = torch.from_numpy(speed_0)
        elif type(speed_0) == float:
            speed_0 = torch.ones(batch_size) * speed_0
        else:
            raise TypeError(f'Unrecognised speed_0 type: "{type(speed_0)}"')
        self.speed_0 = speed_0
        if type(speed_1) == np.ndarray:
            assert speed_1.shape == (batch_size,)
            speed_1 = torch.from_numpy(speed_1)
        elif type(speed_1) == float:
            speed_1 = torch.ones(batch_size) * speed_1
        else:
            raise TypeError(f'Unrecognised speed_1 type: "{type(speed_1)}"')
        self.speed_1 = speed_1

        # Should nonplanar turns induce a longer pause than planar turns
        self.nonp_pause_type = nonp_pause_type
        self.nonp_pause_max = nonp_pause_max

        self._init_particle(x0)
        self._init_state(state0)
        self._init_distributions(theta_dist_params, phi_dist_params)

    def _log(self, msg: str, level: str = 'info'):
        if not self.quiet:
            getattr(logger, level)(msg)

    def _init_particle(self, x0: Optional[torch.Tensor] = None):
        """
        Initialise the particle position and orientation.
        """

        # Starting position
        if x0 is None:
            x0 = torch.zeros((self.batch_size, 3))
        else:
            assert x0.shape == (self.batch_size, 3)
            x0 = x0.to(torch.float64)
        self.register_buffer('x', x0)

        # Heading
        e0 = normalise(torch.rand(self.batch_size, 3, dtype=torch.float64))
        self.register_buffer('e0', e0)

        # Orientation
        e1 = normalise(orthogonalise(torch.rand(self.batch_size, 3, dtype=torch.float64), self.e0))
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
            theta_dist_params: Optional[Dict[str, Any]] = None,
            phi_dist_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialise the angles distributions from which values are sampled.
        """
        if theta_dist_params is None:
            theta_dist_params = {
                'type': 'cauchy',
                'mu': 0,
                'sigma': 0.1,
            }
        if phi_dist_params is None:
            phi_dist_params = {
                'type': 'cauchy',
                'mu': 0,
                'sigma': 0.5,
            }

        self.theta_dist_params = theta_dist_params
        self.phi_dist_params = phi_dist_params
        self.theta_dist = init_dist(theta_dist_params)
        self.phi_dist = init_dist(phi_dist_params)

    def forward(
            self,
            T: float,
            dt: float
    ) -> 'TrajectoryCache':
        """
        Simulate a batch of particle explorers.
        """
        n_steps = int(T / dt)
        ts = torch.arange(n_steps) * dt
        nonp_pause_max = self.nonp_pause_max / dt

        start_time = time.time()

        # Generate the state transitions
        self._log('Generating state transitions.')
        states = torch.zeros((self.batch_size, n_steps))
        bar = Bar('Generating', max=n_steps)
        bar.check_tty = False
        for i in range(n_steps):
            self._update_state()
            states[:, i] = self.state
            bar.next()
        bar.finish()

        # Calculate the sequences
        self._log('Calculating sequences.')
        sequences = []
        for i in range(self.batch_size):
            sequences.append(torch.unique_consecutive(states[i]))

        # Generate the tumbles
        self._log('Generating tumbles.')
        gc.collect()  # probably in the wrong place but seems to fix the random crashes
        tumble_idxs = (states == 2).nonzero()
        vertex_idxs = [tumble_idxs[tumble_idxs[:, 0] == i][:, 1] for i in range(self.batch_size)]
        tumble_ts = [vi * dt for vi in vertex_idxs]
        max_tumbles = max((states == 2).sum(dim=1))
        e0s = torch.zeros((self.batch_size, max_tumbles + 1, 3), dtype=torch.float64)
        e0s[:, 0] = self.e0.clone()
        thetas = torch.zeros((self.batch_size, max_tumbles))
        phis = torch.zeros((self.batch_size, max_tumbles))
        bar = Bar('Generating', max=max_tumbles)
        bar.check_tty = False
        for i in range(max_tumbles):
            self._sample_parameters()
            self._rotate_frames(self.thetas, 'planar')
            self._rotate_frames(self.phis, 'nonplanar')
            thetas[:, i] = self.thetas.clone()
            phis[:, i] = self.phis.clone()
            e0s[:, i + 1] = self.e0.clone()
            bar.next()
        bar.finish()

        # Calculate the run durations
        self._log('Calculating run durations.')
        run_starts = {s: [[] for _ in range(self.batch_size)] for s in [0, 1]}
        durations = {s: [[] for _ in range(self.batch_size)] for s in [0, 1]}

        if N_WORKERS > 1:
            for s in [0, 1]:
                cond_state = torch.cat([
                    torch.zeros(self.batch_size, 1),
                    states == s,
                    torch.zeros(self.batch_size, 1)
                ], dim=1)
                run_starts[s], durations[s] = calculate_durations_parallel(cond_state)
        else:
            bar = Bar('Calculating', max=self.batch_size * 2)
            bar.check_tty = False
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
                    bar.next()
            bar.finish()

        # Calculate the pause durations based on how extreme the nonplanar angle is
        self._log('Calculating pause durations.')
        if self.nonp_pause_type is not None:
            if self.nonp_pause_type == 'linear':
                p = 1
            elif self.nonp_pause_type == 'quadratic':
                p = 2
            else:
                raise RuntimeError(f'Unrecognised pause type "{self.nonp_pause_type}".')
            pauses = ((phis.abs() / (torch.pi / 2))**p * nonp_pause_max).round().to(torch.int32)
        else:
            pauses = torch.zeros_like(phis).to(torch.int32)

        # Simulate the particle exploration
        self._log('Simulating particle exploration.')
        if N_WORKERS > 1:
            with Pool(processes=N_WORKERS) as pool:
                res = pool.map(
                    simulate_particle_trajectory_wrapper,
                    [
                        (states[i], e0s[i], durations[0][i], durations[1][i], pauses[i], self.speed_0[i],
                         self.speed_1[i])
                        for i in range(self.batch_size)
                    ]
                )
                e0_exp = torch.zeros((self.batch_size, n_steps, 3))
                for i in range(self.batch_size):
                    e0_exp[i] = res[i]
        else:
            e0_exp = torch.zeros((self.batch_size, n_steps, 3))
            bar = Bar('Simulating', max=self.batch_size)
            bar.check_tty = False
            for i in range(self.batch_size):
                e0_exp[i] = simulate_particle_trajectory(
                    states=states[i],
                    e0s=e0s[i],
                    durations_0=durations[0][i],
                    durations_1=durations[1][i],
                    pauses=pauses[i],
                    speed_0=self.speed_0[i],
                    speed_1=self.speed_1[i],
                )
                bar.next()
            bar.finish()
        X = torch.cumsum(e0_exp, dim=1)

        # Convert durations into seconds
        for s in [0, 1]:
            for i in range(self.batch_size):
                durations[s][i] = torch.tensor(durations[s][i], dtype=torch.float64) * dt
        pauses = pauses.clone().to(torch.float64) * dt

        # Update the tumble times to account for the pauses
        cum_pauses = torch.cumsum(torch.cat([torch.zeros(self.batch_size, 1), pauses], dim=1), dim=1)
        for i in range(self.batch_size):
            tumble_ts[i] += cum_pauses[i, :len(tumble_ts[i])] + pauses[i, :len(tumble_ts[i])] / 2

        # Calculate intervals between turns
        intervals = []
        for i in range(self.batch_size):
            intervals.append(tumble_ts[i][1:] - tumble_ts[i][:-1])

        # Calculate average speeds between turns
        speeds = []
        for i in range(self.batch_size):
            vertices = X[i, vertex_idxs[i]]
            distances = (vertices[1:] - vertices[:-1]).norm(dim=-1)
            speeds.append(distances / intervals[i])

        # Prune the angles to those actually used in each run
        thetas = [thetas[i, :len(tumble_ts[i])] for i in range(self.batch_size)]
        phis = [phis[i, :len(tumble_ts[i])] for i in range(self.batch_size)]

        sim_time = time.time() - start_time
        self._log(f'Time: {timedelta(seconds=sim_time)}')

        return dict(
            ts=ts,
            tumble_ts=tumble_ts,
            X=X,
            states=states,
            durations_0=durations[0],
            durations_1=durations[1],
            thetas=thetas,
            phis=phis,
            intervals=intervals,
            speeds=speeds
        )

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
                ),
            ),
        ).to(torch.uint8)

    def _sample_parameters(self):
        """
        Sample parameters from the distributions associated with the current states.
        """
        for pk in PARTICLE_PARAMETER_KEYS:
            if pk == 'thetas':
                dist = self.theta_dist
            elif pk == 'phis':
                dist = self.phi_dist
            else:
                continue
            val = dist.sample((self.batch_size,))

            if pk in ['thetas', 'phis']:
                val[val.isnan() | val.isinf()] = 0
                if pk == 'thetas':
                    # Put into range +/- pi
                    val = torch.atan2(torch.sin(val), torch.cos(val))
                else:
                    # Put into range +/- pi/2
                    val = torch.atan(torch.tan(val))
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
        I = torch.eye(3, dtype=torch.float64)[None, ...].repeat(self.batch_size, 1, 1)
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
