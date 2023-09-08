import json
import os
import shutil
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple

import numpy as np
import torch
from progress.bar import Bar
from scipy.stats import norm
from sklearn.decomposition import PCA

from wormlab3d import logger, PE_CACHE_PATH, N_WORKERS
from wormlab3d.data.model import PEParameters
from wormlab3d.particles.fractal_dimensions import calculate_box_dimension
from wormlab3d.particles.three_state_explorer import ThreeStateExplorer
from wormlab3d.particles.tumble_run import find_approximation
from wormlab3d.toolkit.util import hash_data, normalise
from wormlab3d.trajectories.pca import calculate_pcas, PCACache
from wormlab3d.trajectories.util import smooth_trajectory

VAR_NAMES = [
    'ts',
    'tumble_ts',
    'X',
    'states',
    'durations_0',
    'durations_1',
    'thetas',
    'phis',
    'intervals',
    'speeds'
]

VAR_NAMES_VARIABLE_SIZE = [
    'tumble_ts',
    'durations_0',
    'durations_1',
    'thetas',
    'phis',
    'intervals',
    'speeds'
]


def _to_numpy(
        t: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]], Dict[int, Any]]
) -> Union[np.ndarray, List[np.ndarray]]:
    """Returns a numpy version of a given tensor, accounting for grads and devices."""
    if type(t) == dict:
        return {k: _to_numpy(v) for k, v in t.items()}
    elif type(t) == list:
        return [_to_numpy(ti) for ti in t]
    elif type(t) == np.ndarray:
        return t
    elif t is None:
        return t
    return t.detach().cpu().numpy()


def _unique_X(X: np.ndarray):
    """Just calculates the unique count along the first axis and returns the count."""
    return np.unique(X, axis=0).shape[0]


def _compute_approximation_statistics(
        X: np.ndarray,
        e0: np.ndarray,
        dt: float,
        error_limits: List[float],
        planarity_window_vertices: int,
        min_run_speed_duration: Tuple[float, float],
        distance_first,
        height_first,
        smooth_e0_first,
        smooth_K_first,
) -> Dict[str, Dict[int, List[float]]]:
    """
    Calculate approximation statistics for a trajectory at different error limits.
    """
    keys = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles']
    results = {k: {j: [] for j in range(len(error_limits))} for k in keys}

    distance = distance_first
    distance_min = 3
    height = height_first
    smooth_e0 = smooth_e0_first
    smooth_K = smooth_K_first

    for j, error_limit in enumerate(error_limits):
        approx, distance, height, smooth_e0, smooth_K \
            = find_approximation(X, e0, error_limit, planarity_window_vertices, distance, distance_min, height,
                                 smooth_e0, smooth_K, max_attempts=50, quiet=True)
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles_j, nonplanar_angles_j, twist_angles_j, _, _, _ = approx

        # Put in time units
        run_durations *= dt
        run_speeds /= dt

        # Discard long runs where the distance travelled is too small
        include_idxs = np.unique(
            np.concatenate([
                np.argwhere(run_speeds > min_run_speed_duration[0]),
                np.argwhere(run_durations < min_run_speed_duration[1])
            ])
        )
        run_durations = run_durations[include_idxs]
        run_speeds = run_speeds[include_idxs]
        res_j = {
            'durations': run_durations,
            'speeds': run_speeds,
            'planar_angles': planar_angles_j,
            'nonplanar_angles': nonplanar_angles_j,
            'twist_angles': twist_angles_j
        }
        for k in keys:
            results[k][j] = res_j[k].tolist()

    return results


def _compute_approximation_statistics_wrapper(args):
    return _compute_approximation_statistics(*args)


class SimulationState:
    def __init__(
            self,
            parameters: PEParameters,
            read_only: bool = True,
            regenerate: bool = False,
            no_cache: bool = False,
            quiet: bool = False
    ):
        self.parameters: PEParameters = parameters
        self.read_only = read_only
        self.states = {}
        self.stats = {}
        self.meta = {'parameters': str(self.parameters.id)}
        self.needs_save = False
        self.no_cache = no_cache
        self.quiet = quiet

        # Extra properties not calculated immediately
        self.pcas = None
        self.dists = None

        # Load the state
        loaded = False
        if not regenerate and not no_cache:
            loaded = self._load_state(read_only)
            if not loaded and read_only:
                raise RuntimeError('Could not load simulation state.')
        elif regenerate:
            assert not read_only

        # (Re)generate data
        if not loaded:
            self._init_state()
            self._generate_state()
            self.save()

    @property
    def path(self) -> Path:
        return PE_CACHE_PATH / str(self.parameters.id)

    def __len__(self) -> int:
        return self.parameters.batch_size

    def _log(self, msg: str, level: str = 'info'):
        if not self.quiet:
            getattr(logger, level)(msg)

    def __getattr__(self, key: str):
        """
        Load data on demand.
        """
        if key == 'pe':
            self.pe = self._instantiate_explorer()
            return self.pe

        if key in self.states:
            return self.states[key]

        # Check disk
        path_state = self.path / f'{key}.npz'
        try:
            if key in self.shapes and self.shapes[key] != 'ragged':
                if self.shapes[key] == [0, ]:
                    return np.zeros((0,))
                mode = 'r' if self.read_only else 'r+'
                data = np.memmap(path_state, dtype=np.float32, mode=mode, shape=tuple(self.shapes[key]))
            elif key in VAR_NAMES_VARIABLE_SIZE:
                npz = np.load(path_state)
                data = [npz[f'{i:06d}'] for i in range(self.parameters.batch_size)]
            else:
                data = dict(np.load(path_state))
            self.states[key] = data
            return data
        except Exception:
            pass
        raise AttributeError(f'{key} not present!')

    def _load_state(
            self,
            read_only: bool = True,
            partial_load_ok: bool = False
    ) -> bool:
        """
        Try to load the state.
        """

        # Check for metadata first
        path_meta = self.path / 'metadata.json'
        if not path_meta.exists():
            return False
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            self._log(f'Could not load from {path_meta}. {e}', 'warning')
            return False

        # If metadata exists, use the shapes to load the other state files.
        states = {}
        mode = 'r' if read_only else 'r+'
        for k in VAR_NAMES:
            path_state = self.path / f'{k}.npz'
            try:
                if k in VAR_NAMES_VARIABLE_SIZE:
                    # Load on demand, just check that the file exists
                    assert path_state.exists()
                else:
                    state = np.memmap(path_state, dtype=np.float32, mode=mode, shape=tuple(meta['shapes'][k]))
                    states[k] = state
            except Exception as e:
                self._log(f'Could not load from {path_state}. {e}', 'warning')
                if not partial_load_ok:
                    return False

        self.meta = meta
        self.states = states
        self.shapes = meta['shapes']

        self._log(f'Loaded data from {self.path}.')

        return True

    def _init_state(self):
        """
        Initialise empty state.
        """
        if not self.no_cache:
            if self.path.exists():
                self._log(f'Wiping existing state in {self.path}.')
                shutil.rmtree(self.path)
            self._log(f'Initialising state in {self.path}.')
            os.makedirs(self.path, exist_ok=True)
        states = {}
        shapes = {}
        for k in VAR_NAMES:
            states[k], shapes[k] = self._init_state_component(k)
        self.states = states
        self.shapes = shapes

    def _init_state_component(self, k: str, k_like: str = None, shape: Tuple[int] = None):
        """
        Initialise an empty state component.
        """
        mp = self.parameters
        T = mp.n_steps
        path_state = self.path / f'{k}.npz'

        if shape == (0,):
            return np.zeros(shape), shape

        if k_like is None:
            k_like = k

        if k_like in VAR_NAMES_VARIABLE_SIZE:
            state = []
            shape = 'ragged'
        else:
            if k_like == 'ts':
                shape = (T,)
            elif k_like in ['X', 'Xt']:
                shape = (mp.batch_size, T, 3)
            elif k_like == 'states':
                shape = (mp.batch_size, T)
            elif k_like in ['nonp', 'coverage', 'crossings', 'fractal_dimensions']:
                shape = (mp.batch_size,)
            elif shape is None:
                raise RuntimeError(f'Unknown shape for variable key: {k}')
            if self.no_cache:
                state = np.zeros(shape, dtype=np.float32)
            else:
                state = np.memmap(path_state, dtype=np.float32, mode='w+', shape=shape)

        return state, shape

    def _generate_state(self):
        """
        Generate the simulation trajectories.
        """
        self._log('Generating simulation state.')
        res = self.pe.forward(T=self.parameters.duration, dt=self.parameters.dt)
        for k in VAR_NAMES:
            if k in VAR_NAMES_VARIABLE_SIZE:
                self.states[k] = _to_numpy(res[k])
            else:
                self.states[k][:] = _to_numpy(res[k])

    def _instantiate_explorer(self) -> ThreeStateExplorer:
        """
        Instantiate the particle explorer.
        """
        p = self.parameters

        # Sample speeds for the population
        if p.speeds_0_sig > 0:
            speeds_0_dist = norm(loc=p.speeds_0_mu, scale=p.speeds_0_sig)
            speeds0 = np.abs(speeds_0_dist.rvs(p.batch_size))
        else:
            speeds0 = p.speeds_0_mu
        if p.speeds_1_sig > 0:
            speeds_1_dist = norm(loc=p.speeds_1_mu, scale=p.speeds_1_sig)
            speeds1 = np.abs(speeds_1_dist.rvs(p.batch_size))
        else:
            speeds1 = p.speeds_1_mu

        pe = ThreeStateExplorer(
            batch_size=p.batch_size,
            rate_01=p.rate_01,
            rate_10=p.rate_10,
            rate_02=p.rate_02,
            rate_20=p.rate_20,  # not really a rate!
            speed_0=speeds0,
            speed_1=speeds1,
            theta_dist_params={
                'type': p.theta_dist_type,
                'params': p.theta_dist_params
            },
            phi_dist_params={
                'type': p.phi_dist_type,
                'params': p.phi_dist_params
            },
            nonp_pause_type=p.delta_type,
            nonp_pause_max=p.delta_max,
            quiet=self.quiet
        )

        return pe

    def save(self):
        """
        Save the states to the hard drive.
        """
        if self.no_cache:
            return
        self._log(f'Saving simulation state to {self.path}.')

        for k, v in self.states.items():
            if (k in VAR_NAMES and k not in VAR_NAMES_VARIABLE_SIZE) \
                    or (k in self.shapes and self.shapes[k] != 'ragged'):
                if self.shapes[k] != (0,):
                    self.states[k].flush()
            else:
                path_state = self.path / f'{k}.npz'

                if k in VAR_NAMES_VARIABLE_SIZE:
                    data = {}
                    for i in range(self.parameters.batch_size):
                        data[f'{i:06d}'] = v[i]
                    np.savez(path_state, **data)
                else:
                    np.savez(path_state, **v)

        # Save the meta data
        meta = {**self.meta, 'shapes': self.shapes}
        with open(self.path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, separators=(',', ': '))

    def get_approximation_statistics(
            self,
            error_limits: List[float],
            noise_scale: Optional[float],
            smoothing_window: Optional[int],
            planarity_window: int,
            min_run_speed_duration: Tuple[float, float] = (0.01, 60.),
            distance_first: int = 500,
            height_first: int = 100,
            smooth_e0_first: int = 201,
            smooth_K_first: int = 201,
    ):
        """
        Recalculate the run and tumble statistics using the same method used for real data.
        """
        args = {
            'noise_scale': f'{noise_scale:.4f}' if noise_scale is not None else '',
            'smoothing_window': smoothing_window,
            'planarity_window': planarity_window,
            'min_run_speed_duration': f'{min_run_speed_duration[0]:.4f},{min_run_speed_duration[1]:.4f}',
            'distance_first': distance_first,
            'height_first': height_first,
            'smooth_e0_first': smooth_e0_first,
            'smooth_K_first': smooth_K_first,
        }
        arg_key = hash_data(args)
        keys = ['durations', 'speeds', 'planar_angles', 'nonplanar_angles', 'twist_angles']

        # Check if data has already been calculated
        if 'approx_stats' in self.meta and arg_key in self.meta['approx_stats']:
            info = self.meta['approx_stats'][arg_key]
            error_limits_calculated = info['error_limits']
            error_limits_to_calculate = [l for l in error_limits if l not in error_limits_calculated]
        else:
            error_limits_to_calculate = error_limits
        N = len(error_limits_to_calculate)

        # Calculate anything missing
        if N > 0:
            self._log('Calculating approximation statistics.')
            results = {k: {j: [] for j in range(N)} for k in keys}

            # Add some noise to the trajectories then smooth
            Xs = self.X.copy().astype(np.float64)
            Xs = Xs - Xs.mean(axis=1, keepdims=True)
            if noise_scale is not None and noise_scale > 0:
                Xs = Xs + np.random.normal(np.zeros_like(Xs), noise_scale)
            if smoothing_window is not None and smoothing_window > 0:
                Xs = smooth_trajectory(Xs.transpose(1, 0, 2), window_len=smoothing_window).transpose(1, 0, 2)
            e0s = normalise(np.gradient(Xs, axis=1))

            # Common args
            common_args = [
                self.parameters.dt,
                error_limits,
                planarity_window,
                min_run_speed_duration,
                distance_first,
                height_first,
                smooth_e0_first,
                smooth_K_first,
            ]

            if N_WORKERS > 0:
                with Pool(processes=N_WORKERS) as pool:
                    res = pool.map(
                        _compute_approximation_statistics_wrapper,
                        [(Xs[i], e0s[i], *common_args) for i in range(self.parameters.batch_size)]
                    )
                for k in keys:
                    for j in range(N):
                        for i in range(self.parameters.batch_size):
                            results[k][j].extend(res[i][k][j])
            else:
                for i, X in enumerate(self.states['X']):
                    self._log(f'Computing tumble-run model for sim={i + 1}/{self.parameters.batch_size}.')
                    res_i = _compute_approximation_statistics(Xs[i], e0s[i], *common_args)
                    for k in keys:
                        for j in range(N):
                            results[k][j].extend(res_i[k][j])

            # Add the results to the state
            if 'approx_stats' not in self.meta:
                self.meta['approx_stats'] = {}
            if arg_key not in self.meta['approx_stats']:
                self.meta['approx_stats'][arg_key] = {
                    'args': args,
                    'error_limits': [],
                    'results': {}
                }
            self.meta['approx_stats'][arg_key]['error_limits'].extend(error_limits_to_calculate)
            self.meta['approx_stats'][arg_key]['error_limits'].sort()
            for j, error_limit in enumerate(error_limits_to_calculate):
                self.states[f'approx_stats_{arg_key}_{error_limit:.4f}'] = {
                    k: np.array(results[k][j])
                    for k in keys
                }
            self.save()

        # Load from state
        N = len(error_limits)
        results = {k: {i: [] for i in range(N)} for k in keys}
        for j, error_limit in enumerate(error_limits):
            data = getattr(self, f'approx_stats_{arg_key}_{error_limit:.4f}')
            for k in keys:
                results[k][j] = data[k]

        return results

    def get_nonp(self):
        """
        Calculate non-planarity of trajectories and cache result to disk.
        """
        try:
            nonp = getattr(self, 'nonp')
            return nonp
        except AttributeError:
            pass

        self._log('Calculating non-planarity of trajectories.')
        bs = self.parameters.batch_size
        nonp = np.zeros(bs)
        for i, X in enumerate(self.X):
            pca = self.get_pca(i)
            r = pca.explained_variance_ratio_.T
            nonp[i] = r[2] / np.sqrt(r[1] * r[0])
        self.states['nonp'], self.shapes['nonp'] = self._init_state_component('nonp')
        self.states['nonp'][:] = nonp
        self.needs_save = True

        return nonp

    def get_nonp_windowed(self, ws: int, parallel: bool = True):
        """
        Calculate windowed non-planarity of trajectories and cache result to disk.
        """
        key = f'nonp_{ws:04d}'
        try:
            nonp = getattr(self, key)
            return nonp
        except AttributeError:
            pass

        self._log(f'Calculating non-planarity of trajectories with window size={ws}.')
        bs = self.parameters.batch_size
        nonp = np.zeros((bs, self.X.shape[1] - int(ws / 2) * 2))
        for i, X in enumerate(self.X):
            pcas = calculate_pcas(X, window_size=ws, parallel=parallel)
            pcas_cache = PCACache(pcas)
            r = pcas_cache.explained_variance_ratio.T
            nonp[i] = r[2] / np.where(r[2] == 0, 1, np.sqrt(r[1] * r[0]))
        self.states[key], self.shapes[key] = self._init_state_component(key, shape=tuple(nonp.shape))
        self.states[key][:] = nonp
        self.needs_save = True

        return nonp

    def get_coverage(self, vs: float, parallel: bool = True) -> np.ndarray:
        """
        Calculate the trajectory voxel coverage at specified voxel size.
        """
        key = f'coverage_{vs:.3E}'
        try:
            coverage = getattr(self, key)
            return coverage
        except AttributeError:
            pass

        # Discretise the trajectories
        Xd = np.round(self.X / vs).astype(np.int32)

        # Count unique voxels visited multiplied by voxel size
        if parallel and N_WORKERS > 1:
            with Pool(processes=N_WORKERS) as pool:
                res = pool.map(
                    _unique_X,
                    [X for X in Xd]
                )
            coverage = np.array(res) * vs
        else:
            coverage = np.zeros(self.parameters.batch_size)
            for j, X in enumerate(Xd):
                n_voxels = np.unique(X, axis=0).shape[0]
                coverage[j] = n_voxels * vs

        self.states[key], self.shapes[key] = self._init_state_component(key, 'coverage')
        self.states[key][:] = coverage
        self.needs_save = True

        return coverage

    def get_msds(self, deltas: np.ndarray) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        Calculate the msds for the trajectories.
        """
        key = f'msds_{hash_data([f"{d:.5f}" for d in deltas])}'
        try:
            msds_all = getattr(self, key + '_all')
            msds = getattr(self, key)
            return msds_all, msds
        except AttributeError:
            pass

        msds_all = np.zeros(len(deltas))
        # msds_all = {}
        msds = np.zeros((self.parameters.batch_size, len(deltas)))
        # msds = {i: {} for i in range(self.parameters.batch_size)}
        bar = Bar('Calculating MSDs', max=len(deltas))
        bar.check_tty = False
        for i, delta in enumerate(deltas):
            d = np.sum((self.X[:, delta:] - self.X[:, :-delta])**2, axis=-1)
            msds_all[i] = d.mean()
            for j in range(self.parameters.batch_size):
                msds[j, i] = d[j].mean()
            bar.next()
        bar.finish()

        self.states[key + '_all'], self.shapes[key + '_all'] = self._init_state_component(
            key + '_all', shape=tuple(msds_all.shape)
        )
        self.states[key + '_all'][:] = msds_all
        self.states[key], self.shapes[key] = self._init_state_component(key, shape=tuple(msds.shape))
        self.states[key][:] = msds
        self.needs_save = True

        return msds_all, msds

    def get_crossings(self, radius: float) -> np.ndarray:
        """
        Get the number of times each trajectory has crossed a sphere of given radius.
        """
        key = f'crossings_{radius:.4E}'
        try:
            crossings = getattr(self, key)
            return crossings
        except AttributeError:
            pass

        # Calculate crossings
        outside = np.c_[np.zeros((self.parameters.batch_size, 1), dtype=bool), self._get_dists() > radius]
        crossings_binary = outside[:, 1:] ^ outside[:, :-1]
        crossings = crossings_binary.sum(axis=-1)

        self.states[key], self.shapes[key] = self._init_state_component(key, k_like='crossings')
        self.states[key][:] = crossings
        self.needs_save = True

        return crossings

    def get_crossings_nonp(self, radius: float) -> np.ndarray:
        """
        Calculate the non-planarity for all crossing points at given radius relative to the principal plane of the trajectory.
        """
        key = f'crossings_nonp_{radius:.4E}'
        try:
            crossings_nonp = getattr(self, key)
            return crossings_nonp
        except AttributeError:
            pass

        # Calculate crossings
        outside = np.c_[np.zeros((self.parameters.batch_size, 1), dtype=bool), self._get_dists() > radius]
        crossings = outside[:, 1:] ^ outside[:, :-1]

        # Loop over the trajectories
        nonp = []
        for i, X in enumerate(self.X):
            crossing_idxs = crossings[i].nonzero()
            cc = self.X[i][crossing_idxs]
            if len(cc) > 0:
                pca = self.get_pca(i)
                R = np.stack(pca.components_, axis=1)
                cct = np.einsum('ij,bj->bi', R.T, cc)
                nonp.append(np.abs(cct[:, 2]))
        if len(nonp) > 0:
            nonp = np.concatenate(nonp)
        else:
            nonp = np.zeros((0,))

        self.states[key], self.shapes[key] = self._init_state_component(key, shape=tuple(nonp.shape))
        self.states[key][:] = nonp
        self.needs_save = True

        return nonp

    def get_Xt(self) -> np.ndarray:
        """
        Transform the coordinates relative to the principal components of the trajectory.
        """
        try:
            Xt = getattr(self, 'Xt')
            return Xt
        except AttributeError:
            pass

        # Loop over the trajectories, calculate the PCA and transform into the basis vectors.
        Xt = np.zeros_like(self.X)
        for i, X in enumerate(self.X):
            pca = self.get_pca(i)
            R = np.stack(pca.components_, axis=1)
            Xt[i] = np.einsum('ij,bj->bi', R.T, X)

        self.states['Xt'], self.shapes['Xt'] = self._init_state_component('Xt')
        self.states['Xt'][:] = Xt
        self.needs_save = True

        return Xt

    def get_fractal_dimensions(
            self,
            noise_scale: Optional[float],
            smoothing_window: Optional[int],
            voxel_sizes: np.ndarray,
            plateau_threshold: float,
            sample_size: int = 1,
            sf_min: float = 1.,
            sf_max: float = 1.,
    ) -> np.ndarray:
        """
        Calculate the fractal dimensions of the trajectories.
        """

        args = {
            'noise_scale': f'{noise_scale:.4f}' if noise_scale is not None else '',
            'smoothing_window': smoothing_window,
            'vxs': [f'{v:.5E}' for v in voxel_sizes],
            'pt': plateau_threshold,
            'ss': sample_size,
            'sf_min': sf_min,
            'sf_max': sf_max,
        }
        k = 'fractal_dimensions_' + hash_data(args)

        try:
            fractal_dimensions = getattr(self, k)
            return fractal_dimensions
        except AttributeError:
            pass

        # Add some noise to the trajectories then smooth
        Xs = self.X.copy().astype(np.float64)
        Xs = Xs - Xs.mean(axis=1, keepdims=True)
        if noise_scale is not None and noise_scale > 0:
            Xs = Xs + np.random.normal(np.zeros_like(Xs), noise_scale)
        if smoothing_window is not None and smoothing_window > 0:
            Xs = smooth_trajectory(Xs.transpose(1, 0, 2), window_len=smoothing_window).transpose(1, 0, 2)

        # Loop over the trajectories and calculate the dimensions.
        dims = np.zeros(len(self))
        for i, X in enumerate(Xs):
            dims[i] = calculate_box_dimension(
                X=X,
                voxel_sizes=voxel_sizes,
                plateau_threshold=plateau_threshold,
                sample_size=sample_size,
                sf_min=sf_min,
                sf_max=sf_max,
                parallel=True
            )['D']

        self.states[k], self.shapes[k] = self._init_state_component(
            k=k,
            k_like='fractal_dimensions',
        )
        self.states[k][:] = dims
        self.needs_save = True

        return dims

    def _get_dists(self) -> np.ndarray:
        """
        Cache the norms.
        """
        if self.dists is None:
            self.dists = np.linalg.norm(self.X, axis=-1)
        return self.dists

    def get_pca(self, idx: int) -> PCA:
        """
        Cache the PCA calculations.
        """
        if self.pcas is None:
            self.pcas = {}
        if idx not in self.pcas:
            pca = PCA(svd_solver='full', copy=True, n_components=3)
            pca.fit(self.X[idx])
            self.pcas[idx] = pca
        return self.pcas[idx]
