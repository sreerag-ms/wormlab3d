import json
import os
from argparse import Namespace
from typing import Tuple, Dict, Any, List, Union, Optional

import numpy as np
import torch
from numpy.lib.npyio import NpzFile
from progress.bar import Bar
from scipy.spatial import KDTree
from scipy.stats import norm
from sklearn.decomposition import PCA

from wormlab3d import logger, PE_CACHE_PATH, N_WORKERS
from wormlab3d.particles.three_state_explorer import ThreeStateExplorer
from wormlab3d.toolkit.util import hash_data


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


class TrajectoryCache:
    def __init__(
            self,
            batch_size: int,
            ts: Optional[Union[np.ndarray, torch.Tensor]] = None,
            tumble_ts: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
            X: Optional[Union[np.ndarray, torch.Tensor]] = None,
            states: Optional[Union[np.ndarray, torch.Tensor]] = None,
            durations: Optional[Dict[int, List[Union[np.ndarray, torch.Tensor]]]] = None,
            thetas: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
            phis: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
            intervals: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
            speeds: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
            pe_args: Optional[Dict[str, Any]] = None,
            rt_args: Optional[Dict[str, Any]] = None,
            data: Optional[NpzFile] = None,
            meta: Optional[Dict[str, Any]] = None,
    ):
        self.batch_size = batch_size
        self.ts = _to_numpy(ts)
        self.tumble_ts = _to_numpy(tumble_ts)
        self.X = _to_numpy(X)
        self.states = _to_numpy(states)
        self.durations = _to_numpy(durations)
        self.thetas = _to_numpy(thetas)
        self.phis = _to_numpy(phis)
        self.intervals = _to_numpy(intervals)
        self.speeds = _to_numpy(speeds)
        self.pe_args = pe_args
        self.rt_args = rt_args

        if data is None:
            data = {}
        self.data = data
        if meta is None:
            meta = {}
        self.meta = meta

        # Extra properties not calculated immediately
        self.nonp = None
        self.coverage = None
        self.msds_all = None
        self.msds = None
        self.targets = {}
        self.finds = None
        self.find_times = None
        self.finds_pop = None
        self.crossings = None
        self.crossings_nonp = None
        self.dists = None
        self.pcas = None
        self.Xt = None
        self.needs_save = False

    def __getattribute__(self, k: str):
        data_keys = ['ts', 'tumble_ts', 'X', 'states', 'durations', 'thetas', 'phis', 'intervals', 'speeds']
        if k not in data_keys:
            return super().__getattribute__(k)

        # Check if the variable has been defined or loaded already
        v = super().__getattribute__(k)
        if v is not None:
            return v

        # Load from disk
        if k in ['ts', 'X', 'states']:
            v = self.data[k]
        elif k in ['tumble_ts', 'thetas', 'phis', 'intervals', 'speeds']:
            v = [self.data[f'{k}_{i:06d}'] for i in range(self.batch_size)]
        elif k == 'durations':
            v = {
                0: [self.data[f'durations0_{i:06d}'] for i in range(self.batch_size)],
                1: [self.data[f'durations1_{i:06d}'] for i in range(self.batch_size)]
            }

        setattr(self, k, v)
        return v

    @property
    def arg_hash(self):
        return hash_data({**self.pe_args, **self.rt_args})

    @property
    def path_meta(self):
        return PE_CACHE_PATH / f'{self.arg_hash}.meta'

    @property
    def path_data(self):
        return PE_CACHE_PATH / f'{self.arg_hash}.npz'

    def save(self):
        """
        Save the data and metadata to disk.
        """
        logger.info(f'Saving trajectory data to {self.path_data}.')
        meta = {**self.pe_args, **self.rt_args}
        data = dict(self.data)
        data['ts'] = self.ts
        data['X'] = self.X
        data['states'] = self.states
        for i in range(self.batch_size):
            data[f'tumble_ts_{i:06d}'] = self.tumble_ts[i]
            data[f'durations0_{i:06d}'] = self.durations[0][i]
            data[f'durations1_{i:06d}'] = self.durations[1][i]
            data[f'thetas_{i:06d}'] = self.thetas[i]
            data[f'phis_{i:06d}'] = self.phis[i]
            data[f'intervals_{i:06d}'] = self.intervals[i]
            data[f'speeds_{i:06d}'] = self.speeds[i]
        if self.nonp is not None:
            data['nonp'] = self.nonp
        if self.coverage is not None:
            coverage_keys = []
            for k, v in self.coverage.items():
                data[f'coverage_{k}'] = v
                coverage_keys.append(k)
            meta['coverage_keys'] = coverage_keys
        if self.msds_all is not None:
            deltas = list(map(int, list(self.msds_all.keys())))
            meta['msd_deltas'] = deltas
            data['msds_all'] = np.array(list(self.msds_all.values()))
            data['msds'] = np.stack([list(self.msds[i].values()) for i in range(self.batch_size)])
        if self.finds is not None:
            r_keys = []
            n_keys = []
            e_keys = []
            for r, finds_r in self.finds.items():
                r_keys.append(r)
                for n, finds_rt in finds_r.items():
                    n_keys.append(n)
                    for e, finds_rte in finds_rt.items():
                        e_keys.append(e)
                        data[f'finds_r{r}_n{n}_e{e}'] = finds_rte
                        data[f'find_times_r{r}_n{n}_e{e}'] = self.find_times[r][n][e]
                        data[f'finds_pop_r{r}_n{n}_e{e}'] = self.finds_pop[r][n][e]
            meta['finds_r_keys'] = list(set(r_keys))
            meta['finds_n_keys'] = list(set(n_keys))
            meta['finds_e_keys'] = list(set(e_keys))
        if self.crossings is not None:
            r_keys = list(self.crossings.keys())
            meta['crossings_r_keys'] = list(set(r_keys))
            for r_key, crossings_r in self.crossings.items():
                data[f'crossings_r{r_key}'] = crossings_r
        if self.crossings_nonp is not None:
            r_keys = list(self.crossings_nonp.keys())
            meta['crossings_nonp_r_keys'] = list(set(r_keys))
            for r_key, crossings_nonp_r in self.crossings_nonp.items():
                data[f'crossings_nonp_r{r_key}'] = crossings_nonp_r
        if self.Xt is not None:
            data['Xt'] = self.Xt

        np.savez_compressed(self.path_data, **data)
        with open(self.path_meta, 'w') as f:
            json.dump(meta, f)

    def get_nonp(self) -> np.ndarray:
        """
        Calculate non-planarity of trajectories and cache result to disk.
        """
        if self.nonp is None and 'nonp' in self.data:
            self.nonp = self.data['nonp']
        if self.nonp is not None:
            return self.nonp
        logger.info('Calculating non-planarity of trajectories.')
        nonp = np.zeros(self.batch_size)
        for i, X in enumerate(self.X):
            pca = self._get_pca(i)
            r = pca.explained_variance_ratio_.T
            nonp[i] = r[2] / np.sqrt(r[1] * r[0])
        self.nonp = nonp
        self.needs_save = True
        return nonp

    def get_coverage(self, vs: float) -> np.ndarray:
        """
        Calculate the trajectory voxel coverage at specified voxel size.
        """
        if self.coverage is None and 'coverage_keys' in self.meta:
            coverage_keys = self.meta['coverage_keys']
            self.coverage = {k: self.data[f'coverage_{k}'] for k in coverage_keys}
        vs_key = f'{vs:.4E}'
        if self.coverage is not None and vs_key in self.coverage:
            return self.coverage[vs_key]
        if self.coverage is None:
            self.coverage = {}

        # Discretise the trajectories
        Xd = np.round(self.X / vs).astype(np.int32)

        # Score the trajectories as the sum of unique voxels visited multiplied by voxel size
        coverage_vs = np.zeros(self.batch_size)
        for j, X in enumerate(Xd):
            n_voxels = np.unique(X, axis=0).shape[0]
            coverage_vs[j] = n_voxels * vs
        self.coverage[vs_key] = coverage_vs
        self.needs_save = True
        return coverage_vs

    def get_msds(self, deltas: np.ndarray) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
        """
        Calculate the msds for the trajectories.
        """
        if self.msds_all is None and 'msd_deltas' in self.meta:
            logger.info('Loading msds from disk.')
            deltas = self.meta['msd_deltas']
            self.msds_all = {
                delta: self.data[f'msds_all'][d]
                for d, delta in enumerate(deltas)
            }
            self.msds = {
                i: {
                    delta: self.data[f'msds'][i, d]
                    for d, delta in enumerate(deltas)
                }
                for i in range(self.batch_size)
            }

        if self.msds_all is None or self.msds is None:
            self.needs_save = True
            self.msds_all = {}
            self.msds = {i: {} for i in range(self.batch_size)}

        bar = Bar('Calculating', max=len(deltas))
        bar.check_tty = False
        for delta in deltas:
            if delta not in self.msds_all:
                self.needs_save = True
                d = np.sum((self.X[:, delta:] - self.X[:, :-delta])**2, axis=-1)
                self.msds_all[delta] = d.mean()
                for i in range(self.batch_size):
                    self.msds[i][delta] = d[i].mean()
            bar.next()
        bar.finish()

        return self.msds_all, self.msds

    def get_finds(self, radius: float, n_targets: int, epsilon: float, parallel: bool = True) \
            -> Dict[str, Dict[str, Dict[int, Dict[str, np.ndarray]]]]:
        """
        Calculate which trajectories found targets at different radius and placement density.
        """
        if self.finds is None and 'finds_r_keys' in self.meta:
            logger.info('Loading finds from disk.')
            self.finds = {}
            self.find_times = {}
            self.finds_pop = {}
            for r in self.meta['finds_r_keys']:
                self.finds[r] = {}
                self.find_times[r] = {}
                self.finds_pop[r] = {}
                for n in self.meta['finds_n_keys']:
                    self.finds[r][n] = {}
                    self.find_times[r][n] = {}
                    self.finds_pop[r][n] = {}
                    for e in self.meta['finds_e_keys']:
                        if f'finds_r{r}_n{n}_e{e}' in self.data:
                            self.finds[r][n][e] = self.data[f'finds_r{r}_n{n}_e{e}']
                            self.find_times[r][n][e] = self.data[f'find_times_r{r}_n{n}_e{e}']
                            self.finds_pop[r][n][e] = self.data[f'finds_pop_r{r}_n{n}_e{e}']

        r_key = f'{radius:.4f}'
        e_key = f'{epsilon:.4f}'
        if self.finds is not None \
                and r_key in self.finds \
                and n_targets in self.finds[r_key] \
                and e_key in self.finds[r_key][n_targets]:
            return {
                'finds': self.finds[r_key][n_targets][e_key],
                'find_times': self.find_times[r_key][n_targets][e_key],
                'finds_pop': self.finds_pop[r_key][n_targets][e_key],
            }
        if self.finds is None:
            self.finds = {}
            self.find_times = {}
            self.finds_pop = {}
        if r_key not in self.finds:
            self.finds[r_key] = {}
            self.find_times[r_key] = {}
            self.finds_pop[r_key] = {}
        if n_targets not in self.finds[r_key]:
            self.finds[r_key][n_targets] = {}
            self.find_times[r_key][n_targets] = {}
            self.finds_pop[r_key][n_targets] = {}

        finds = np.zeros(self.batch_size, dtype=np.int32)
        find_times = np.ones((self.batch_size, n_targets)) * -1
        finds_pop = np.zeros(n_targets, dtype=np.bool)
        targets = self._get_targets(n_targets)

        # for i, X in enumerate(self.X):
        #     X_tree = KDTree(X)
        #     res = X_tree.query_ball_point(
        #         targets * radius,
        #         epsilon,
        #         workers=N_WORKERS if parallel else 1,
        #         return_sorted=True
        #     )
        #     found_targets = [int(len(idxs) > 0) for idxs in res]
        #     finds[i] = sum(found_targets)
        #     find_times[i] = [-1 if len(idxs) == 0 else idxs[0] * self.rt_args['dt'] for idxs in res]
        #     finds_pop = finds_pop | found_targets

        X_tree = KDTree(np.concatenate(self.X, axis=0))
        res = X_tree.query_ball_point(
            targets * radius,
            epsilon,
            workers=N_WORKERS if parallel else 1,
            return_sorted=True
        )
        res = [np.array(r) for r in res]
        T = len(self.ts)
        for i in range(self.batch_size):
            res_i = [
                res[t][(res[t] >= i * T) & (res[t] < (i + 1) * T)] - i * T
                for t in range(n_targets)
            ]
            found_targets = [int(len(idxs) > 0) for idxs in res_i]
            finds[i] = sum(found_targets)
            find_times[i] = [-1 if len(idxs) == 0 else idxs[0] * self.rt_args['dt'] for idxs in res_i]
            finds_pop = finds_pop | found_targets

        self.finds[r_key][n_targets][e_key] = finds
        self.find_times[r_key][n_targets][e_key] = find_times
        self.finds_pop[r_key][n_targets][e_key] = finds_pop
        self.needs_save = True
        return {
            'finds': self.finds[r_key][n_targets][e_key],
            'find_times': self.find_times[r_key][n_targets][e_key],
            'finds_pop': self.finds_pop[r_key][n_targets][e_key],
        }

    def _get_targets(self, n_targets: int) -> np.ndarray:
        """
        Generate targets using the golden spiral method:
        https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
        """
        if n_targets in self.targets:
            return self.targets[n_targets]
        idxs = np.arange(n_targets) + 0.5
        phi = np.arccos(1 - 2 * idxs / n_targets)
        theta = np.pi * (1 + 5**0.5) * idxs
        targets = np.stack([
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi)
        ], axis=1)
        self.targets[n_targets] = targets
        return targets

    def get_crossings(self, radius: float) -> np.ndarray:
        """
        Get the number of times each trajectory has crossed the sphere of given radius.
        """
        if self.crossings is None and 'crossings_r_keys' in self.meta:
            r_keys = self.meta['crossings_r_keys']
            self.crossings = {r_key: self.data[f'crossings_r{r_key}'] for r_key in r_keys}
        r_key = f'{radius:.3f}'
        if self.crossings is not None and r_key in self.crossings:
            return self.crossings[r_key]
        if self.crossings is None:
            self.crossings = {}

        # Calculate crossings
        outside = np.c_[np.zeros((self.batch_size, 1), dtype=bool), self._get_dists() > radius]
        crossings = outside[:, 1:] ^ outside[:, :-1]
        self.crossings[r_key] = crossings.sum(axis=-1)
        self.needs_save = True

        return self.crossings[r_key]

    def get_crossings_nonp(self, radius: float) -> np.ndarray:
        """
        Calculate the non-planarity for all crossing points at given radius relative to the principal plane of the trajectory.
        """
        if self.crossings_nonp is None and 'crossings_nonp_r_keys' in self.meta:
            r_keys = self.meta['crossings_nonp_r_keys']
            self.crossings_nonp = {r_key: self.data[f'crossings_nonp_r{r_key}'] for r_key in r_keys}
        r_key = f'{radius:.3f}'
        if self.crossings_nonp is not None and r_key in self.crossings_nonp:
            return self.crossings_nonp[r_key]
        if self.crossings_nonp is None:
            self.crossings_nonp = {}

        # Calculate crossings
        outside = np.c_[np.zeros((self.batch_size, 1), dtype=bool), self._get_dists() > radius]
        crossings = outside[:, 1:] ^ outside[:, :-1]

        # Loop over the trajectories
        nonp = []
        for i, X in enumerate(self.X):
            crossing_idxs = crossings[i].nonzero()
            cc = self.X[i][crossing_idxs]
            if len(cc) > 0:
                pca = self._get_pca(i)
                R = np.stack(pca.components_, axis=1)
                cct = np.einsum('ij,bj->bi', R.T, cc)
                nonp.append(np.abs(cct[:, 2]))
        if len(nonp) > 0:
            nonp = np.concatenate(nonp)
        else:
            nonp = np.zeros((0,))

        self.crossings_nonp[r_key] = nonp
        self.needs_save = True

        return self.crossings_nonp[r_key]

    def get_Xt(self) -> np.ndarray:
        """
        Transform the coordinates relative to the principal components of the trajectory.
        """
        if self.Xt is None and 'Xt' in self.data:
            self.Xt = self.data['Xt']
        if self.Xt is not None:
            return self.Xt

        # Loop over the trajectories, calculate the PCA and transform into the basis vectors.
        Xt = np.zeros_like(self.X)
        for i, X in enumerate(self.X):
            pca = self._get_pca(i)
            R = np.stack(pca.components_, axis=1)
            Xt[i] = np.einsum('ij,bj->bi', R.T, X)

        self.Xt = Xt
        self.needs_save = True

        return self.Xt

    def _get_dists(self) -> np.ndarray:
        """
        Cache the norms.
        """
        if self.dists is None:
            self.dists = np.linalg.norm(self.X, axis=-1)
        return self.dists

    def _get_pca(self, idx: int) -> PCA:
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


def _unpack_cache_data(pe_args, rt_args) -> TrajectoryCache:
    """
    Load the disk cache into the object.
    """
    logger.info('Unpacking cache data.')
    batch_size = pe_args['batch_size']
    arg_hash = hash_data({**pe_args, **rt_args})
    filename_meta = f'{arg_hash}.meta'
    filename_traj = f'{arg_hash}.npz'
    path_meta = PE_CACHE_PATH / filename_meta
    path_traj = PE_CACHE_PATH / filename_traj
    with open(path_meta, 'r') as f:
        meta = json.load(f)
    data = np.load(path_traj)

    return TrajectoryCache(
        batch_size=batch_size,
        rt_args=rt_args,
        pe_args=pe_args,
        data=data,
        meta=meta,
    )


def instantiate_explorer(
        batch_size: int,

        rate_01: float,
        rate_10: float,
        rate_02: float,
        rate_20: float,
        speeds_0_mu: float,
        speeds_0_sig: float,
        speeds_1_mu: float,
        speeds_1_sig: float,

        theta_dist_type: str,
        theta_dist_params: Tuple[float],
        phi_dist_type: str,
        phi_dist_params: Tuple[float],

        nonp_pause_type: str,
        nonp_pause_max: float,
) -> ThreeStateExplorer:
    """
    Instantiate the particle explorer and generate trajectories.
    """
    if speeds_0_sig > 0:
        speeds_0_dist = norm(loc=speeds_0_mu, scale=speeds_0_sig)
        speeds0 = np.abs(speeds_0_dist.rvs(batch_size))
    else:
        speeds0 = speeds_0_mu
    if speeds_1_sig > 0:
        speeds_1_dist = norm(loc=speeds_1_mu, scale=speeds_1_sig)
        speeds1 = np.abs(speeds_1_dist.rvs(batch_size))
    else:
        speeds1 = speeds_1_mu

    pe = ThreeStateExplorer(
        batch_size=batch_size,
        rate_01=rate_01,
        rate_10=rate_10,
        rate_02=rate_02,
        rate_20=rate_20,  # not really a rate!
        speed_0=speeds0,
        speed_1=speeds1,
        theta_dist_params={
            'type': theta_dist_type,
            'params': theta_dist_params
        },
        phi_dist_params={
            'type': phi_dist_type,
            'params': phi_dist_params
        },
        nonp_pause_type=nonp_pause_type,
        nonp_pause_max=nonp_pause_max
    )

    return pe


def generate_trajectories(
        pe: ThreeStateExplorer,
        pe_args: Dict[str, Any],
        rt_args: Dict[str, Any],
) -> TrajectoryCache:
    """
    Generate trajectories.
    """
    TC = pe.forward(**rt_args)
    TC.pe_args = pe_args
    TC.rt_args = rt_args
    return TC


def generate_or_load_trajectories(
        T: float,
        dt: float,
        batch_size: int,

        rate_01: float,
        rate_10: float,
        rate_02: float,
        rate_20: float,
        speeds_0_mu: float,
        speeds_0_sig: float,
        speeds_1_mu: float,
        speeds_1_sig: float,

        theta_dist_type: str,
        theta_dist_params: Tuple[float],
        phi_dist_type: str,
        phi_dist_params: Tuple[float],

        nonp_pause_type: str,
        nonp_pause_max: float,

        rebuild_cache: bool = False,
) -> Tuple[ThreeStateExplorer, TrajectoryCache]:
    """
    Try to load an existing trajectory cache or generate it otherwise.
    """
    pe_args = {
        'batch_size': batch_size,
        'rate_01': rate_01,
        'rate_10': rate_10,
        'rate_02': rate_02,
        'rate_20': rate_20,
        'speeds_0_mu': speeds_0_mu,
        'speeds_0_sig': speeds_0_sig,
        'speeds_1_mu': speeds_1_mu,
        'speeds_1_sig': speeds_1_sig,
        'theta_dist_type': theta_dist_type,
        'theta_dist_params': theta_dist_params,
        'phi_dist_type': phi_dist_type,
        'phi_dist_params': phi_dist_params,
        'nonp_pause_type': nonp_pause_type,
        'nonp_pause_max': nonp_pause_max,
    }
    rt_args = {
        'T': T,
        'dt': dt,
    }
    pe = instantiate_explorer(**pe_args)
    arg_hash = hash_data({**pe_args, **rt_args})
    filename_meta = f'{arg_hash}.meta'
    filename_traj = f'{arg_hash}.npz'
    path_meta = PE_CACHE_PATH / filename_meta
    path_traj = PE_CACHE_PATH / filename_traj
    if not rebuild_cache and os.path.exists(path_meta) and os.path.exists(path_traj):
        try:
            TC = _unpack_cache_data(pe_args, rt_args)
            logger.info(f'Loaded trajectory data from {path_traj}.')
            return pe, TC
        except Exception as e:
            logger.warning(f'Could not load trajectory data from {path_traj}. {e}')
    elif not rebuild_cache:
        logger.info('Trajectory file cache unavailable, building.')
    else:
        logger.info('Rebuilding trajectory cache.')

    # Generate and save the trajectory data cache
    logger.info(f'Generating trajectories.')
    TC = generate_trajectories(pe, pe_args, rt_args)
    TC.save()

    return pe, TC


def get_trajectories_from_args(args: Namespace) -> Tuple[ThreeStateExplorer, TrajectoryCache]:
    """
    Generate or load the trajectories from parameters set in an argument namespace.
    """
    pe, TC = generate_or_load_trajectories(
        T=args.sim_duration,
        dt=args.sim_dt,
        batch_size=args.batch_size,
        rate_01=args.rate_01,
        rate_10=args.rate_10,
        rate_02=args.rate_02,
        rate_20=args.rate_20,
        speeds_0_mu=args.speeds_0_mu,
        speeds_0_sig=args.speeds_0_sig,
        speeds_1_mu=args.speeds_1_mu,
        speeds_1_sig=args.speeds_1_sig,
        theta_dist_type=args.theta_dist_type,
        theta_dist_params=args.theta_dist_params,
        phi_dist_type=args.phi_dist_type,
        phi_dist_params=args.phi_dist_params,
        nonp_pause_type=args.nonp_pause_type,
        nonp_pause_max=args.nonp_pause_max,
        rebuild_cache=args.rebuild_cache,
    )
    return pe, TC
