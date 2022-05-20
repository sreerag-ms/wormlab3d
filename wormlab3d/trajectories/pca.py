import json
import os
from argparse import Namespace
from multiprocessing import Pool
from typing import Tuple, Dict, Any, List, Union

import numpy as np
from sklearn.decomposition import PCA

from wormlab3d import logger, PCA_CACHE_PATH, N_WORKERS
from wormlab3d.toolkit.util import hash_data
from wormlab3d.trajectories.cache import get_trajectory


class PCACache:
    def __init__(self, pca_data: Union[np.ndarray, List[PCA]]):
        if type(pca_data) == list:
            data = _map_pcas_to_data(pca_data)
        else:
            data = pca_data

        self.data = data
        N = len(data)
        self.components = data[:, :9].reshape((N, 3, 3))
        self.singular_values = data[:, 9:12]
        self.explained_variance = data[:, 12:15]
        self.explained_variance_ratio = data[:, 15:18]


def _map_pcas_to_data(pcas: List[PCA]) -> np.ndarray:
    data = np.zeros((len(pcas), 3 * 3 + 3 + 3 + 3))
    for i, pca in enumerate(pcas):
        data[i] = np.concatenate([
            pca.components_.flatten(),
            pca.singular_values_,
            pca.explained_variance_,
            pca.explained_variance_ratio_,
        ])

    return data


def calculate_pca(X: np.ndarray, i: int, window_size: int) -> PCA:
    """
    Calculate a PCA basis for a window of a trajectory X centred at index i.
    """
    w2 = int(window_size / 2)
    start_idx = max(0, min(i - w2, len(X) - window_size))
    end_idx = min(len(X), start_idx + window_size)
    window_size_actual = end_idx - start_idx

    # Cast to float64 otherwise PCA fails on very large windows
    if X.dtype == np.float32:
        X = X.astype(np.float64)

    # logger.debug(f'Computing PCA in window [{start_idx}:{end_idx}]')
    X_window = X[start_idx:end_idx]
    if X_window.ndim == 3:
        X_window = X_window.reshape((window_size_actual * X.shape[1], 3))
    assert X_window.ndim == 2
    pca = PCA(svd_solver='full', copy=True, n_components=3)
    pca.fit(X_window)

    return pca


def calculate_pca_wrapper(args) -> float:
    return calculate_pca(*args)


def _calculate_pcas_parallel(X: np.ndarray, window_size: int) -> List[PCA]:
    N = X.shape[0]
    w2 = int(window_size / 2)
    with Pool(processes=N_WORKERS) as pool:
        pcas = pool.map(
            calculate_pca_wrapper,
            [[X, i, window_size] for i in range(w2, N - w2)]
        )
    return pcas


def _calculate_pcas(X: np.ndarray, window_size: int) -> List[PCA]:
    N = X.shape[0]
    w2 = int(window_size / 2)
    pcas = []
    for i in range(w2, N - w2):
        pca = calculate_pca(X, i, window_size)
        pcas.append(pca)
    return pcas


def calculate_pcas(X: np.ndarray, window_size: int, parallel: bool = True) -> List[PCA]:
    assert window_size <= X.shape[0], f'Window size ({window_size}) is larger than trajectory length ({X.shape[0]})!'
    if parallel and N_WORKERS > 1:
        pcas = _calculate_pcas_parallel(X, window_size)
    else:
        pcas = _calculate_pcas(X, window_size)
    return pcas


def generate_pca_cache_data(
        reconstruction_id: str = None,
        trial_id: str = None,
        midline_source: str = None,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        smoothing_window: int = None,
        directionality: str = None,
        prune_slowest_ratio: float = None,
        trajectory_point: str = None,
        window_size: int = 5,
        tracking_only: bool = False,
        rebuild_cache: bool = False
) -> Tuple[PCACache, Dict[str, Any]]:
    X, X_meta = get_trajectory(
        reconstruction_id=reconstruction_id,
        trial_id=trial_id,
        midline_source=midline_source,
        midline_source_file=midline_source_file,
        start_frame=start_frame,
        end_frame=end_frame,
        smoothing_window=smoothing_window,
        directionality=directionality,
        prune_slowest_ratio=prune_slowest_ratio,
        trajectory_point=trajectory_point,
        tracking_only=tracking_only,
        rebuild_cache=rebuild_cache
    )
    pcas = calculate_pcas(X, window_size=window_size)
    pcas_cache = PCACache(pcas)

    return pcas_cache, X_meta


def generate_or_load_pca_cache(
        reconstruction_id: str = None,
        trial_id: str = None,
        midline_source: str = None,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        smoothing_window: int = None,
        directionality: str = None,
        window_size: int = 5,
        trajectory_point: str = None,
        tracking_only: bool = False,
        rebuild_cache: bool = False,
) -> Tuple[PCACache, Dict[str, Any]]:
    """
    Try to load an existing pca cache or generate it otherwise.
    """
    if start_frame is None:
        start_frame = 0
    args = {
        'reconstruction_id': str(reconstruction_id) if reconstruction_id is not None else None,
        'trial_id': trial_id,
        'midline_source': midline_source,
        'midline_source_file': midline_source_file,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'smoothing_window': smoothing_window,
        'directionality': directionality,
        'window_size': window_size,
        'tracking_only': tracking_only,
    }
    if trajectory_point is not None:
        args['trajectory_point'] = trajectory_point
    arg_hash = hash_data(args)
    filename_meta = f'{arg_hash}.meta'
    filename_pca = f'{arg_hash}.npz'
    path_meta = PCA_CACHE_PATH / filename_meta
    path_pca = PCA_CACHE_PATH / filename_pca
    if not rebuild_cache and os.path.exists(path_meta) and os.path.exists(path_pca):
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
            pca_data = np.memmap(path_pca, dtype=np.float32, mode='r', shape=tuple(meta['shape']))
            pca_cache = PCACache(pca_data)
            logger.info(f'Loaded PCA data from {path_pca}.')
            return pca_cache, meta
        except Exception as e:
            logger.warning(f'Could not load PCA data from {path_pca}. {e}')
    elif not rebuild_cache:
        logger.info('PCA file cache unavailable, building.')
    else:
        logger.info('Rebuilding PCA cache.')

    # Generate the pca data cache
    pca_cache, X_meta = generate_pca_cache_data(**args, rebuild_cache=rebuild_cache)
    pca_data = np.memmap(path_pca, dtype=np.float32, mode='w+', shape=pca_cache.data.shape)
    pca_data[:] = pca_cache.data

    # Save the cache onto the hard drive
    logger.info(f'Saving PCA file cache to {path_pca}.')
    pca_data.flush()

    # Save the meta data
    meta = {**X_meta, **args, **{'shape': pca_data.shape}}
    with open(path_meta, 'w') as f:
        json.dump(meta, f)

    return pca_cache, meta


def get_pca_cache_from_args(args: Namespace) -> PCACache:
    """
    Generate or load the pca from parameters set in an argument namespace.
    """
    if hasattr(args, 'planarity_window') and args.planarity_window is not None:
        ws = args.planarity_window
    else:
        ws = args.window_size

    pcas, meta = generate_or_load_pca_cache(
        reconstruction_id=args.reconstruction,
        trial_id=args.trial,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        smoothing_window=args.smoothing_window,
        directionality=args.directionality,
        window_size=ws,
        trajectory_point=args.trajectory_point,
        tracking_only=args.tracking_only,
        rebuild_cache=args.rebuild_cache,
    )
    return pcas


def get_planarity_from_args(args: Namespace) -> np.ndarray:
    """
    Planarity metric is 1 - the relative contribution from the 3rd component.
    """
    pcas = get_pca_cache_from_args(args)
    r = pcas.explained_variance_ratio_
    return 1 - r[2] / np.sqrt(r[1] * r[0])
