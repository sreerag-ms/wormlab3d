import json
import os
from argparse import Namespace
from multiprocessing import Pool
from typing import Tuple, Dict, Any

import numpy as np
from simple_worm.frame import FrameSequenceNumpy
from simple_worm.material_parameters import MP_DEFAULT_K
from simple_worm.util import estimate_K_from_x
from wormlab3d import logger, DATA_PATH, N_WORKERS
from wormlab3d.toolkit.util import hash_data
from wormlab3d.trajectories.cache import get_trajectory

K_ESTIMATES_CACHE_PATH = DATA_PATH + '/K_estimates_cache'


def est_K(X: np.ndarray, i: int, n_sample_frames: int, K0: float) -> float:
    X_sample = X[i:i + n_sample_frames]
    psi = np.zeros(X_sample.shape[:-1])
    FS = FrameSequenceNumpy(x=X_sample.transpose(0, 2, 1), psi=psi, calculate_components=True)
    K_est = estimate_K_from_x(FS, K0, verbosity=0)
    logger.info(f'Frame #{i}: {K_est:.3E}')
    return K_est


def est_K_wrapper(args) -> float:
    return est_K(*args)


def _calculate_estimates_parallel(X: np.ndarray, n_sample_frames: int, K0: float) -> np.ndarray:
    end_frame = X.shape[0] - n_sample_frames

    with Pool(processes=N_WORKERS) as pool:
        K_ests = pool.map(
            est_K_wrapper,
            [[X, i, n_sample_frames, K0] for i in range(end_frame)]
        )
    K_ests = np.array(K_ests)

    return K_ests


def _calculate_estimates(X: np.ndarray, n_sample_frames: int, K0: float) -> np.ndarray:
    K_ests = []
    end_frame = X.shape[0] - n_sample_frames

    i = 0
    while i < end_frame:
        K0 = K0 if len(K_ests) == 0 else K_ests[-1]
        K_est = est_K(X, i, n_sample_frames, K0)
        K_ests.append(K_est)
        i += 1
    K_ests = np.array(K_ests)

    return K_ests


def calculate_K_estimates(X: np.ndarray, n_sample_frames: int, K0: float) -> np.ndarray:
    if N_WORKERS > 1:
        K_ests = _calculate_estimates_parallel(X, n_sample_frames, K0)
    else:
        K_ests = _calculate_estimates(X, n_sample_frames, K0)
    return K_ests


def generate_K_estimates_cache_data(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        smoothing_window: int = None,
        directionality: str = None,
        n_sample_frames: int = 5,
        K0: float = MP_DEFAULT_K,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    X, X_meta = get_trajectory(
        trial_id=trial_id,
        midline_source=midline_source,
        midline_source_file=midline_source_file,
        start_frame=start_frame,
        end_frame=end_frame,
        smoothing_window=smoothing_window,
        directionality=directionality,
        rebuild_cache=rebuild_cache
    )

    K_ests = calculate_K_estimates(
        X,
        n_sample_frames=n_sample_frames,
        K0=K0
    )

    return K_ests, X_meta


def generate_or_load_K_estimates_cache(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        smoothing_window: int = None,
        directionality: str = None,
        n_sample_frames: int = 5,
        K0: float = MP_DEFAULT_K,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Try to load an existing K estimate cache or generate it otherwise.
    """
    if start_frame is None:
        start_frame = 0
    args = {
        'trial_id': trial_id,
        'midline_source': midline_source,
        'midline_source_file': midline_source_file,
        'start_frame': start_frame,
        'end_frame': end_frame,
        'smoothing_window': smoothing_window,
        'n_sample_frames': n_sample_frames,
        'K0': K0
    }
    if directionality is not None:
        args['directionality'] = directionality
    arg_hash = hash_data(args)
    filename_meta = f'{arg_hash}K.meta'
    filename_K = f'{arg_hash}K.npz'
    path_meta = K_ESTIMATES_CACHE_PATH + '/' + filename_meta
    path_K = K_ESTIMATES_CACHE_PATH + '/' + filename_K
    if not rebuild_cache and os.path.exists(path_meta) and os.path.exists(path_K):
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
            K_ests = np.memmap(path_K, dtype=np.float32, mode='r', shape=tuple(meta['shape']))
            logger.info(f'Loaded data from {path_K}.')
            return K_ests, meta
        except Exception as e:
            logger.warning(f'Could not load from {path_K}. {e}')
    elif not rebuild_cache:
        logger.info('K estimates file cache unavailable, building.')
    else:
        logger.info('Rebuilding K estimates cache.')

    # Generate the K estimates data cache
    K_data, X_meta = generate_K_estimates_cache_data(**args, rebuild_cache=rebuild_cache)
    K_ests = np.memmap(path_K, dtype=np.float32, mode='w+', shape=K_data.shape)
    K_ests[:] = K_data

    # Save the cache onto the hard drive
    logger.debug(f'Saving K estimates file cache to {path_K}.')
    K_ests.flush()

    # Save the meta data
    meta = {**X_meta, **args, **{'shape': K_ests.shape}}
    with open(path_meta, 'w') as f:
        json.dump(meta, f)

    return K_ests, meta


def get_K_estimates_from_args(args: Namespace) -> np.ndarray:
    """
    Generate or load the K estimates from parameters set in an argument namespace.
    """
    K_ests, meta = generate_or_load_K_estimates_cache(
        trial_id=args.trial,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        smoothing_window=args.smoothing_window,
        directionality=args.directionality,
        n_sample_frames=args.K_sample_frames,
        K0=args.K0,
        rebuild_cache=args.rebuild_cache
    )
    return K_ests
