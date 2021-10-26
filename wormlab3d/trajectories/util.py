import json
import os
from typing import Tuple, List, Any, Dict

import numpy as np
from wormlab3d import logger, DATA_PATH
from wormlab3d.data.model import Midline3D
from wormlab3d.toolkit.util import hash_data

TRAJECTORY_CACHE_PATH = DATA_PATH + '/trajectory_cache'
SMOOTHING_WINDOW_TYPES = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def get_trajectory(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        projection: str = None,
        trajectory_point: float = None,
        smoothing_window: int = None,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load the full 3D trajectory and then either take a slice at the required point or use the centre of mass.
    """
    X, meta = generate_or_load_trajectory_cache(
        trial_id=trial_id,
        midline_source=midline_source,
        midline_source_file=midline_source_file,
        start_frame=start_frame,
        end_frame=end_frame,
        rebuild_cache=rebuild_cache
    )
    N = X.shape[1]

    if smoothing_window is not None:
        X = smooth_trajectory(X, window_len=smoothing_window)

    if projection is not None:
        if projection == 'xy':
            X = np.delete(X, 2, 2)
        elif projection == 'yz':
            X = np.delete(X, 0, 2)
        elif projection == 'xz':
            X = np.delete(X, 1, 2)
        elif projection == 'x':
            X = X[:, :, 0]
        elif projection == 'y':
            X = X[:, :, 1]
        elif projection == 'z':
            X = X[:, :, 2]
        else:
            raise RuntimeError(f'Unrecognised projection: {projection}')

    if trajectory_point is not None:
        if trajectory_point == -1:
            X = X.mean(axis=1)
        else:
            u = round(trajectory_point * N)
            if u == N:
                u -= 1
            assert 0 <= u < N, f'Incompatible trajectory point: {u}.'
            X = X[:, u]

    return X, meta


def generate_trajectory_cache_data(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Load trial midlines from the database and stitch together a contiguous block.
    """
    matches = {'source': midline_source}
    if midline_source_file is not None:
        matches['source_file'] = midline_source_file

    # Filter the frames by trial id and (optionally) frame numbers
    frame_matches = {'frame.trial': trial_id}
    if start_frame is not None or end_frame is not None:
        frame_num_constraints = {}
        if start_frame is not None:
            frame_num_constraints['$gte'] = start_frame
        if end_frame is not None:
            frame_num_constraints['$lte'] = end_frame
        frame_matches['frame.frame_num'] = frame_num_constraints

    pipeline = [
        {'$match': matches},
        {'$lookup': {'from': 'frame', 'localField': 'frame', 'foreignField': '_id', 'as': 'frame'}},
        {'$unwind': {'path': '$frame'}},
        {'$match': frame_matches},
        {'$project': {
            '_id': 0,
            'frame_id': '$frame._id',
            'frame_num': '$frame.frame_num',
            'midline_id': '$_id',
            'midline_error': '$error',
        }},
        {'$sort': {'frame_num': 1, 'midline_error': 1}},
    ]
    cursor = Midline3D.objects().aggregate(pipeline)

    # Fetch results
    frame_ids = []
    frame_nums = []
    midline_ids = {}
    i = 0
    for res in cursor:
        # Check that the first frame is the one requested
        if i == 0 and start_frame is not None and res['frame_num'] != start_frame:
            raise RuntimeError(
                f'First frame found ({res["frame_num"]}) not equal to requested start frame {start_frame}.')

        # Check frame id is not the same
        frame_id = res['frame_id']
        if len(frame_ids) and frame_id == frame_ids[-1]:
            logger.debug(f'Multiple results returned for frame = {frame_id}. Skipping.')
            continue
        frame_ids.append(frame_id)

        # Check that the frame number is increasing
        frame_num = res['frame_num']
        if len(frame_nums) and frame_num != frame_nums[-1] + 1:
            logger.debug(f'Breakage in reconstruction sequence, stopping here.')
            break
        frame_nums.append(frame_num)

        # Save midline ids with their position
        midline_ids[res['midline_id']] = i
        i += 1

    # Check that the last frame is the one requested
    if end_frame is not None and res['frame_num'] != end_frame:
        raise RuntimeError(f'Last frame found ({res["frame_num"]}) not equal to requested start frame {end_frame}.')

    n_results = len(midline_ids)
    logger.info(f'Fetched midlines for frames {frame_nums[0]}-{frame_nums[-1]} ({n_results}).')

    # Collate the midline data
    pipeline = [
        {'$match': {'_id': {'$in': list(midline_ids.keys())}}},
        {'$project': {'_id': 1, 'X': 1}}
    ]
    cursor = Midline3D.objects().aggregate(pipeline)

    Xs = None
    m_ids = []
    for res in cursor:
        X = Midline3D.X.to_python(res['X'])
        m_id = res['_id']
        m_ids.append(m_id)
        idx = midline_ids[m_id]

        # Shape results based on the first midline and hope the rest all fit!
        if Xs is None:
            Xs = np.zeros([n_results, *X.shape])

        Xs[idx] = X

    # Check for bad data
    max_result_idx = None
    for i, X in enumerate(Xs):
        if np.isnan(X).any():
            logger.warning(f'3D midlines id={m_ids[i]} has a broken midline!')
            max_result_idx = i
            break

    # If the sequence was broken by bad data then trim it down
    if max_result_idx is not None:
        Xs = Xs[:max_result_idx]
        logger.warning(f'Trimming results to first {max_result_idx}.')

    # Check we have enough data still
    if len(Xs) < 10:
        raise RuntimeError(f'Not enough data - only {len(Xs)} results.')

    return Xs, frame_nums


def generate_or_load_trajectory_cache(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Try to load an existing trajectory cache or generate it otherwise.
    """
    identifiers = [trial_id, midline_source, midline_source_file]
    if start_frame is not None:
        identifiers.append(start_frame)
    if end_frame is not None:
        identifiers.append(end_frame)
    arg_hash = hash_data(identifiers)
    filename_meta = f'{arg_hash}x.meta'
    filename_X = f'{arg_hash}x.npz'
    path_meta = TRAJECTORY_CACHE_PATH + '/' + filename_meta
    path_X = TRAJECTORY_CACHE_PATH + '/' + filename_X
    if not rebuild_cache and os.path.exists(path_meta) and os.path.exists(path_X):
        try:
            with open(path_meta, 'r') as f:
                meta = json.load(f)
            X = np.memmap(path_X, dtype=np.float32, mode='r', shape=tuple(meta['shape']))
            logger.info(f'Loaded data from {path_X}.')
            return X, meta
        except Exception as e:
            logger.warning(f'Could not load from {path_X}. {e}')
    elif not rebuild_cache:
        logger.info('Trajectory file cache unavailable, building.')
    else:
        logger.info('Rebuilding trajectory cache.')

    # Generate the trajectory data cache
    X_data, frame_nums = generate_trajectory_cache_data(
        trial_id, midline_source, midline_source_file, start_frame, end_frame
    )
    X = np.memmap(path_X, dtype=np.float32, mode='w+', shape=X_data.shape)
    X[:] = X_data

    # Save the cache onto the hard drive
    path_Xs = TRAJECTORY_CACHE_PATH + '/' + filename_X
    logger.debug(f'Saving trajectory file cache to {path_Xs}.')
    X.flush()

    meta = {
        'shape': X.shape,
        'start_frame': frame_nums[0],
        'end_frame': frame_nums[-1]
    }

    # Save the meta data
    with open(path_meta, 'w') as f:
        json.dump(meta, f)

    return X, meta


def smooth_trajectory(X: np.ndarray, window_len=5, window_type='flat'):
    """
    Smooth the trajectory data using a window with requested size.
    Adapted from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Extended to smooth along each of the body points and coordinate axes.
    todo: remove for loops!
    """

    assert X.ndim == 3, 'X must have 3 dimensions: [T, N, 3].'
    assert X.shape[0] > window_len, 'Time dimension needs to be bigger than window size.'
    assert window_len > 2, 'Window size must be > 2.'
    assert window_len % 2 == 1, 'Window size must be odd.'
    assert window_type in SMOOTHING_WINDOW_TYPES, f'Window type must be one of {SMOOTHING_WINDOW_TYPES}.'

    X_padded = np.r_[
        X[1:window_len // 2 + 1][::-1],
        X,
        X[-window_len // 2:-1][::-1]
    ]

    if window_type == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window_type)(window_len)

    # Normalise window
    w /= w.sum()

    # Convolve window with trajectory
    X_s = np.zeros_like(X)
    for i in range(3):
        for u in range(X.shape[1]):
            X_s[:, u, i] = np.convolve(w, X_padded[:, u, i], mode='valid')

    return X_s
