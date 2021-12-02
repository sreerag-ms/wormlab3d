import json
import os
from argparse import Namespace
from typing import Tuple, List, Any, Dict, Union

import numpy as np
from wormlab3d import logger, DATA_PATH
from wormlab3d.data.model import Midline3D, Frame
from wormlab3d.toolkit.util import hash_data
from wormlab3d.trajectories.util import smooth_trajectory, prune_slowest_frames, prune_directionality

TRAJECTORY_CACHE_PATH = DATA_PATH + '/trajectory_cache'
SMOOTHING_WINDOW_TYPES = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def get_trajectory(
        trial_id: str,
        midline_source: str,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        smoothing_window: int = None,
        directionality: str = None,
        prune_slowest_ratio: float = None,
        projection: str = None,
        trajectory_point: float = None,
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
    meta['frame_nums'] = np.arange(meta['start_frame'], meta['end_frame'] + 1)

    # Convert to float64
    X = X.astype(np.float64)

    if smoothing_window is not None:
        X = smooth_trajectory(X, window_len=smoothing_window)

    if directionality is not None and directionality != 'both':
        X, kept_idxs = prune_directionality(X, directionality=directionality)
        meta['frame_nums'] = meta['frame_nums'][kept_idxs]

    if prune_slowest_ratio is not None:
        X, kept_idxs = prune_slowest_frames(X, cut_ratio=prune_slowest_ratio)
        meta['frame_nums'] = meta['frame_nums'][kept_idxs]

    if projection is not None and projection != '3D':
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

    # Convert frame nums to a list of ints
    meta['frame_nums'] = meta['frame_nums'].tolist()

    return X, meta


def get_trajectory_from_args(args: Namespace, return_meta: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Get the trajectory using parameters from an argument namespace.
    """
    X, meta = get_trajectory(
        trial_id=args.trial,
        midline_source=args.midline3d_source,
        midline_source_file=args.midline3d_source_file,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        smoothing_window=args.smoothing_window,
        directionality=args.directionality,
        prune_slowest_ratio=args.prune_slowest_ratio,
        projection=args.projection,
        trajectory_point=args.trajectory_point,
        rebuild_cache=args.rebuild_cache
    )

    if return_meta:
        return X, meta
    else:
        return X


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

    # Filter the frames by trial id and (optionally) frame numbers
    frame_matches = {'trial': trial_id}
    if start_frame is not None or end_frame is not None:
        frame_num_constraints = {}
        if start_frame is not None:
            frame_num_constraints['$gte'] = start_frame
        if end_frame is not None:
            frame_num_constraints['$lte'] = end_frame
        frame_matches['frame_num'] = frame_num_constraints

    # Filter the midlines by appropriate source
    midline_matches = {'midline.source': midline_source}
    if midline_source_file is not None:
        midline_matches['midline.source_file'] = midline_source_file

    pipeline = [
        {'$match': frame_matches},
        {'$lookup': {'from': 'midline3d', 'localField': '_id', 'foreignField': 'frame', 'as': 'midline'}},
        {'$unwind': {'path': '$midline'}},
        {'$match': midline_matches},
        {'$project': {
            '_id': 0,
            'frame_id': '$_id',
            'frame_num': '$frame_num',
            'midline_id': '$midline._id',
            'midline_error': '$midline.error',
        }},
        {'$sort': {'frame_num': 1, 'midline_error': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)

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
    if n_results == 0:
        raise RuntimeError('No results found!')

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
            logger.info(f'Loaded trajectory data from {path_X}.')
            return X, meta
        except Exception as e:
            logger.warning(f'Could not load trajectory from {path_X}. {e}')
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
