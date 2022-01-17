import json
from argparse import Namespace
from typing import Tuple, Any, Dict, Union

import numpy as np
from wormlab3d import logger, DATA_PATH
from wormlab3d.data.model import Midline3D, Frame, Reconstruction, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.trajectories.util import smooth_trajectory, prune_slowest_frames, prune_directionality

TRAJECTORY_CACHE_PATH = DATA_PATH / 'trajectory_cache'
SMOOTHING_WINDOW_TYPES = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def get_trajectory(
        reconstruction_id: str = None,
        trial_id: str = None,
        midline_source: str = None,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        depth: int = -1,
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
        reconstruction_id=reconstruction_id,
        trial_id=trial_id,
        midline_source=midline_source,
        midline_source_file=midline_source_file,
        start_frame=start_frame,
        end_frame=end_frame,
        depth=depth,
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
        reconstruction_id=args.reconstruction,
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


def _generate_trajectory_cache_data(
        reconstruction: Reconstruction,
) -> np.ndarray:
    """
    Load trial midlines from the database and combine across a full trial array container.
    """

    # Filter the midlines by appropriate source
    midline_matches = {'midline.source': reconstruction.source}
    if reconstruction.source_file is not None:
        midline_matches['midline.source_file'] = reconstruction.source_file

    pipeline = [
        {'$match': {'trial': reconstruction.trial.id}},
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
    midline_ids = {}
    for res in cursor:
        # Check frame id is not the same
        frame_id = res['frame_id']
        if len(frame_ids) and frame_id == frame_ids[-1]:
            logger.debug(f'Multiple results returned for frame = {frame_id}. Skipping.')
            continue
        frame_ids.append(frame_id)

        # Save midline ids with their frame numbers
        midline_ids[res['midline_id']] = res['frame_num']

    n_results = len(midline_ids)
    if n_results == 0:
        raise RuntimeError('No results found!')

    logger.info(f'Fetched {n_results} midlines.')

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
        frame_num = midline_ids[m_id]

        # Shape results based on the first midline and hope the rest all fit!
        if Xs is None:
            Xs = np.zeros([reconstruction.trial.n_frames_min, *X.shape])

        Xs[frame_num] = X

    # Check for bad data
    for i, X in enumerate(Xs):
        if np.isnan(X).any():
            logger.warning(f'3D midlines id={m_ids[i]} has a broken midline!')

    return Xs


def _fetch_reconstruction(
        reconstruction_id: str = None,
        trial_id: str = None,
        midline_source: str = None,
        midline_source_file: str = None,
) -> Union[Reconstruction, None]:
    """
    Try to find a reconstruction satisfying arguments.
    """
    reconstruction = None
    if reconstruction_id is not None:
        reconstruction = Reconstruction.objects.get(id=reconstruction_id)
    else:
        # Try to find a suitable reconstruction
        filters = {'trial': trial_id}
        if midline_source is not None:
            filters['source'] = midline_source
        if midline_source_file is not None:
            filters['source_file'] = midline_source_file

        reconstructions = Reconstruction.objects(**filters).order_by('-updated')
        if reconstructions.count() == 0:
            logger.warning(f'Found no reconstructions for parameters {filters}.')
        else:
            logger.info(
                f'Found {reconstructions.count()} matching reconstructions. '
                f'Using most recent.'
            )
            reconstruction = reconstructions[0]

    return reconstruction


def _fetch_mf_trajectory(
        reconstruction: Reconstruction,
        start_frame: int,
        end_frame: int,
        depth: int = None
) -> Tuple[np.ndarray, dict]:
    """
    Fetch a MF trajectory from the corresponding TrialState.
    """
    ts = TrialState(
        reconstruction=reconstruction,
        start_frame=start_frame,
        end_frame=end_frame
    )
    XD = ts.get('points')
    if depth is None:
        X_full = XD
    else:
        D = reconstruction.mf_parameters.depth
        if depth == -1:
            depth = D
        assert depth <= D
        from_idx = sum([2**d for d in range(depth - 1)])
        to_idx = from_idx + 2**(depth - 1)
        X_full = XD[:, from_idx:to_idx]

    X_base = ts.get('points_3d_base')
    X_full = X_full + X_base[:, None, :]
    meta = {'shape': X_full.shape, }

    return X_full, meta


def generate_or_load_trajectory_cache(
        reconstruction_id: str = None,
        trial_id: str = None,
        midline_source: str = None,
        midline_source_file: str = None,
        start_frame: int = None,
        end_frame: int = None,
        depth: int = -1,
        rebuild_cache: bool = False
) -> Tuple[np.ndarray, dict]:
    """
    Try to load an existing trajectory cache or generate it otherwise.
    """
    reconstruction = _fetch_reconstruction(reconstruction_id, trial_id, midline_source, midline_source_file)

    # Get trial
    trial: Trial
    if reconstruction is None:
        logger.warning('No matching reconstruction found, using tracking data.')
        assert trial_id is not None
        trial = Trial.objects.get(id=trial_id)
    else:
        trial = reconstruction.trial

    # Set defaults for frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = trial.n_frames_min

    if reconstruction is not None:
        start_frame = max(start_frame, reconstruction.start_frame)
        end_frame = min(end_frame, reconstruction.end_frame)

    if reconstruction is None:
        # If no reconstruction construct a trajectory from the tracking data
        centres_3d, timestamps = trial.get_tracking_data(fixed=True)
        X_full = centres_3d[:, None, :]
        meta = {'shape': X_full.shape, 'type': 'tracking-only'}

    elif reconstruction.source == M3D_SOURCE_MF:
        X_full, meta = _fetch_mf_trajectory(reconstruction, start_frame, end_frame, depth)

    else:
        # Generate or load a reconstruction trajectory for the reconst or WT3D sources.
        filename_meta = f'{reconstruction.id}.meta'
        filename_X = f'{reconstruction.id}.npz'
        path_meta = TRAJECTORY_CACHE_PATH / filename_meta
        path_X = TRAJECTORY_CACHE_PATH / filename_X
        X_full = None
        if not rebuild_cache and path_meta.exists() and path_X.exists():
            try:
                with open(path_meta, 'r') as f:
                    meta = json.load(f)
                X_full = np.memmap(path_X, dtype=np.float32, mode='r', shape=tuple(meta['shape']))
                logger.info(f'Loaded trajectory data from {path_X}.')
            except Exception as e:
                logger.warning(f'Could not load trajectory from {path_X}. {e}')
        elif not rebuild_cache:
            logger.info('Trajectory file cache unavailable, building.')
        else:
            logger.info('Rebuilding trajectory cache.')

        if X_full is None:
            # Generate the trajectory data cache
            X_data = _generate_trajectory_cache_data(reconstruction)
            X_full = np.memmap(path_X, dtype=np.float32, mode='w+', shape=X_data.shape)
            X_full[:] = X_data

            # Save the cache onto the hard drive
            path_Xs = TRAJECTORY_CACHE_PATH / filename_X
            logger.debug(f'Saving trajectory file cache to {path_Xs}.')
            X_full.flush()

            # Save the meta data
            meta = {'shape': X_full.shape, }
            with open(path_meta, 'w') as f:
                json.dump(meta, f)

    # Take a slice of the full array
    X = X_full[start_frame:end_frame]
    meta['start_frame'] = start_frame
    meta['end_frame'] = end_frame

    return X, meta
