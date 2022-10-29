from argparse import Namespace
from typing import List, Tuple, Union

import numpy as np

from wormlab3d import DATA_PATH, logger
from wormlab3d.data.model import Frame, Tag, Reconstruction

DEFAULT_FPS = 25
TRAJECTORY_CACHE_PATH = DATA_PATH / 'trajectory_cache'
SMOOTHING_WINDOW_TYPES = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


def smooth_trajectory(X: np.ndarray, window_len: int = 5, window_type: str = 'flat') -> np.ndarray:
    """
    Smooth the trajectory data using a window with requested size.
    Adapted from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    Extended to smooth along each of the body points and coordinate axes.
    todo: remove for loops!
    """
    squeeze = False
    if X.ndim == 1:
        X = X[..., None]
        squeeze = True
    if X.ndim == 2:
        X = X[..., None]
        squeeze = True
    if X.ndim != 3:
        raise ValueError('X must have 2 or 3 dimensions.')
    assert X.shape[0] > window_len, 'Time dimension needs to be bigger than window size.'
    assert window_len > 2, 'Window size must be > 2.'
    assert window_len % 2 == 1, 'Window size must be odd.'
    assert window_type in SMOOTHING_WINDOW_TYPES, f'Window type must be one of {SMOOTHING_WINDOW_TYPES}.'

    X_padded = np.r_[
        np.stack([X[0]] * (window_len // 2)),
        X,
        np.stack([X[-1]] * (window_len // 2))
    ]

    if window_type == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window_type)(window_len)

    # Normalise window
    w /= w.sum()

    # Convolve window with trajectory
    X_s = np.zeros_like(X)
    for i in range(X.shape[-1]):
        for u in range(X.shape[1]):
            X_s[:, u, i] = np.convolve(w, X_padded[:, u, i], mode='valid')

    # Remove extra dimension is one was added
    if squeeze:
        X_s = X_s.squeeze(axis=-1)

    return X_s


def calculate_speeds(X: np.ndarray, signed: bool = False) -> np.ndarray:
    """
    Calculate the speed using the centre of mass.
    If signed speed requested, use the head-tail vector to determine direction.
    """
    assert X.shape[-1] == 3
    if X.ndim == 2:
        X = X[:, None, :]
    com = X.mean(axis=1)
    directional_gradients = np.gradient(com, axis=0)
    speeds = np.linalg.norm(directional_gradients, axis=1)

    if signed:
        assert X.shape[1] > 1, 'Not enough information to calculate signed speeds!'
        ht_directions = X[:, 0] - X[:, -1]
        ht_dot_dir = np.einsum('ni,ni->n', ht_directions, directional_gradients)
        fwd_or_back = np.sign(ht_dot_dir)
        speeds = speeds * fwd_or_back

    return speeds


def calculate_htd(X: np.ndarray) -> np.ndarray:
    """
    Calculate the distances from the head to the tail as a proportion of the worm length.
    """
    htd = np.linalg.norm(X[:, 0] - X[:, -1], axis=-1)
    sl = np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1)
    wl = sl.sum(axis=-1)
    htd_rel = htd / wl
    return htd_rel


def prune_directionality(
        X: np.ndarray,
        directionality: str = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Remove forward- or backwards-locomotion frames.
    """
    assert directionality in ['forwards', 'backwards']
    speeds = calculate_speeds(X, signed=True)

    # If directionality is forwards, remove backwards frames
    if directionality == 'forwards':
        frame_idxs_to_cut = (speeds < 0).nonzero()[0]

    # If directionality is backwards, remove forwards frames
    else:
        frame_idxs_to_cut = (speeds > 0).nonzero()[0]

    X_pruned, kept_idxs = _prune_frames(X, frame_idxs_to_cut)

    return X_pruned, kept_idxs


def prune_slowest_frames(
        X: np.ndarray,
        cut_ratio: float = 0.1,
) -> Tuple[np.ndarray, List[int]]:
    """
    Remove slowest speed frames.
    """
    assert 0 < cut_ratio < 1, 'cut_ratio must be between 0 and 1.'
    speeds = calculate_speeds(X)
    N = len(X)
    n_frames_to_cut = int(N * cut_ratio)
    frame_idxs_to_cut = np.argsort(speeds)[:n_frames_to_cut]
    X_pruned, kept_idxs = _prune_frames(X, frame_idxs_to_cut)

    return X_pruned, kept_idxs


def _prune_frames(
        X: np.ndarray,
        frame_idxs_to_cut: np.ndarray
) -> Tuple[np.ndarray, List[int]]:
    """
    Prune the given frame idxs from the trajectory.
    """
    assert X.ndim == 3, 'Pruning requires the full trajectory.'
    N = len(X)
    N_pruned = N - len(frame_idxs_to_cut)
    frame_idxs_to_cut.sort()

    X_pruned = np.zeros((N_pruned, *X.shape[1:]))
    j = 0
    k = -1
    com = X.mean(axis=1)
    com_adj = np.zeros(3)
    kept_idxs = []

    for i in range(N):
        if len(frame_idxs_to_cut) > 0 and i == frame_idxs_to_cut[0]:
            next_cut_idx = frame_idxs_to_cut[0]
            if k == -1:
                k = next_cut_idx - 1 if next_cut_idx > 0 else 0
            frame_idxs_to_cut = frame_idxs_to_cut[1:]
            if (len(frame_idxs_to_cut) == 0 or i + 1 != frame_idxs_to_cut[0]) and i < N - 1:
                com_adj = com[k] - com[i + 1] + com_adj
                k = -1
        else:
            kept_idxs.append(i)
            X_pruned[j] = X[i] + com_adj
            j += 1

    return X_pruned, kept_idxs


def fetch_annotations(trial_id: str, frame_nums: List[int] = None) -> Tuple[List[Tag], List[np.ndarray]]:
    """
    Load frame annotations.
    """
    matches = {'trial': trial_id}
    if frame_nums is not None:
        matches['frame_num'] = {'$in': frame_nums}

    pipeline = [
        {'$match': matches},
        {'$project': {'_id': 0, 'frame_num': 1, 'tags': 1}},
        {'$unwind': '$tags'},
        {'$group': {'_id': '$tags', 'frames': {'$addToSet': '$frame_num'}}},
        {'$sort': {'_id': 1}}
    ]
    data = list(Frame.objects().aggregate(pipeline))

    tags = []
    frame_idxs = []
    for datum in data:
        tags.append(Tag.objects.get(id=datum['_id']))
        tag_frame_nums = np.sort(datum['frames'])
        tag_frame_idxs = np.isin(frame_nums, tag_frame_nums).nonzero()[0]
        frame_idxs.append(tag_frame_idxs)

    return tags, frame_idxs


def calculate_rotation_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Calculate the rotation matrix between two vectors.
    """
    assert len(v1) == 3 and len(v2) == 3, 'Only 3D vectors supported!'
    if np.allclose(v1, v2):
        return np.eye(3)
    a = v1 / np.linalg.norm(v1)
    b = v2 / np.linalg.norm(v2)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    R = np.eye(3) + vx + np.dot(vx, vx) * (1 - c) / s**2
    return R


def fetch_reconstruction(
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
        N = reconstructions.count()
        if N == 0:
            logger.warning(f'Found no reconstructions for parameters {filters}.')
        else:
            if N == 1:
                logger.info('Found 1 matching reconstruction.')
            else:
                logger.info(
                    f'Found {N} matching reconstructions. '
                    f'Using most recent.'
                )
            reconstruction = reconstructions[0]

    return reconstruction


def get_deltas_from_args(args: Namespace, fps: int = DEFAULT_FPS, min_delta: int = None) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Get deltas and delta times from command line arguments.
    """
    # Use exponentially-spaced deltas
    if args.delta_step < 0:
        delta = args.min_delta
        deltas = []
        while delta < args.max_delta:
            deltas.append(delta)
            delta = delta**(-args.delta_step)
        deltas = np.array(deltas).astype(np.int64)

    # Use equally-spaced deltas
    else:
        deltas = np.arange(args.min_delta, args.max_delta, step=int(args.delta_step))

    if min_delta is not None:
        deltas = deltas.clip(min=min_delta)

    delta_ts = deltas / fps

    return deltas, delta_ts


def resample_curve_points(X: np.ndarray, N_new: int) -> np.ndarray:
    """
    Resample the curve to the specified resolution.
    """
    from scipy import interpolate
    is_complex = np.iscomplexobj(X)
    if is_complex:
        X = np.stack([np.real(X), np.imag(X)], axis=-1)
    assert X.ndim == 3 or X.ndim == 2, f'X is the wrong shape! {X.shape}'
    squeeze_dim0 = False
    if X.ndim == 2:
        X = X[None, ...]
        squeeze_dim0 = True
    if X.shape[1] == N_new:
        return X

    X_new = np.zeros((X.shape[0], N_new, X.shape[-1]))
    sl = np.linalg.norm(X[:, :-1] - X[:, 1:], axis=-1)
    u = np.c_[np.zeros((X.shape[0], 1)), sl.cumsum(axis=-1)]
    u = u / u[:, -1][:, None]
    u_new = np.linspace(0, 1, N_new)

    for i, Xi in enumerate(X):
        for j in range(X.shape[-1]):
            tck = interpolate.splrep(u[i], Xi[:, j], s=0, k=3)
            X_new[i, :, j] = interpolate.splev(u_new, tck)

    if squeeze_dim0:
        X_new = X_new[0]
    if is_complex:
        X_new = X_new[..., 0] + 1.j * X_new[..., 1]

    return X_new
