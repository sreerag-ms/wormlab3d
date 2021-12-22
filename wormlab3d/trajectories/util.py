from typing import List, Tuple

import numpy as np

from wormlab3d import DATA_PATH
from wormlab3d.data.model import Frame, Tag

TRAJECTORY_CACHE_PATH = DATA_PATH / 'trajectory_cache'
SMOOTHING_WINDOW_TYPES = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']


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


def calculate_speeds(X: np.ndarray, signed: bool = False) -> np.ndarray:
    """
    Calculate the speed using the centre of mass.
    If signed speed requested, use the head-tail vector to determine direction.
    """
    assert X.shape[-1] == 3
    assert X.ndim == 3
    com = X.mean(axis=1)
    directional_gradients = np.gradient(com, axis=0)
    speeds = np.linalg.norm(directional_gradients, axis=1)

    if signed:
        ht_directions = X[:, 0] - X[:, -1]
        ht_dot_dir = np.einsum('ni,ni->n', ht_directions, directional_gradients)
        fwd_or_back = np.sign(ht_dot_dir)
        speeds = speeds * fwd_or_back

    return speeds


def calculate_htd(X: np.ndarray) -> np.ndarray:
    """
    Calculate the distances from the head to the tail.
    """
    htd = np.linalg.norm(X[:, 0] - X[:, -1], axis=1)
    return htd


def prune_directionality(
        X: np.ndarray,
        directionality: str = None,
) -> np.ndarray:
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
) -> np.ndarray:
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
):
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


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the signed angle between two vectors.
    """
    if len(v1) == 2:
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    elif len(v1) == 3:
        abs_val = np.linalg.norm(v1) * np.linalg.norm(v2)
        try:
            cos = np.dot(v1, v2) / abs_val
            angle = np.arccos(cos)
            if np.isnan(angle):
                angle = 0
        except Exception:
            angle = 0
    else:
        raise ValueError('Vectors of the wrong dimension!')

    return angle
