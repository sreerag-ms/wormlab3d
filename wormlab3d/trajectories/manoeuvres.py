from typing import List, Dict, Any

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.util import calculate_speeds


def _nonp(pca: PCA) -> np.ndarray:
    r = pca.explained_variance_ratio_.T
    return r[2] / np.where(r[2] == 0, 1, np.sqrt(r[1] * r[0]))


def align_with_traj(v: np.ndarray, X: np.ndarray) -> np.ndarray:
    va = v
    if np.dot(v, X[-1] - X[0]) < 0:
        va *= -1
    return va


def get_manoeuvres(
        X_full: np.ndarray,
        X_slice: np.ndarray,
        min_reversal_frames: int = 25,
        min_reversal_distance: float = 0.,
        window_size: int = 500,
        cut_windows_at_manoeuvres: bool = False,
        align_with_traj: bool = False
) -> List[Dict[str, Any]]:
    """
    Get the manoeuvre sections, the parts prior and subsequent to a reversal.
    """
    signed_speeds = calculate_speeds(X_full, signed=True)
    reversal_centre_idxs, reversal_props = find_peaks(signed_speeds < 0, width=min_reversal_frames)
    manoeuvres = []

    # Loop over reversal events
    for i, reversal_centre_idx in enumerate(reversal_centre_idxs):
        # Identify the plane of motion during the reversal
        rev_start_idx = reversal_props['left_bases'][i] + 1
        rev_end_idx = reversal_props['right_bases'][i]
        X_rev = X_slice[rev_start_idx:rev_end_idx]

        # Skip if not reversed far enough
        reversal_distance = np.linalg.norm(X_rev[0] - X_rev[-1], axis=-1)
        if reversal_distance < min_reversal_distance:
            continue

        pca_rev = PCA(svd_solver='full', copy=True, n_components=3)
        pca_rev.fit(X_rev)

        # Identify the plane of motion prior to the reversal
        prev_end_idx = reversal_props['left_bases'][i]
        prev_start_idx = max(0, prev_end_idx - window_size)
        if i > 0 and cut_windows_at_manoeuvres:
            prev_start_idx = max(reversal_props['right_bases'][i - 1], prev_start_idx)
        X_prev = X_slice[prev_start_idx:prev_end_idx]
        if X_prev.shape[0] < 10:
            continue
        pca_prev = PCA(svd_solver='full', copy=True, n_components=3)
        pca_prev.fit(X_prev)

        # Identify the plane of motion subsequent to the reversal
        next_start_idx = reversal_props['right_bases'][i] - 1
        next_end_idx = next_start_idx + window_size
        if i < len(reversal_centre_idxs) - 1 and cut_windows_at_manoeuvres:
            next_end_idx = min(reversal_props['left_bases'][i + 1], next_end_idx)
        X_next = X_slice[next_start_idx:next_end_idx]
        if X_next.shape[0] < 10:
            continue
        pca_next = PCA(svd_solver='full', copy=True, n_components=3)
        pca_next.fit(X_next)

        # Get component vectors for each section
        prev_t = pca_prev.components_[0]
        rev_t = pca_rev.components_[0]
        next_t = pca_next.components_[0]
        prev_n = pca_prev.components_[2]
        rev_n = pca_rev.components_[2]
        next_n = pca_next.components_[2]

        # Align component vectors with trajectory
        if align_with_traj:
            prev_t = align_with_traj(prev_t, X_prev)
            rev_t = align_with_traj(rev_t, X_rev)
            next_t = align_with_traj(next_t, X_next)
            prev_n = align_with_traj(prev_n, X_prev)
            rev_n = align_with_traj(rev_n, X_rev)
            next_n = align_with_traj(next_n, X_next)

        # Calculate angles
        angle_prev_rev_t = calculate_angle(prev_t, rev_t)
        angle_prev_rev_n = calculate_angle(prev_n, rev_n)
        angle_rev_next_t = calculate_angle(rev_t, next_t)
        angle_rev_next_n = calculate_angle(rev_n, next_n)
        angle_prev_next_t = calculate_angle(prev_t, next_t)
        angle_prev_next_n = calculate_angle(prev_n, next_n)

        # Calculate the distance, speed and non-planarity for the whole manoeuvre window
        X_window = X_slice[prev_start_idx:next_end_idx]
        pca_all = PCA(svd_solver='full', copy=True, n_components=3)
        pca_all.fit(X_window)
        duration_all = next_end_idx - prev_start_idx
        distance_all = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()
        # distance_all = np.linalg.norm(X_window[0] - X_window[-1], axis=-1)
        # speed_all = distance_all / duration_all
        speed_all = np.abs(signed_speeds[prev_start_idx:next_end_idx]).mean()

        manoeuvres.append({
            'centre_idx': reversal_centre_idx,
            'start_idx': prev_start_idx,
            'end_idx': next_end_idx,
            'X_prev': X_prev,
            'X_rev': X_rev,
            'X_next': X_next,
            'pca_prev': pca_prev,
            'pca_rev': pca_rev,
            'pca_next': pca_next,
            'reversal_duration': reversal_props['widths'][i],
            'reversal_distance': np.linalg.norm(X_rev[0] - X_rev[-1], axis=-1),
            'angle': angle_prev_next_n,
            'angle_prev_rev_t': angle_prev_rev_t,
            'angle_prev_rev_n': angle_prev_rev_n,
            'angle_rev_next_t': angle_rev_next_t,
            'angle_rev_next_n': angle_rev_next_n,
            'angle_prev_next_t': angle_prev_next_t,
            'angle_prev_next_n': angle_prev_next_n,
            'nonp_prev': _nonp(pca_prev),
            'nonp_rev': _nonp(pca_rev),
            'nonp_next': _nonp(pca_next),
            'nonp_all': _nonp(pca_all),
            'duration_all': duration_all,
            'distance_all': distance_all,
            'speed_all': speed_all,
        })

    return manoeuvres


def get_forward_durations(
        X_full: np.ndarray,
        min_forward_frames: int = 25,
) -> np.ndarray:
    signed_speeds = calculate_speeds(X_full, signed=True)
    _, forward_props = find_peaks(signed_speeds > 0, width=min_forward_frames)
    return forward_props['widths']


def get_forward_stats(
        X_full: np.ndarray,
        X_slice: np.ndarray,
        min_forward_frames: int = 25,
        min_speed: float = 0.,
) -> np.ndarray:
    """
    Get some stats on the forward sections between manoeuvres.
    """
    signed_speeds = calculate_speeds(X_full, signed=True)
    fwd_centre_idxs, fwd_props = find_peaks(signed_speeds > min_speed, width=min_forward_frames)

    runs = []

    # Loop over forward sections
    for i, fwd_centre_idx in enumerate(fwd_centre_idxs):
        start_idx = fwd_props['left_bases'][i] + 1
        end_idx = fwd_props['right_bases'][i]

        # Calculate the distance, speed and non-planarity for the whole manoeuvre window
        X_window = X_slice[start_idx:end_idx]
        pca = PCA(svd_solver='full', copy=True, n_components=3)
        pca.fit(X_window)
        duration = end_idx - start_idx
        distance = np.linalg.norm(X_window[1:] - X_window[:-1], axis=-1).sum()
        # distance = np.linalg.norm(X_window[0] - X_window[-1], axis=-1)
        speed = signed_speeds[start_idx:end_idx].mean()  # distance / duration

        runs.append({
            'centre_idx': fwd_centre_idx,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'pca': pca,
            'duration': duration,
            'distance': distance,
            'speed': speed,
            'nonp': _nonp(pca),
        })

    return runs
