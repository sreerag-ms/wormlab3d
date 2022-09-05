from typing import List, Dict, Any

import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from wormlab3d.trajectories.angles import calculate_angle
from wormlab3d.trajectories.util import calculate_speeds


def get_manoeuvres(
        X_full: np.ndarray,
        X_slice: np.ndarray,
        min_reversal_frames: int = 25,
        window_size: int = 500,
        cut_windows_at_manoeuvres: bool = False
) -> List[Dict[str, Any]]:
    """
    Get the manoeuvre sections, the parts prior and subsequent to a reversal.
    """
    signed_speeds = calculate_speeds(X_full, signed=True)
    reversal_centre_idxs, reversal_props = find_peaks(signed_speeds < 0, width=min_reversal_frames)
    manoeuvres = []

    def nonp(pca_):
        r = pca_.explained_variance_ratio_.T
        return r[2] / np.where(r[2] == 0, 1, np.sqrt(r[1] * r[0]))

    # Loop over reversal events
    for i, reversal_centre_idx in enumerate(reversal_centre_idxs):
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

        # Identify the plane of motion during the reversal
        rev_start_idx = reversal_props['left_bases'][i] + 1
        rev_end_idx = reversal_props['right_bases'][i]
        X_rev = X_slice[rev_start_idx:rev_end_idx]
        pca_rev = PCA(svd_solver='full', copy=True, n_components=3)
        pca_rev.fit(X_rev)

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

        # Calculate the angle between planes
        n1 = pca_prev.components_[2]
        n2 = pca_next.components_[2]
        angle = calculate_angle(n1, n2)

        # Calculate the trajectory angles (between first PCA components)
        t1 = pca_prev.components_[0]
        t2 = pca_next.components_[0]
        angle_traj = calculate_angle(t1, t2)

        # Calculate angles between prev->rev, rev->next
        angle_prev_rev_t = calculate_angle(pca_prev.components_[0], pca_rev.components_[0])
        angle_prev_rev_n = calculate_angle(pca_prev.components_[2], pca_rev.components_[2])
        angle_rev_next_t = calculate_angle(pca_rev.components_[0], pca_next.components_[0])
        angle_rev_next_n = calculate_angle(pca_rev.components_[2], pca_next.components_[2])

        manoeuvres.append({
            'centre_idx': reversal_centre_idx,
            'start_idx': prev_start_idx,
            'end_idx': next_end_idx,
            'X_prev': X_prev,
            'X_next': X_next,
            'pca_prev': pca_prev,
            'pca_next': pca_next,
            'reversal_duration': reversal_props['widths'][i],
            'reversal_distance': np.linalg.norm(X_rev[0] - X_rev[-1], axis=-1),
            'angle': angle,
            'angle_prev_rev_t': angle_prev_rev_t,
            'angle_prev_rev_n': angle_prev_rev_n,
            'angle_rev_next_t': angle_rev_next_t,
            'angle_rev_next_n': angle_rev_next_n,
            'angle_prev_next_t': angle_traj,
            'angle_prev_next_n': angle,
            'nonp_prev': nonp(pca_prev),
            'nonp_rev': nonp(pca_rev),
            'nonp_next': nonp(pca_next),
        })

    return manoeuvres


def get_forward_durations(
        X_full: np.ndarray,
        min_forward_frames: int = 25,
) -> np.ndarray:
    signed_speeds = calculate_speeds(X_full, signed=True)
    _, forward_props = find_peaks(signed_speeds > 0, width=min_forward_frames)
    return forward_props['widths']
