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

        manoeuvres.append({
            'centre_idx': reversal_centre_idx,
            'start_idx': prev_start_idx,
            'end_idx': next_end_idx,
            'X_prev': X_prev,
            'X_next': X_next,
            'pca_prev': pca_prev,
            'pca_next': pca_next,
            'reversal_duration': reversal_props['widths'][i],
            'angle': angle
        })

    return manoeuvres


def get_forward_durations(
        X_full: np.ndarray,
        min_forward_frames: int = 25,
) -> np.ndarray:
    signed_speeds = calculate_speeds(X_full, signed=True)
    _, forward_props = find_peaks(signed_speeds > 0, width=min_forward_frames)
    return forward_props['widths']
