from multiprocessing import Pool
from typing import List, Dict, Union

import numpy as np

from wormlab3d import logger, N_WORKERS


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


def calculate_trajectory_angles(
        trajectory: np.ndarray,
        deltas: Union[int, List[int]],
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Calculate the angles for given trajectory and deltas.
    """
    a = {}
    return_dict = True
    if type(deltas) == int or isinstance(deltas, np.int64):
        return_dict = False
        deltas = [deltas]

    L = len(trajectory)
    for delta in deltas:
        n_out = L - 2 * delta
        if n_out < 1:
            logger.info(f'Trajectory too short to calculate angles for delta = {delta}.')
            a[delta] = np.zeros((0,))
            break
        logger.info(f'Calculating angles for delta = {delta}.')
        s = np.zeros(n_out)
        for i in range(L - 2 * delta):
            v1 = trajectory[i + delta] - trajectory[i]
            v2 = trajectory[i + 2 * delta] - trajectory[i + delta]
            angle = calculate_angle(v1, v2)
            s[i] = angle
        a[delta] = s

    if return_dict:
        return a
    else:
        return a[deltas[0]]


def calculate_trajectory_angles_wrapper(args):
    return calculate_trajectory_angles(*args)


def calculate_trajectory_angles_parallel(
        trajectory: np.ndarray,
        deltas: Union[int, List[int]],
):
    """
    Calculate the displacements in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_trajectory_angles_wrapper,
            [[trajectory, delta] for delta in deltas]
        )
    d = {}
    for i, delta in enumerate(deltas):
        d[delta] = res[i]
    return d
