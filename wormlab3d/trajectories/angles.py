from multiprocessing import Pool
from typing import List, Dict, Union

import numpy as np

from wormlab3d import logger, N_WORKERS
from wormlab3d.trajectories.pca import calculate_pcas


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the signed angle between two vectors.
    """
    if len(v1) == 2:
        angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    elif len(v1) == 3:
        abs_val = np.linalg.norm(v1) * np.linalg.norm(v2)
        try:
            cos = np.clip(np.dot(v1, v2) / abs_val, a_max=1, a_min=-1)
            angle = np.arccos(cos)
            if np.isnan(angle):
                angle = 0
        except Exception:
            angle = 0
    else:
        raise ValueError('Vectors of the wrong dimension!')

    return angle


def calculate_trajectory_angles(
        X: np.ndarray,
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

    L = len(X)
    for delta in deltas:
        n_out = L - 2 * delta
        if n_out < 1:
            logger.info(f'Trajectory too short to calculate angles for delta = {delta}.')
            a[delta] = np.zeros((0,))
            break
        logger.info(f'Calculating angles for delta = {delta}.')
        s = np.zeros(n_out)
        for i in range(L - 2 * delta):
            v1 = X[i + delta] - X[i]
            v2 = X[i + 2 * delta] - X[i + delta]
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
        X: np.ndarray,
        deltas: Union[int, List[int]],
):
    """
    Calculate the trajectory angles in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_trajectory_angles_wrapper,
            [[X, delta] for delta in deltas]
        )
    d = {}
    for i, delta in enumerate(deltas):
        d[delta] = res[i]
    return d


def calculate_planar_angles(
        X: np.ndarray,
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

    L = len(X)
    for delta in deltas:
        n_out = L - 2 * delta
        if n_out < 1:
            logger.info(f'Trajectory too short to calculate planar angles for delta = {delta}.')
            a[delta] = np.zeros((0,))
            break
        logger.info(f'Calculating angles for delta = {delta}.')
        pcas = calculate_pcas(X, window_size=delta, parallel=False)
        s = np.zeros(n_out)
        for i in range(L - 2 * delta):
            v1 = pcas[i].components_[2]
            v2 = pcas[i + delta].components_[2]
            angle = calculate_angle(v1, v2)
            s[i] = angle
        a[delta] = s

    if return_dict:
        return a
    else:
        return a[deltas[0]]


def calculate_planar_angles_wrapper(args):
    return calculate_planar_angles(*args)


def calculate_planar_angles_parallel(
        X: np.ndarray,
        deltas: Union[int, List[int]],
):
    """
    Calculate the planar angles in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_planar_angles_wrapper,
            [[X, delta] for delta in deltas]
        )
    d = {}
    for i, delta in enumerate(deltas):
        d[delta] = res[i]
    return d
