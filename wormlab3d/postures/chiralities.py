from multiprocessing import Pool

import numpy as np

from wormlab3d import N_WORKERS
from wormlab3d.postures.natural_frame import NaturalFrame


def calculate_chirality(X: np.ndarray) -> float:
    """
    Calculate the chirality of a curve.
    """
    NF = NaturalFrame(X)
    return NF.chirality()


def _calculate_chiralities_parallel(X: np.ndarray) -> np.ndarray:
    """
    Calculate chiralities in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_chirality,
            [Xi for Xi in X]
        )
    return np.array(res)


def calculate_chiralities(X: np.ndarray, parallel: bool = True) -> np.ndarray:
    """
    Calculate the chirality metric for all the postures.
    """
    if parallel and N_WORKERS > 1:
        c = _calculate_chiralities_parallel(X)
    else:
        c = np.zeros(len(X))
        for i, Xi in enumerate(X):
            c[i] = calculate_chirality(Xi)
    return c
