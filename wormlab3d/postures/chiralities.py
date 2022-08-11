from multiprocessing import Pool

import numpy as np
from matplotlib.axes import Axes

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


def plot_chiralities(
        ax: Axes,
        chiralities: np.ndarray,
        xs: np.ndarray = None,
        alpha_max: int = 1,
        n_fade_lines: int = 100
):
    """
    Helper method to plot the chiralities on an axes.
    """
    fade_lines_pos = np.linspace(0, chiralities.max(), n_fade_lines)
    fade_lines_neg = np.linspace(0, chiralities.min(), n_fade_lines)
    if xs is None:
        xs = np.arange(len(chiralities))
    for i in range(n_fade_lines):
        ax.fill_between(
            xs,
            np.ones_like(chiralities) * fade_lines_pos[i],
            chiralities,
            where=chiralities > fade_lines_pos[i],
            color='purple',
            alpha=alpha_max / n_fade_lines,
            linewidth=0,
        )
        ax.fill_between(
            xs,
            chiralities,
            np.ones_like(chiralities) * fade_lines_neg[i],
            where=chiralities < fade_lines_neg[i],
            color='green',
            alpha=alpha_max / n_fade_lines,
            linewidth=0,
        )
