from multiprocessing import Pool

import numpy as np
from matplotlib.axes import Axes

from wormlab3d import N_WORKERS
from wormlab3d.postures.natural_frame import NaturalFrame


def calculate_helicity(X: np.ndarray) -> float:
    """
    Calculate the helicity of a curve.
    """
    NF = NaturalFrame(X)
    return NF.helicity()


def _calculate_helicities_parallel(X: np.ndarray) -> np.ndarray:
    """
    Calculate helicities in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_helicity,
            [Xi for Xi in X]
        )
    return np.array(res)


def calculate_helicities(X: np.ndarray, parallel: bool = True) -> np.ndarray:
    """
    Calculate the helicity metric for all the postures.
    """
    if parallel and N_WORKERS > 1:
        c = _calculate_helicities_parallel(X)
    else:
        c = np.zeros(len(X))
        for i, Xi in enumerate(X):
            c[i] = calculate_helicity(Xi)
    return c


def plot_helicities(
        ax: Axes,
        helicities: np.ndarray,
        xs: np.ndarray = None,
        alpha_max: int = 1,
        n_fade_lines: int = 100
):
    """
    Helper method to plot the helicities on an axes.
    """
    fade_lines_pos = np.linspace(0, helicities.max(), n_fade_lines)
    fade_lines_neg = np.linspace(0, helicities.min(), n_fade_lines)
    if xs is None:
        xs = np.arange(len(helicities))
    for i in range(n_fade_lines):
        ax.fill_between(
            xs,
            np.ones_like(helicities) * fade_lines_pos[i],
            helicities,
            where=helicities > fade_lines_pos[i],
            color='purple',
            alpha=alpha_max / n_fade_lines,
            linewidth=0,
        )
        ax.fill_between(
            xs,
            helicities,
            np.ones_like(helicities) * fade_lines_neg[i],
            where=helicities < fade_lines_neg[i],
            color='green',
            alpha=alpha_max / n_fade_lines,
            linewidth=0,
        )
