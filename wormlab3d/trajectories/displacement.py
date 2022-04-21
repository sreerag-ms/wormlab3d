from multiprocessing import Pool
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from wormlab3d import logger, N_WORKERS
from wormlab3d.trajectories.util import DEFAULT_FPS

DISPLACEMENT_AGGREGATION_L2 = 'l2'
DISPLACEMENT_AGGREGATION_SQUARED_SUM = 'ss'
DISPLACEMENT_AGGREGATION_OPTIONS = [DISPLACEMENT_AGGREGATION_SQUARED_SUM, DISPLACEMENT_AGGREGATION_L2]


def calculate_displacements(
        trajectory: np.ndarray,
        deltas: Union[int, List[int]],
        aggregation: str = DISPLACEMENT_AGGREGATION_L2
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Calculate the squared displacement for given trajectory and deltas.
    """
    assert aggregation in DISPLACEMENT_AGGREGATION_OPTIONS
    d = {}
    return_dict = True
    if type(deltas) == int or isinstance(deltas, np.int64):
        return_dict = False
        deltas = [deltas]

    if trajectory.ndim == 3:
        logger.info('Using center of mass for displacement calculations.')
        trajectory = trajectory.mean(axis=1)

    for delta in deltas:
        logger.info(f'Calculating displacements for delta = {delta}.')
        diff = trajectory[delta:] - trajectory[:-delta]
        if diff.ndim == 1:
            diff = diff[:, None]
        if aggregation == DISPLACEMENT_AGGREGATION_L2:
            displacements = np.linalg.norm(diff, axis=-1)
        elif aggregation == DISPLACEMENT_AGGREGATION_SQUARED_SUM:
            displacements = np.sum(diff**2, axis=-1)
        d[delta] = displacements

    if return_dict:
        return d
    else:
        return d[deltas[0]]


def calculate_displacements_wrapper(args):
    return calculate_displacements(*args)


def calculate_displacements_parallel(
        trajectory: np.ndarray,
        deltas: Union[int, List[int]],
        aggregation: str = DISPLACEMENT_AGGREGATION_L2
):
    """
    Calculate the displacements in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_displacements_wrapper,
            [[trajectory, delta, aggregation] for delta in deltas]
        )
    d = {}
    for i, delta in enumerate(deltas):
        d[delta] = res[i]
    return d


def calculate_displacement_projections(
        trajectory: np.ndarray,
        deltas: Union[int, List[int]]
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Calculate the displacements in x, y, z for given trajectory and deltas.
    """
    d = {}
    return_dict = True
    if type(deltas) == int:
        return_dict = False
        deltas = [deltas]

    for delta in deltas:
        logger.info(f'Calculating displacement for delta = {delta}.')
        N = len(trajectory) - 2 * delta
        displacements = np.zeros((N, 3))
        for i in range(N):
            displacements[i] = trajectory[i] - trajectory[i + delta]
        d[delta] = displacements

    if return_dict:
        return d
    else:
        return d[deltas[0]]


def calculate_transitions_and_dwells(
        displacements: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Given a time series of windowed-displacements, calculate the transition indices
    and dwell times above and below the average displacement.
    """
    if len(displacements) == 0:
        return {
            'on': [],
            'on_durations': [],
            'off': [],
            'off_durations': [],
        }

    avg = displacements.mean()
    states = (displacements > avg).astype(np.int8)
    on_dwells = []
    off_dwells = []
    start_t = 0
    active_state = states[0]
    t = 1
    while t < len(states):
        state_t = states[t]
        if active_state != state_t:
            if active_state == 1:
                on_dwells.append([start_t, t])
            else:
                off_dwells.append([start_t, t])
            start_t = t
            active_state = state_t
        t += 1
    if t != start_t:
        if active_state == 1:
            on_dwells.append([start_t, t])
        else:
            off_dwells.append([start_t, t])

    on_dwell_durations = np.array([
        dwell[1] - dwell[0]
        for dwell in on_dwells
    ])
    off_dwell_durations = np.array([
        dwell[1] - dwell[0]
        for dwell in off_dwells
    ])

    dwells = {
        'on': np.array(on_dwells),
        'on_durations': on_dwell_durations,
        'off': np.array(off_dwells),
        'off_durations': off_dwell_durations,
    }

    return dwells


def calculate_transitions_and_dwells_multiple_deltas(
        displacements: Dict[int, np.ndarray]
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Calculate the transitions and dwell times for multiple deltas.
    """
    logger.info('Calculating transitions and dwells.')
    ret = {}
    for delta, d in displacements.items():
        ret[delta] = calculate_transitions_and_dwells(d)

    return ret


def calculate_msd(
        trajectory: np.ndarray,
        deltas: List[int]
) -> Dict[int, float]:
    """
    Calculate the mean squared displacements.
    """
    msds = {}
    d = calculate_displacements_parallel(trajectory, deltas, aggregation=DISPLACEMENT_AGGREGATION_SQUARED_SUM)
    for delta in deltas:
        msds[delta] = np.mean(d[delta])
    return msds


def plot_displacement_histograms(
        displacements: Dict[int, np.ndarray],
        fps: int = DEFAULT_FPS
) -> Figure:
    """
    Plot histograms of the squared displacement results.
    """
    deltas = list(displacements.keys())
    fig, axes = plt.subplots(len(deltas), figsize=(10, 4 + 2 * len(deltas)), sharex=True, sharey=True)
    for i, delta in enumerate(deltas):
        ax = axes[i]
        ax.hist(displacements[delta], bins=100, density=True, facecolor='green', alpha=0.75)
        ax.set_title(f'$\Delta={delta / fps:.2f}s$')
        ax.set_ylabel('$P(d;\Delta)$')
        ax.set_xlabel('$d=(x(t+\Delta)-x(t))^2$')
    fig.tight_layout()
    return fig


def plot_displacement_projections_histograms(
        displacements: Dict[int, np.ndarray],
        fps: int = DEFAULT_FPS
) -> Figure:
    """
    Plot histograms of the displacements in x, y, z.
    """
    deltas = list(displacements.keys())
    fig, axes = plt.subplots(len(deltas), 3, figsize=(10, 4 + 2 * len(deltas)), sharex=False, sharey=False)
    for i, delta in enumerate(deltas):
        for j in range(3):
            ax = axes[i, j]
            ax.hist(displacements[delta][:, j], bins=100, density=True, facecolor='green', alpha=0.75)
            ax.set_title(f'$\Delta={delta / fps:.2f}s$')
            ax.set_ylabel('$P(d;\Delta)$')
            xyz = ['x', 'y', 'z'][j]
            ax.set_xlabel(f'$d={xyz}(t+\Delta)-{xyz}(t)$')
    fig.tight_layout()
    return fig


def plot_msd(
        msd: np.ndarray,
        fps: int = DEFAULT_FPS,
        title: str = 'MSD'
) -> Figure:
    """
    Plot the mean squared displacement on a log-log scale.
    """
    deltas = np.array(list(msd.keys())) / fps
    msd_vals = np.array(list(msd.values()))
    fig, ax = plt.subplots(1, figsize=(12, 10), sharex=True, sharey=True)
    ax.plot(deltas, msd_vals)
    ax.set_title(title)
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    fig.tight_layout()
    return fig


def plot_msd_multiple(
        msd: Dict[float, Dict[str, np.ndarray]],
        deltas: List[int],
        fps: int = DEFAULT_FPS,
        title: str = 'MSD'
) -> Figure:
    """
    Plot multiple mean squared displacements across different points and projections.
    """
    deltas = deltas / fps
    fig, ax = plt.subplots(1, figsize=(12, 8), sharex=True, sharey=True)
    linestyles = {
        '3D': 'solid',
        'x': 'dotted',
        'y': 'dashed',
        'z': 'dashdot'
    }
    colors = ['red', 'blue', 'green']

    for i, (u, msds_u) in enumerate(msd.items()):
        for p, msds_up in msds_u.items():
            msd_vals = np.array(list(msds_up.values()))
            ax.plot(deltas, msd_vals, label=f'u={u:.1f}, p={p}', linestyle=linestyles[p], color=colors[i], alpha=0.5)

    ax.set_title(title)
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.legend()
    fig.tight_layout()
    return fig
