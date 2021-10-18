from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from wormlab3d import logger

DEFAULT_FPS = 25


def calculate_squared_displacements(
        trajectory: np.ndarray,
        deltas: Union[int, List[int]]
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """
    Calculate the squared displacement for given trajectory and deltas.
    """
    d = {}
    return_dict = True
    if type(deltas) == int:
        return_dict = False
        deltas = [deltas]

    for delta in deltas:
        logger.info(f'Calculating displacements for delta = {delta}.')
        N = len(trajectory) - 2 * delta
        displacements = np.zeros(N)
        for i in range(N):
            # displacements[i] = np.sum(trajectory[i + delta] - trajectory[i])**2
            displacements[i] = np.linalg.norm(trajectory[i + delta] - trajectory[i])
        rel_displacements = displacements / displacements.mean()
        d[delta] = rel_displacements

    if return_dict:
        return d
    else:
        return d[deltas[0]]


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
        logger.info(f'Calculating angles for delta = {delta}.')
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
    ret = {}
    for delta, d in displacements.items():
        ret[delta] = calculate_transitions_and_dwells(d)

    return ret


def plot_squared_displacement_histograms(
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


def plot_squared_displacements_over_time(
        displacements: Dict[int, np.ndarray],
        dwells: Dict[int, dict],
        fps: int = DEFAULT_FPS
):
    """
    Plot time trace of the results.
    """
    deltas = list(displacements.keys())
    fig, axes = plt.subplots(len(deltas), 2, figsize=(12, 4 + 2 * len(deltas)))
    for i, delta in enumerate(deltas):
        d = displacements[delta]

        # Trace over time
        ax = axes[i, 0]
        ax.plot(d, alpha=0.75)
        ax.set_title(f'$\Delta={delta / fps:.2f}s$')
        ax.set_ylabel('$|x(t+\Delta)-x(t)|$')
        ax.set_xlabel('$t$')

        # Add average indicator
        avg = d.mean()
        ax.axhline(y=avg, color='red')

        # Highlight regions where above/below average
        for on_dwell in dwells[delta]['on']:
            ax.fill_between(np.arange(on_dwell[0], on_dwell[1]), max(d), color='green', alpha=0.3, zorder=-1,
                            linewidth=0)
        for off_dwell in dwells[delta]['off']:
            ax.fill_between(np.arange(off_dwell[0], off_dwell[1]), max(d), color='orange', alpha=0.3, zorder=-1,
                            linewidth=0)

        # Plot histogram of dwell times
        ax = axes[i, 1]
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.hist(dwells[delta]['on_durations'], bins=50, density=True, alpha=0.5, color='green')
        ax.hist(dwells[delta]['off_durations'], bins=50, density=True, alpha=0.5, color='orange')

    fig.tight_layout()
    return fig
