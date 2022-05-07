import itertools
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from scipy.stats import rv_continuous
from torch.distributions import LogNormal, Cauchy, Normal

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import FrameArtist
from wormlab3d import logger
from wormlab3d.particles.sdbn_explorer import PARTICLE_PARAMETER_KEYS
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import orthogonalise, normalise
from wormlab3d.trajectories.displacement import calculate_displacements, DISPLACEMENT_AGGREGATION_L2
from wormlab3d.trajectories.pca import PCACache, calculate_pcas
from wormlab3d.trajectories.util import calculate_speeds


def _calculate_transition_rates(states_list_: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the transition rates from a list of telegraph signals.
    """
    M = np.zeros((2, 2))
    for states_ in states_list_:
        states_ = states_.astype(np.int32)
        for (from_state, to_state) in zip(states_[:-1], states_[1:]):
            M[from_state, to_state] += 1
    M /= M.sum(axis=1, keepdims=True)
    return np.array([M[0, 1], M[1, 0]])


def _centre_select(X: np.ndarray, T: int) -> np.ndarray:
    """
    Take the central T portion of an array X.
    """
    assert len(X) >= T
    Xc = X[int(np.ceil((len(X) - T) / 2)):-int(np.floor((len(X) - T) / 2))]
    assert len(Xc) == T
    return Xc


def plot_trajectory_with_frame(
        X: np.ndarray,
        e0: np.ndarray,
        e1: np.ndarray,
        e2: np.ndarray,
        T: int,
        arrow_scale: float = 0.1,
        title: str = None
) -> Figure:
    """
    Draw a trajectory with frame arrows along it.
    """
    F = FrameNumpy(x=X[:T].T, e0=e0[:T].T, e1=e1[:T].T, e2=e2[:T].T)
    fig = plt.figure(figsize=(10, 10))
    if title is not None:
        fig.suptitle(title)
    ax = fig.add_subplot(projection='3d')
    fa = FrameArtist(F, arrow_scale=arrow_scale)
    fa.add_component_vectors(ax)
    fa.add_midline(ax)
    equal_aspect_ratio(ax)
    fig.tight_layout()
    return fig


def plot_displacements_and_states(D: np.ndarray, states: np.ndarray, title: str = None) -> Figure:
    """
    Plot the continuous displacements and discrete states.
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']

    fig, axes = plt.subplots(2, figsize=(12, 8))
    if title is not None:
        fig.suptitle(title)

    ax = axes[0]
    ax.set_title('Displacements')
    for i in range(D.shape[1]):
        ax.plot(D[:, i], c=default_colours[i], alpha=0.5)
        ax.axhline(y=D[:, i].mean(), c=default_colours[i])

    ax = axes[1]
    ax.set_title('States')
    s = ax.imshow(states.T, aspect='auto', interpolation='none')
    fig.colorbar(s)

    fig.tight_layout()
    return fig


def plot_state_parameters(
        state_parameters: Dict[str, Dict[str, Dict[str, Union[str, List[float]]]]],
        state_values: Dict[str, Dict[str, List[float]]],
        title: str = None
) -> Figure:
    """
    Plot the state parameters on top of histograms of the values.
    Show the pdf for both scipy and torch dists.
    """
    state_keys = list(state_parameters.keys())
    fig, axes = plt.subplots(3, len(state_keys), figsize=(22, 12))
    if title is not None:
        fig.suptitle(title)

    for i, state_key in enumerate(state_keys):
        for j, param_key in enumerate(PARTICLE_PARAMETER_KEYS):
            dist_params = state_parameters[state_key][param_key]
            dist_type = dist_params['dist']
            mu, sigma = dist_params['mu'], dist_params['sigma']

            ax = axes[j, i]
            title = ''
            if j == 0:
                title = state_key + '\n'
            title += f'dist={dist_type}\n'

            ax.set_title(title + f'$\mu={mu:.2f}$, $\sigma={sigma:.2f}$')
            if i == 0:
                ax.set_ylabel(param_key)
            else:
                ax.sharex(axes[j, 0])
                ax.sharey(axes[j, 0])
            ax.hist(state_values[state_key][param_key], bins=50, density=True, facecolor='green', alpha=0.75)

            # Plot the scipy distribution
            dist_cls: rv_continuous = getattr(scipy.stats, dist_type)
            dist = dist_cls(*dist_params['params'])
            x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 1000)
            ax.plot(x, dist.pdf(x), linestyle='--', alpha=0.7)

            # Plot the torch distribution
            if dist_type == 'norm':
                tdist = Normal(loc=mu, scale=sigma)
            elif dist_type == 'lognorm':
                tdist = LogNormal(loc=mu, scale=sigma)
            elif dist_type == 'cauchy':
                tdist = Cauchy(loc=mu, scale=sigma)
            tpdf = np.exp(tdist.log_prob(torch.from_numpy(x)))
            ax.plot(x, tpdf, linestyle=':', alpha=0.7, color='red')

    fig.tight_layout()
    return fig


def plot_states(
        ts: np.ndarray,
        states: np.ndarray,
        speeds: np.ndarray,
        planar_angles: np.ndarray,
        nonplanar_angles: np.ndarray,
        title: str = None
) -> Figure:
    """
    Plot the states, speeds and angles.
    """
    fig, axes = plt.subplots(4, figsize=(16, 10), sharex=True)
    if title is not None:
        fig.suptitle(title)

    # State
    ax = axes[0]
    ax.set_title('State')
    for d in range(states.shape[1]):
        ys = states[:, d] + 2 * d
        ax.scatter(ts, ys, s=2, label=f'd={d}')
    ax.legend()

    # Speeds
    ax = axes[1]
    ax.set_title('Speeds')
    ax.plot(ts, speeds)

    # Planar angles
    ax = axes[2]
    ax.set_title('Planar angles')
    ax.plot(ts, planar_angles)

    # Non-planar angles
    ax = axes[3]
    ax.set_title('Non-planar angles')
    ax.plot(ts, nonplanar_angles)

    fig.tight_layout()
    return fig


def plot_2d_trajectory(
        ts: np.ndarray,
        X: np.ndarray,
        title: str = None
) -> Figure:
    """
    Plot the 2D trajectory components.
    """

    # Construct colours
    colours = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    # Create figure
    fig, axes = plt.subplots(3, figsize=(12, 10))
    if title is not None:
        fig.suptitle(title)

    for i in range(3):
        ax = axes[i]
        ax.set_title(['x', 'y', 'z'][i])
        ax.scatter(ts, X[:, i], s=2, c=c)

    fig.tight_layout()
    return fig


def plot_3d_trajectory(
        X: np.ndarray,
        draw_edges: bool = True,
        title: str = None
) -> Figure:
    """
    Plot a 3D trajectory path.
    """
    x, y, z = X.T

    # Construct colours
    colours = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    if title is not None:
        fig.suptitle(title)
    ax = fig.add_subplot(projection='3d')

    # Scatter the vertices
    ax.scatter(x, y, z, c=c, s=10, alpha=0.4, zorder=-1)

    # Draw lines connecting points
    if draw_edges:
        points = X[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
        ax.add_collection(lc)

    # Setup axis
    equal_aspect_ratio(ax)
    fig.tight_layout()
    return fig


def plot_3d_trajectories(
        Xs_sim: np.ndarray,
        Xs_real: List[np.ndarray] = None,
        draw_edges: bool = True,
        title: str = None
) -> Figure:
    """
    Plot multiple 3D trajectories together.
    """
    if Xs_real is None:
        Xs_real = []
        max_t = Xs_sim.shape[1]
    else:
        max_t = max([X.shape[0] for X in Xs_real] + [Xs_sim.shape[1], ])

    # Construct colours
    cmap_sim = plt.get_cmap('autumn')
    cmap_real = plt.get_cmap('winter')
    colours = np.linspace(0, 1, max_t)
    c_sim = [cmap_sim(c_) for c_ in colours]
    c_real = [cmap_real(c_) for c_ in colours]

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    if title is not None:
        fig.suptitle(title)
    ax = fig.add_subplot(projection='3d')

    for i, Xs in enumerate([Xs_sim, Xs_real]):
        for X in Xs:
            T = len(X)
            X -= X.mean(axis=0, keepdims=True)
            x, y, z = X.T

            # Scatter the vertices
            ax.scatter(x, y, z, c=[c_sim, c_real][i][:T], s=10, alpha=0.4, zorder=-1)

            # Draw lines connecting points
            if draw_edges:
                points = X[:, None, :]
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = Line3DCollection(segments, array=colours[:T], cmap=[cmap_sim, cmap_real][i], zorder=-2)
                ax.add_collection(lc)

    # Setup axis
    equal_aspect_ratio(ax)
    fig.tight_layout()
    return fig


def calculate_pe_parameters_for_trajectory(
        X: np.ndarray,
        deltas: List[int],
        pcas: PCACache = None,
        pca_window: int = 25,
        displacement_aggregation: str = DISPLACEMENT_AGGREGATION_L2,
        distributions: Dict[str, str] = None,
        return_additional: bool = False,
):
    """
    Calculate the particle model parameters from a trajectory.
    """
    deltas = sorted(deltas, reverse=True)
    X_com = X.mean(axis=1)
    X_com_centred = X_com - X_com.mean(axis=0, keepdims=True)
    T = len(X) - max(deltas)

    # Calculate speeds
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=False)
    speeds = _centre_select(speeds, T)

    # If the pcas are not provided then calculate them
    if pcas is None:
        pcas = calculate_pcas(X, window_size=pca_window)
        pcas = PCACache(pcas)

    # Calculate e0 (the tangent to the trajectory curve)
    logger.info('Calculating frame components.')
    e0 = normalise(np.gradient(X_com_centred, axis=0))
    e0 = _centre_select(e0, T)

    # e1 is the frame vector pointing out into the principal plane of the curve
    v1 = pcas.components[:, 1].copy()

    # Correct sign flips
    v1dotv1p = np.einsum('bi,bi->b', v1[:-1], v1[1:])
    flip_idxs = (v1dotv1p < 0).nonzero()[0]
    sign = 1
    for i in range(len(v1)):
        if i - 1 in flip_idxs:
            sign = -sign
        v1[i] = sign * v1[i]
    v1 = _centre_select(v1, T)

    # Orthogonalise the pca planar direction vector against the trajectory to get e1
    e1 = normalise(orthogonalise(v1, e0))

    # e2 is the remaining cross product
    e2 = normalise(np.cross(e0, e1))

    # Calculate the rotation angles
    logger.info('Calculating rotation angles.')
    planar_angles = np.zeros(T)
    nonplanar_angles = np.zeros(T)
    for t in range(T - 1):
        prev_frame = np.stack([e0[t], e1[t], e2[t]])
        next_frame = np.stack([e0[t + 1], e1[t + 1], e2[t + 1]])
        R, rmsd = Rotation.align_vectors(next_frame, prev_frame)
        R = R.as_matrix()

        # Decompose rotation matrix R into the axes of A
        A = prev_frame
        rp = Rotation.from_matrix(A.T @ R @ A)
        a2, a1, a0 = rp.as_euler('zyx')
        # xp = A @ Rotation.from_rotvec(a0 * np.array([1, 0, 0])).as_matrix() @ A.T
        # yp = A @ Rotation.from_rotvec(a1 * np.array([0, 1, 0])).as_matrix() @ A.T
        # zp = A @ Rotation.from_rotvec(a2 * np.array([0, 0, 1])).as_matrix() @ A.T
        # assert np.allclose(xp @ yp @ zp, R)
        planar_angles[t] = a1
        nonplanar_angles[t] = a2

    # Calculate the aligned displacements
    displacements = calculate_displacements(X, deltas, displacement_aggregation)
    max_T = len(displacements[min(deltas)])
    D = []
    for delta in deltas:
        offset = int((delta - min(deltas)) / 2)
        if delta == min(deltas):
            d = displacements[delta]
        else:
            d = np.zeros(max_T)
            d[offset:-offset - 1] = displacements[delta]
        D.append(d)
    D = np.array(D).T
    D = _centre_select(D, T)

    # Convert to states
    states = D > D.mean(axis=0, keepdims=True)

    # Calculate the transition rates
    logger.info('Calculating transition rates.')
    transition_rates = []
    for i in range(len(deltas)):
        if i == 0:
            transition_rates.append(_calculate_transition_rates([states[:, 0], ]))
        else:
            rates_i = np.zeros((2,) * (i + 1))

            # Condition on previous deltas
            prev_state_possibilities = list(itertools.product([0, 1], repeat=i))
            for prev_state in prev_state_possibilities:
                cond_state = np.ones(T, dtype=np.bool)
                for j, prev_state_value in enumerate(prev_state):
                    cond_state = np.logical_and(cond_state, states[:, j] == prev_state_value)

                # Add 0's on to ends otherwise peaks are ignored
                cond_state = np.r_[[False, ], cond_state, [False, ]]
                centre_idxs, section_props = find_peaks(cond_state, width=1)

                # Find transitions for all sections
                states_list = []
                for j in range(len(centre_idxs)):
                    start = section_props['left_bases'][j]
                    end = section_props['right_bases'][j] - 1
                    w = section_props['widths'][j]
                    if w < 2:
                        continue
                    states_list.append(states[start:end, i])
                rates_i[prev_state] = _calculate_transition_rates(states_list)
            transition_rates.append(rates_i)

    # Collate the values in each state
    state_values = {}
    state_keys = [''.join(map(str, c)) for c in list(itertools.product([0, 1], repeat=len(deltas)))]
    for state_key in state_keys:
        state_values[state_key] = {
            'speeds': [],
            'planar_angles': [],
            'nonplanar_angles': []
        }
    for t in range(T):
        state_key = ''.join(map(str, states[t].astype(np.uint8).tolist()))
        state_values[state_key]['speeds'].append(speeds[t])
        state_values[state_key]['planar_angles'].append(planar_angles[t])
        state_values[state_key]['nonplanar_angles'].append(nonplanar_angles[t])

    # Calculate the distribution values
    if distributions is None:
        distributions = {
            'speeds': 'lognorm',
            'planar_angles': 'cauchy',
            'nonplanar_angles': 'cauchy',
        }
    state_parameters = {}
    for state_key in state_keys:
        state_parameters[state_key] = {}
        for param_key in PARTICLE_PARAMETER_KEYS:
            dist_type = distributions[param_key]
            dist_cls: rv_continuous = getattr(scipy.stats, dist_type)
            dist_params = dist_cls.fit(state_values[state_key][param_key], floc=0)

            if dist_type == 'norm':
                mu, sigma = dist_params[0], dist_params[1]
            elif dist_type == 'lognorm':
                mu, sigma = np.log(dist_params[2]), dist_params[0]
            elif dist_type == 'cauchy':
                mu, sigma = dist_params[0], dist_params[1]
            else:
                raise RuntimeError(f'Distribution type {dist_type} not supported!')

            state_parameters[state_key][param_key] = {
                'dist': dist_type,
                'mu': float(mu),
                'sigma': float(sigma),
                'params': dist_params
            }

    if return_additional:
        additional = {
            'frame': {
                'X': _centre_select(X_com_centred, T),
                'e0': e0,
                'e1': e1,
                'e2': e2
            },
            'displacements': D,
            'speeds': speeds,
            'planar_angles': planar_angles,
            'nonplanar_angles': nonplanar_angles,
            'states': states,
            'state_values': state_values,
        }

        return transition_rates, state_parameters, additional

    return transition_rates, state_parameters
