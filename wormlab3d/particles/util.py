from argparse import Namespace
from typing import List, Dict, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from progress.bar import Bar
from scipy.spatial.transform import Rotation
from scipy.stats import rv_continuous
from torch.distributions import LogNormal, Cauchy, Normal

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import FrameArtist
from wormlab3d import logger
from wormlab3d.particles.sdbn_explorer import PARTICLE_PARAMETER_KEYS
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import normalise, orthogonalise
from wormlab3d.trajectories.pca import PCACache, calculate_pcas
from wormlab3d.trajectories.util import get_deltas_from_args, calculate_speeds


def centre_select(X: np.ndarray, T: int) -> np.ndarray:
    """
    Take the central T portion of an array X.
    """
    assert len(X) >= T
    if len(X) == T:
        return X
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
    if states.ndim == 2:
        for d in range(states.shape[1]):
            ys = states[:, d] + 2 * d
            ax.scatter(ts, ys, s=2, label=f'd={d}')
    else:
        for i in range(int(max(states)) + 1):
            ax.scatter(ts[states == i], states[states == i], s=2, label=f's={i}')
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
    ax = fig.add_subplot(projection='3d')
    if title is not None:
        ax.set_title(title)

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


def plot_msd(
        args: Namespace,
        Xs_real: List[np.ndarray],
        Xs_sim: List[torch.Tensor],
) -> Figure:
    """
    Make an MSD plot of the simulated results against real trajectories.
    """
    deltas, delta_ts = get_deltas_from_args(args)

    if type(Xs_real) == np.ndarray:
        Xs_real = [Xs_real, ]

    # Use centre of mass for real trajectories
    tmp = []
    for X_real in Xs_real:
        if X_real.ndim == 3:
            X_real = X_real.mean(axis=1)
        tmp.append(X_real)
    Xs_real = tmp

    rs = len(Xs_real)
    bs = len(Xs_sim)
    msds_all_real = {}
    msds_real = {i: {} for i in range(rs)}
    msds_all_sim = {}
    msds_sim = {i: {} for i in range(bs)}

    logger.info(f'Calculating displacements.')
    bar = Bar('Calculating', max=len(deltas))
    bar.check_tty = False
    for delta in deltas:

        # Calculate MSD for real trajectories
        if rs > 0:
            d_all = []
            for i, X_real in enumerate(Xs_real):
                if delta > X_real.shape[0] / 3:
                    continue
                d = np.sum((X_real[delta:] - X_real[:-delta])**2, axis=-1)
                d_all.append(d)
                msds_real[i][delta] = d.mean()
            if len(d_all):
                msds_all_real[delta] = np.concatenate(d_all).mean()

        # Calculate MSD for simulated trajectories
        if bs > 0:
            d_all = []
            for i, X_sim in enumerate(Xs_sim):
                if delta > X_sim.shape[0] / 3:
                    continue
                d = torch.sum((X_sim[delta:] - X_sim[:-delta])**2, dim=-1)
                d_all.append(d)
                msds_sim[i][delta] = d.mean()
            if len(d_all):
                msds_all_sim[delta] = np.concatenate(d_all).mean()

            # d = torch.sum((Xs_sim[:, delta:] - Xs_sim[:, :-delta])**2, dim=-1)
            # msds_all_sim[delta] = d.mean()
            # for i in range(bs):
            #     msds_sim[i][delta] = d[i].mean()

        bar.next()
    bar.finish()

    # Set up plots and colours
    logger.info('Plotting MSD.')
    fig, ax = plt.subplots(1, figsize=(12, 10))
    cmap = plt.get_cmap('winter')
    colours_real = cmap(np.linspace(0, 1, rs))
    cmap = plt.get_cmap('autumn')
    colours_sim = cmap(np.linspace(0, 1, bs))

    if rs > 0:
        # Plot MSD for each real trajectory
        for i, (idx, msd_vals_real) in enumerate(msds_real.items()):
            msd_vals = np.array(list(msd_vals_real.values()))
            ax.plot(delta_ts[:len(msd_vals)], msd_vals, alpha=0.5, c=colours_real[i])

        # Plot average of the real MSDs
        msd_vals_all_real = np.array(list(msds_all_real.values()))
        ax.plot(delta_ts[:len(msd_vals_all_real)], msd_vals_all_real, label='Trajectory average',
                alpha=0.8, c='black', linestyle='--', linewidth=3, zorder=60)

    if bs > 0:
        # Plot MSD for each simulation
        for i, (idx, msd_vals_sim) in enumerate(msds_sim.items()):
            msd_vals = np.array(list(msd_vals_sim.values()))
            ax.plot(delta_ts[:len(msd_vals)], msd_vals, alpha=0.5, c=colours_sim[i])

        # Plot average of the simulation MSDs
        msd_vals_all_sim = np.array(list(msds_all_sim.values()))
        ax.plot(delta_ts[:len(msd_vals_all_sim)], msd_vals_all_sim, label='Simulation average',
                alpha=0.9, c='black', linestyle=':', linewidth=3, zorder=80)

    # Complete MSD plot
    ax.set_ylabel('MSD$=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta\ (s)$')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()
    ax.legend()

    fig.tight_layout()

    return fig


def calculate_trajectory_frame(
        X: np.ndarray,
        pcas: PCACache = None,
        pca_window: int = 25,
):
    """
    Calculate the frame for a trajectory.
    """
    X_com = X.mean(axis=1)
    X_com_centred = X_com - X_com.mean(axis=0, keepdims=True)
    T = len(X)

    # Calculate speeds
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=X.shape[1] > 1)
    speeds = centre_select(speeds, T)

    # If the pcas are not provided then calculate them
    if pcas is None:
        pcas = calculate_pcas(X, window_size=pca_window)
        pcas = PCACache(pcas)

    # Pad the PCAs if shorter than the sequence
    components = pcas.components.copy()
    if components.shape[0] < T:
        diff = T - components.shape[0]
        components = np.concatenate([
            np.repeat(components[0][None, ...], repeats=np.ceil(diff / 2), axis=0),
            components,
            np.repeat(components[-1][None, ...], repeats=np.floor(diff / 2), axis=0)
        ], axis=0)

    # Calculate e0 (the tangent to the trajectory curve)
    logger.info('Calculating frame components.')
    e0 = normalise(np.gradient(X_com_centred, axis=0))
    e0 = centre_select(e0, T)

    # Flip e0 where the speed is negative
    e0[speeds < 0] *= -1

    # e1 is the frame vector pointing out into the principal plane of the curve
    v1 = components[:, 1].copy()

    # Correct sign flips
    v1dotv1p = np.einsum('bi,bi->b', v1[:-1], v1[1:])
    flip_idxs = (v1dotv1p < 0).nonzero()[0]
    sign = 1
    for i in range(len(v1)):
        if i - 1 in flip_idxs:
            sign = -sign
        v1[i] = sign * v1[i]
    v1 = centre_select(v1, T)

    # Orthogonalise the pca planar direction vector against the trajectory to get e1
    e1 = normalise(orthogonalise(v1, e0))

    # e2 is the remaining cross product
    e2 = normalise(np.cross(e0, e1))

    return e0, e1, e2


def calculate_rotation_angles(e0: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rotation angles to trace out the curve defined by the frame e0/e1/e2.
    """
    T = len(e0)
    assert len(e0) == len(e1) == len(e2)

    logger.info('Calculating rotation angles.')
    planar_angles = np.zeros(T)
    nonplanar_angles = np.zeros(T)
    for t in range(T - 1):
        prev_frame = np.stack([e0[t], e1[t], e2[t]])
        next_frame = np.stack([e0[t + 1], e1[t + 1], e2[t + 1]])
        R, rmsd = Rotation.align_vectors(prev_frame, next_frame)
        R = R.as_matrix()

        # Decompose rotation matrix R into the axes of A
        A = prev_frame
        rp = Rotation.from_matrix(A.T @ R @ A)
        a2, a1, a0 = rp.as_euler('zyx')
        # xp = A @ Rotation.from_rotvec(a0 * np.array([1, 0, 0])).as_matrix() @ A.T
        # yp = A @ Rotation.from_rotvec(a1 * np.array([0, 1, 0])).as_matrix() @ A.T
        # zp = A @ Rotation.from_rotvec(a2 * np.array([0, 0, 1])).as_matrix() @ A.T
        # assert np.allclose(xp @ yp @ zp, R)
        planar_angles[t] = a2
        nonplanar_angles[t] = a1

    return planar_angles, nonplanar_angles
