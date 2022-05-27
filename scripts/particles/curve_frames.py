import os
from argparse import Namespace
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from simple_worm.controls import ControlsNumpy
from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import FrameArtist, MidpointNormalize
from wormlab3d import START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Trial
from wormlab3d.particles.tumble_run import find_approximation
from wormlab3d.particles.util import calculate_trajectory_frame, centre_select
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import normalise
from wormlab3d.trajectories.angles import calculate_trajectory_angles, calculate_planar_angles
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import PCACache, calculate_pcas
from wormlab3d.trajectories.util import smooth_trajectory, calculate_speeds

show_plots = False
save_plots = True
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['trial', 'frames', 'src', 'smoothing_window', 'smoothing_window_curvature']:
        if k in excludes:
            continue
        if k == 'trial':
            fn += f'_trial={args.trial}'
        elif k == 'frames':
            frames_str_fn = ''
            if args.start_frame is not None or args.end_frame is not None:
                start_frame = args.start_frame if args.start_frame is not None else 0
                end_frame = args.end_frame if args.end_frame is not None else -1
                frames_str_fn = f'_f={start_frame}-{end_frame}'
            fn += frames_str_fn
        elif k == 'src':
            if args.tracking_only:
                fn += f'_tracking'
            else:
                fn += f'_{args.midline3d_source}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'smoothing_window_curvature' and args.smoothing_window_curvature is not None:
            fn += f'_swc={args.smoothing_window_curvature}'

    return LOGS_PATH / (fn + '.' + img_extension)


def calculate_frenet_frame_components(
        X: np.ndarray,
        smooth_K: 101
):
    """
    Calculate frenet frame components.
    """

    # Use centre of mass of trajectory and centre whole trajectory around the origin
    X_com = X.mean(axis=1)
    X = X_com - X_com.mean(axis=0, keepdims=True)

    # Calculate the tangent to the trajectory curve
    T = np.gradient(X, axis=0)

    # Calculate the speeds
    speeds = np.linalg.norm(T, axis=-1)

    # Normalise and smooth the tangent
    T = smooth_trajectory(T, window_len=smooth_K)
    T = normalise(T)

    # Find sections where the speed is too low
    stationary_speed_threshold = 0.0015
    cutting = False
    for i in range(1, len(T)):
        if not cutting and speeds[i] < stationary_speed_threshold:
            cutting = True
            cut_start = i
            T_start = T[i - 1]
        elif cutting and speeds[i] > stationary_speed_threshold:
            cutting = False
            y = np.linspace(0, 1, i - cut_start + 2)[1:-1, None]
            T[cut_start:i] = (1 - y) * T_start[None, :] + y * T[i][None, :]
        elif cutting and i == len(T) - 1:
            T[cut_start:] = T_start

    # Normalise and smooth the tangent again
    T = smooth_trajectory(T, window_len=25)
    T = normalise(T)

    # Calculate vector curvature
    K = np.gradient(T, 1 / (len(T) - 1), axis=0, edge_order=1)
    K = smooth_trajectory(K, window_len=smooth_K)

    # Curvature magnitude
    kappa = np.linalg.norm(K, axis=-1)

    # Find sections where the curvature is too low
    curvature_threshold = 0.1
    cutting = False
    for i in range(1, len(K)):
        if not cutting and kappa[i] < curvature_threshold:
            cutting = True
            cut_start = i
            K_start = K[i - 1]
        elif cutting and kappa[i] > curvature_threshold:
            cutting = False
            y = np.linspace(0, 1, i - cut_start + 2)[1:-1, None]
            K[cut_start:i] = (1 - y) * K_start[None, :] + y * K[i][None, :]
        elif cutting and i == len(K) - 1:
            K[cut_start:] = K_start

    # Principal normal
    N = normalise(smooth_trajectory(K, window_len=25))

    # Binormal
    B = normalise(np.cross(T, N))

    # Torsion
    tau = np.einsum('bi,bi->b', -N, np.gradient(B, axis=0))

    return {
        'kappa': kappa,
        'tau': tau,
        'T': T,
        'N': N,
        'B': B,
    }


def calculate_bishop_frame_components(
        X: np.ndarray,
        smooth_K: 101
) -> NaturalFrame:
    """
    Calculate bishop frame components.
    """
    speeds = calculate_speeds(X)

    # Use centre of mass of trajectory and centre whole trajectory around the origin
    if X.ndim == 2:
        X = X[:, None, :]
    X_com = X.mean(axis=1)
    X = X_com - X_com.mean(axis=0, keepdims=True)

    # Find sections where the speed is too low and smoothly move the trajectory
    stationary_speed_threshold = 0.0015
    cutting = False
    for i in range(1, len(X)):
        if not cutting and speeds[i] < stationary_speed_threshold:
            cutting = True
            cut_start = i
            X_start = X[i - 1]
        elif cutting and speeds[i] > stationary_speed_threshold or i == len(X) - 1:
            cutting = False
            y = np.linspace(0, 1, i - cut_start + 2)[1:-1, None]
            X[cut_start:i] = (1 - y) * X_start[None, :] + y * X[i][None, :]

    nf = NaturalFrame(X)

    m1m2 = np.stack([nf.m1, nf.m2], axis=-1)
    m1m2 = smooth_trajectory(m1m2, smooth_K)
    m1m2 = smooth_trajectory(m1m2, smooth_K)
    # m1m2 = smooth_trajectory(m1m2, smooth_K)
    # m1m2 = smooth_trajectory(m1m2, smooth_K)
    nf = NaturalFrame(m1m2[:, 0] + 1.j * m1m2[:, 1])

    return nf


def plot_frenet_frame_components(
        X: np.ndarray,
        components: Dict[str, np.ndarray]
):
    """
    Plot frenet frame components.
    """
    dt = 1 / 25
    ts = np.arange(len(components['kappa'])) * dt
    x, y, z = X.T

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(6, 6)

    # Curvature
    ax = fig.add_subplot(gs[:2, :3])
    ax.set_title('$\kappa$')
    ax.plot(ts, components['kappa'])
    ax.set_xlabel('Time (s)')

    # Torsion
    ax = fig.add_subplot(gs[:2, 3:])
    ax.set_title('$\\tau$')
    ax.plot(ts, components['tau'])
    ax.set_xlabel('Time (s)')

    # 3D trajectory coloured by curvature
    ax = fig.add_subplot(gs[2:, :3], projection='3d')
    ax.scatter(x, y, z, c=components['kappa'], cmap='Reds', s=10, alpha=0.4, zorder=-1)
    equal_aspect_ratio(ax)

    # 3D trajectory coloured by torsion
    ax = fig.add_subplot(gs[2:, 3:], projection='3d')
    ax.scatter(x, y, z, c=components['tau'], cmap='Reds', s=10, alpha=0.4, zorder=-1)
    equal_aspect_ratio(ax)

    fig.tight_layout()

    return fig


def plot_bishop_frame_components(
        X: np.ndarray,
        nf: NaturalFrame
):
    """
    Plot bishop frame components.
    """
    dt = 1 / 25
    ts = np.arange(len(X)) * dt
    speeds = calculate_speeds(X) / dt

    # Use centre of mass of trajectory and centre whole trajectory around the origin
    X_com = X.mean(axis=1)
    X = X_com - X_com.mean(axis=0, keepdims=True)
    x, y, z = X.T

    # Construct colours
    colours = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    trace_opts = {'c': c, 's': 1}
    traj_opts = {'s': 10, 'alpha': 0.4}

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(12, 12)

    F = FrameNumpy(x=X.T, e0=nf.T.T, e1=nf.M1.T, e2=nf.M2.T)
    alpha = nf.m1.T / np.abs(nf.m1).max()
    beta = nf.m2.T / np.abs(nf.m2).max()
    C = ControlsNumpy(alpha=alpha, beta=beta, gamma=np.zeros(len(X) - 1))
    fa = FrameArtist(F, arrow_scale=.04, n_arrows=200)

    # Speeds
    ax = fig.add_subplot(gs[0:2, 0:4])
    ax.set_title('Speed')
    ax.scatter(ts, speeds, **trace_opts)
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[2:6, 0:4], projection='3d')
    s = ax.scatter(x, y, z, c=speeds, cmap='Reds', **traj_opts)
    fig.colorbar(s)
    equal_aspect_ratio(ax)

    # Time
    ax = fig.add_subplot(gs[6:12, 0:4], projection='3d')
    ax.set_title('Time')
    s = ax.scatter(x, y, z, c=ts, cmap='viridis_r', **traj_opts)
    fa.add_component_vectors(ax, draw_e0=False)
    fig.colorbar(s, location='bottom')
    equal_aspect_ratio(ax)

    # Curvature
    ax = fig.add_subplot(gs[0:2, 4:8])
    ax.set_title('$\kappa$')
    ax.scatter(ts, nf.kappa, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[2:6, 4:8], projection='3d')
    ax.scatter(x, y, z, c=nf.kappa, cmap='Reds', **traj_opts)
    equal_aspect_ratio(ax)

    # Twist
    ax = fig.add_subplot(gs[0:2, 8:12])
    ax.set_title('$\psi$')
    ax.scatter(ts, nf.psi, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_ylim([-np.pi - 1, np.pi + 1])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[2:6, 8:12], projection='3d')
    ax.scatter(x, y, z, c=nf.psi, cmap='twilight_shifted', **traj_opts)
    equal_aspect_ratio(ax)

    # m1
    ax = fig.add_subplot(gs[6:8, 4:8])
    ax.set_title('$m_1$')
    ax.scatter(ts, nf.m1, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[8:12, 4:8], projection='3d')
    ax.scatter(x, y, z, c=nf.m1, cmap='PRGn', norm=MidpointNormalize(midpoint=0), **traj_opts)
    fa.add_component_vectors(ax, draw_e0=False, draw_e2=False, C=C)
    equal_aspect_ratio(ax)

    # m2
    ax = fig.add_subplot(gs[6:8, 8:12])
    ax.set_title('$m_2$')
    ax.scatter(ts, nf.m2, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[8:12, 8:12], projection='3d')
    ax.scatter(x, y, z, c=nf.m2, cmap='PRGn', norm=MidpointNormalize(midpoint=0), **traj_opts)
    fa.add_component_vectors(ax, draw_e0=False, draw_e1=False, C=C)
    equal_aspect_ratio(ax)

    # fig.tight_layout()

    return fig


def plot_approximation_bishop_frame_components(
        X: np.ndarray,
        nf: NaturalFrame
):
    """
    Plot bishop frame components.
    """
    dt = 1 / 25
    ts = np.arange(len(X)) * dt
    speeds = calculate_speeds(X) / dt

    # Centre whole trajectory around the origin
    X -= X.mean(axis=0, keepdims=True)
    x, y, z = X.T

    # Construct colours
    colours = np.linspace(0, 1, len(X))
    cmap = plt.get_cmap('viridis_r')
    c = [cmap(c_) for c_ in colours]

    trace_opts = {'c': c, 's': 1}
    traj_opts = {'s': 10, 'alpha': 0.4}

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(12, 12)

    F = FrameNumpy(x=X.T, e0=nf.T.T, e1=nf.M1.T, e2=nf.M2.T)
    alpha = nf.m1.T / np.abs(nf.m1).max()
    beta = nf.m2.T / np.abs(nf.m2).max()
    C = ControlsNumpy(alpha=alpha, beta=beta, gamma=np.zeros(len(X) - 1))
    fa = FrameArtist(F, arrow_scale=.02, n_arrows=200)

    # Speeds
    ax = fig.add_subplot(gs[0:2, 0:4])
    ax.set_title('Speed')
    ax.scatter(ts, speeds, **trace_opts)
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[2:6, 0:4], projection='3d')
    s = ax.scatter(x, y, z, c=speeds, cmap='Reds', **traj_opts)
    fig.colorbar(s)
    equal_aspect_ratio(ax)

    # Time
    ax = fig.add_subplot(gs[6:12, 0:4], projection='3d')
    ax.set_title('Time')
    s = ax.scatter(x, y, z, c=ts, cmap='viridis_r', **traj_opts)
    fa.add_component_vectors(ax, draw_e0=False)
    fig.colorbar(s, location='bottom')
    equal_aspect_ratio(ax)

    # Curvature
    ax = fig.add_subplot(gs[0:2, 4:8])
    ax.set_title('$\kappa$')
    ax.scatter(ts, nf.kappa, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[2:6, 4:8], projection='3d')
    ax.scatter(x, y, z, c=nf.kappa, cmap='Reds', **traj_opts)
    equal_aspect_ratio(ax)

    # Twist
    ax = fig.add_subplot(gs[0:2, 8:12])
    ax.set_title('$\psi$')
    ax.scatter(ts, nf.psi, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_ylim([-np.pi - 1, np.pi + 1])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[2:6, 8:12], projection='3d')
    ax.scatter(x, y, z, c=nf.psi, cmap='PRGn', norm=MidpointNormalize(midpoint=0), **traj_opts)
    equal_aspect_ratio(ax)

    # m1
    ax = fig.add_subplot(gs[6:8, 4:8])
    ax.set_title('$m_1$')
    ax.scatter(ts, nf.m1, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[8:12, 4:8], projection='3d')
    ax.scatter(x, y, z, c=nf.m1, cmap='PRGn', norm=MidpointNormalize(midpoint=0), **traj_opts)
    fa.add_component_vectors(ax, draw_e0=False, draw_e2=False, C=C)
    equal_aspect_ratio(ax)

    # m2
    ax = fig.add_subplot(gs[6:8, 8:12])
    ax.set_title('$m_2$')
    ax.scatter(ts, nf.m2, **trace_opts)
    ax.axhline(y=0, color='grey', linestyle='--')
    ax.set_xlabel('Time (s)')

    ax = fig.add_subplot(gs[8:12, 8:12], projection='3d')
    ax.scatter(x, y, z, c=nf.m2, cmap='PRGn', norm=MidpointNormalize(midpoint=0), **traj_opts)
    fa.add_component_vectors(ax, draw_e0=False, draw_e1=False, C=C)
    equal_aspect_ratio(ax)

    # fig.tight_layout()

    return fig


def plot_approximation_planes(
        tumble_idxs: np.ndarray,
        X: np.ndarray,
        vertices: np.ndarray,
        deltas: np.ndarray,
        traj_angles: np.ndarray,
        planar_angles: np.ndarray,
        nonp: Dict[int, np.ndarray],
):
    """
    Plot approximation planes.
    """
    dt = 1 / 25
    ts = np.arange(len(X)) * dt
    tumble_ts = tumble_idxs * dt

    # Use centre of mass of trajectory and centre whole trajectory around the origin
    X_com = X.mean(axis=1)
    X = X_com - X_com.mean(axis=0, keepdims=True)

    # Construct colours
    colours = np.linspace(0, 1, len(vertices))
    cmap = plt.get_cmap('viridis_r')
    c = np.array([cmap(c_) for c_ in colours])

    # Set up plot
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(len(deltas) + 3, 6)

    # Trace of the angles
    for i, delta in enumerate(deltas):
        angle_ts = centre_select(tumble_ts, len(traj_angles[delta]))
        angle_cs = centre_select(c, len(traj_angles[delta]))
        ax = fig.add_subplot(gs[i, :])
        ax.set_title(f'Delta = {delta} vertices')
        s1 = ax.scatter(angle_ts, np.sin(traj_angles[delta]), label='TA', marker='x')
        s2 = ax.scatter(angle_ts, np.sin(planar_angles[delta]), label='PA', marker='o')
        ax.set_xlim(left=ts[0], right=ts[-1])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$sin(\\theta)$')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['0', '1'])
        for j, t in enumerate(angle_ts):
            ax.axvline(x=t, color=angle_cs[j], zorder=-1)

        ax2 = ax.twinx()
        nonp_ts = centre_select(tumble_ts, len(nonp[delta]))
        s3 = ax2.plot(nonp_ts, nonp[delta], marker='2', alpha=0.7, color='purple')

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc=1)

        # series = s1 + s2 + s3
        # labels = [s.get_label() for s in series]
        # ax.legend(series, labels, loc=1)

    # 3D trajectory of approximation
    T = len(vertices)
    ax = fig.add_subplot(gs[len(deltas):, :3], projection='3d')
    x, y, z = vertices[:T].T
    ax.scatter(x, y, z, c=c, marker='x', s=50, alpha=0.6, zorder=1)

    # Add approximation trajectory
    points = vertices[:T][:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, color=c, zorder=5, linewidth=2, linestyle=':', alpha=0.5)
    ax.add_collection(lc)
    equal_aspect_ratio(ax)

    # Actual 3D trajectory
    ax = fig.add_subplot(gs[len(deltas):, 3:], projection='3d')
    x, y, z = X.T
    ax.scatter(x, y, z, c=ts, cmap='viridis_r', s=10, alpha=0.4, zorder=-1)
    x, y, z = vertices[:T].T
    ax.scatter(x, y, z, color='blue', marker='x', s=50, alpha=0.9, zorder=10)
    equal_aspect_ratio(ax)

    return fig


def plot_lab_bearing(X: np.ndarray):
    # Use centre of mass of trajectory
    X_com = X.mean(axis=1)

    # Centre whole trajectory around the origin
    X_centred = X_com - X_com.mean(axis=0, keepdims=True)
    X_from_start = X_com - X_com[0]

    grad_opts = {'color': 'green', 'alpha': 0.5}

    def get_grad(x):
        xs = smooth_trajectory(x[:, None], window_len=201)[:, 0]
        xs = smooth_trajectory(xs[:, None], window_len=201)[:, 0]
        g = np.gradient(xs)
        gs = smooth_trajectory(g[:, None], window_len=201)[:, 0]
        gs = smooth_trajectory(gs[:, None], window_len=201)[:, 0]
        return gs

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))

    for i, X in enumerate([X_centred, X_from_start]):
        # Convert to spherical coordinates
        r = np.linalg.norm(X, axis=-1)
        theta = np.where(r == 0, np.zeros_like(r), np.arccos(X[:, 2] / r))
        phi = np.arctan2(X[:, 1], X[:, 0])

        if i == 1:
            theta[0] = theta[1]
            phi[0] = phi[1]

        ax = axes[0, i]
        ax.set_title('r - ' + ['X_centred', 'X_from_start'][i])
        ax.plot(r)
        ax2 = ax.twinx()
        ax2.plot(get_grad(r), **grad_opts)

        ax = axes[1, i]
        ax.set_title('$\\theta$')
        ax.plot(theta)
        ax2 = ax.twinx()
        ax2.plot(get_grad(theta), **grad_opts)

        ax = axes[2, i]
        ax.set_title('$\phi$')
        ax.plot(phi)
        ax2 = ax.twinx()
        ax2.plot(get_grad(phi), **grad_opts)

    return fig


def plot_frenet_frame_components_for_trial():
    """
    Plot frenet frame components.
    """
    args = get_args()
    smooth_K = args.smoothing_window_curvature
    smooth_X = args.smoothing_window
    trial = Trial.objects.get(id=args.trial)
    X = get_trajectory_from_args(args)
    components = calculate_frenet_frame_components(X, smooth_K)
    fig = plot_frenet_frame_components(X, components)
    fig.suptitle(f'Frenet frame components. Trial={trial.id}. Smooth X={smooth_X}. Smooth K={smooth_K}')
    fig.tight_layout()

    if save_plots:
        plt.savefig(
            make_filename('frenet_components', args),
            transparent=True
        )

    if show_plots:
        plt.show()


def plot_bishop_frame_components_for_trial():
    """
    Plot bishop frame components.
    """
    args = get_args()
    smooth_K = args.smoothing_window_curvature
    smooth_X = args.smoothing_window
    trial = Trial.objects.get(id=args.trial)
    X = get_trajectory_from_args(args)

    if 0:
        nf = calculate_bishop_frame_components(X, smooth_K)
        fig = plot_bishop_frame_components(X, nf)
        fig.suptitle(f'Bishop frame components. Trial={trial.id}. Smooth X={smooth_X}. Smooth K={smooth_K}')
        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename('bishop_components', args),
                transparent=True
            )
        if show_plots:
            plt.show()

    if 0:
        error_limit = 0.02
        e0, e1, e2 = calculate_trajectory_frame(X, pca_window=args.planarity_window)
        approx, distance, height, smooth_e0, smooth_K \
            = find_approximation(X, e0, error_limit=error_limit, max_attempts=50)
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles_j, nonplanar_angles_j, twist_angles_j, e0, e1, e2 = approx

        deltas = np.arange(2, 5, step=1)
        traj_angles = calculate_trajectory_angles(vertices, deltas)
        planar_angles = calculate_planar_angles(vertices, deltas)

        nonp = {}
        for delta in deltas:
            pcas = PCACache(calculate_pcas(vertices, delta + 2))
            r = pcas.explained_variance_ratio.T
            nonp[delta] = r[2] / np.sqrt(r[1] * r[0])

        # nf = calculate_bishop_frame_components(X, smooth_K)
        fig = plot_approximation_planes(
            tumble_idxs, X, vertices, deltas, traj_angles, planar_angles, nonp
        )
        mse = np.mean(np.sum((X.mean(axis=1) - X.mean(axis=(0, 1)) - X_approx)**2, axis=-1))
        fig.suptitle(f'Approximation angles. Trial={trial.id}. Smooth X={smooth_X}. Approximation error = {mse:.5f}.')
        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename(f'approx_angles_e={error_limit:.5f}', args),
                transparent=True
            )
        if show_plots:
            plt.show()

    if 0:
        error_limit = 0.0005
        e0, e1, e2 = calculate_trajectory_frame(X, pca_window=args.planarity_window)
        approx, distance, height, smooth_e0, smooth_K \
            = find_approximation(X, e0, error_limit=error_limit, max_attempts=50)
        X_approx, vertices, tumble_idxs, run_durations, run_speeds, planar_angles_j, nonplanar_angles_j, twist_angles_j, _, _, _ = approx

        nf = calculate_bishop_frame_components(vertices, smooth_K)
        fig = plot_approximation_bishop_frame_components(vertices, nf)
        fig.suptitle(
            f'Bishop frame components. Trial={trial.id}. Smooth X={smooth_X}. Smooth K={smooth_K}. Approximation error = {error_limit:.5f}.')
        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename(f'bishop_components_approx_e={error_limit:.5f}', args),
                transparent=True
            )
        if show_plots:
            plt.show()

    if 1:
        fig = plot_lab_bearing(X)
        fig.suptitle(
            f'Lab bearing. Trial={trial.id}. Smooth X={smooth_X}.')
        fig.tight_layout()
        if save_plots:
            plt.savefig(
                make_filename(f'lab_bearing', args),
                transparent=True
            )
        if show_plots:
            plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # plot_frenet_frame_components_for_trial()
    plot_bishop_frame_components_for_trial()
