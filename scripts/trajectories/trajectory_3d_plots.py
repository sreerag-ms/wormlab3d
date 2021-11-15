import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.toolkit.plot_utils import tex_mode, equal_aspect_ratio
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.brownian_particle import BrownianParticle, ActiveParticle, BoundedParticle, ConfinedParticle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.util import calculate_planarity, calculate_speeds, calculate_htd

show_plots = True
save_plots = True
img_extension = 'png'

tex_mode()


def get_trajectory(args: Namespace):
    X_slice = get_trajectory_from_args(args)
    args.trajectory_point = None
    X_full = get_trajectory_from_args(args)
    return X_full, X_slice


def make_plot(
        title: str,
        X: np.ndarray,
        colours: np.ndarray = None,
        cmap: str = 'jet',
        show_colourbar: bool = False,
        ax: Axes = None
):
    x, y, z = X.T
    if colours is None:
        colours = np.linspace(0, 1, len(X))

    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()

    # Scatter the vertices
    s = ax.scatter(x, y, z, c=colours, cmap=cmap, s=10, alpha=0.4)
    if show_colourbar:
        fig.colorbar(s)

    # Draw lines connecting points
    points = X[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap)
    ax.add_collection(lc)

    equal_aspect_ratio(ax)

    ax.set_title(title)
    fig.tight_layout()


def plot_trajectory_head_tail_distance():
    """
    Draw the trajectory coloured by the head-to-tail distance.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    htd = calculate_htd(X_full)
    # scores = (ht_distances - ht_distances.min()) / (ht_distances.max() - ht_distances.min())

    make_plot(
        title=f'Head-tail distance. Trial {args.trial}.',
        X=X_slice,
        colours=htd,
        cmap='OrRd_r',
        show_colourbar=True
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_HTD'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_signed_speed():
    """
    Draw the trajectory coloured by the signed speed.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    signed_speeds = calculate_speeds(X_full, signed=True)

    make_plot(
        title=f'Speed. Trial {args.trial}.',
        X=X_slice,
        colours=signed_speeds,
        cmap='PRGn',
        show_colourbar=True
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_signed-speed'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_K():
    """
    Draw the trajectory coloured by the estimate of K.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    K_ests = np.log(get_K_estimates_from_args(args))

    make_plot(
        title=f'log(K_est). Trial {args.trial}.',
        X=X_slice,
        colours=K_ests,
        cmap='OrRd_r',
        show_colourbar=True
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_K-est'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_trajectory_planarity():
    """
    Draw the trajectory coloured by the planarity.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    planarity = calculate_planarity(X_full, window_size=args.planarity_window)

    make_plot(
        title=f'Planarity. Trial {args.trial}. PCA window: {args.planarity_window} frames.',
        X=X_slice,
        colours=planarity,
        cmap='OrRd',
        show_colourbar=True
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_trajectory_planarity'
            f'_w={args.planarity_window}'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_brownian_trajectory():
    """
    Generate and plot trajectory of a randomly generated brownian particle.
    """
    D = 100
    n_steps = 1000
    total_time = 1
    p = BrownianParticle(D=D)
    X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)

    make_plot(
        title=f'Brownian particle. D={D}, n\_steps={n_steps}, total\_time={total_time}.',
        X=X,
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_brownian_particle'
            f'_D={D}_n={n_steps}_T={total_time}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_active_particle_trajectory():
    """
    Generate and plot trajectory of a randomly generated brownian particle with momentum.
    """
    D = 100
    momentum = 0.2
    n_steps = 100
    total_time = 1
    p = ActiveParticle(D=D, momentum=momentum)
    X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)

    make_plot(
        title=f'Active particle. D={D}, momentum={momentum:.1f}, n\_steps={n_steps}, total\_time={total_time}.',
        X=X,
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_active_particle'
            f'_D={D}_m={momentum:.2f}_n={n_steps}_T={total_time}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_active_particle_trajectories():
    """
    Generate and plot trajectory of a randomly generated brownian particle with momentum.
    """
    D = 100
    momenta = [0, 0.33, 0.66, 1]
    n_steps = 100
    total_time = 1
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(nrows=2, ncols=2)

    i = 0
    for row_idx in range(2):
        for col_idx in range(2):
            p = ActiveParticle(D=D, momentum=momenta[i])
            X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d')
            make_plot(
                title=f'Momentum={momenta[i]:.2f}.',
                X=X,
                ax=ax
            )
            i += 1

    fig.suptitle(f'Active particles. D={D}, n\_steps={n_steps}, total\_time={total_time}.')

    fig.tight_layout()
    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_active_particles'
            f'_D={D}_ms={",".join([f"{m:.2f}" for m in momenta])}_n={n_steps}_T={total_time}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_bounded_particle_trajectory():
    """
    Generate and plot trajectory of a bounded randomly generated brownian particle with momentum.
    """
    D = 0.005
    momentum = 0.99
    n_steps = 1000
    total_time = 0.1
    bounds = np.array([[-0.1, 0.1], [-10, 10], [-0.1, 0.1]])

    p = BoundedParticle(D=D, momentum=momentum, bounds=bounds)
    X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)

    make_plot(
        title=f'Bounded particle. D={D}, momentum={momentum:.2f}, n\_steps={n_steps}, total\_time={total_time}.',
        X=X,
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_bounded_particle'
            f'_D={D}_m={momentum:.2f}_n={n_steps}_T={total_time}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


def plot_confined_particle_trajectory():
    """
    Generate and plot trajectory of a randomly generated brownian particle with momentum and confinement.
    """
    D = 10
    momentum = 0.8
    n_steps = 1000
    total_time = 10
    unconfined_duration_mean = 1
    unconfined_duration_variance = 0.1
    confined_duration_mean = 3
    confined_duration_variance = 0.1
    D_confined = 0.05

    p = ConfinedParticle(
        D=D,
        momentum=momentum,
        unconfined_duration_mean=unconfined_duration_mean,
        unconfined_duration_variance=unconfined_duration_variance,
        confined_duration_mean=confined_duration_mean,
        confined_duration_variance=confined_duration_variance,
        D_confined=D_confined,
    )
    X = p.generate_trajectory(n_steps=n_steps, total_time=total_time)

    make_plot(
        title=f'Confined particle.\n'
              f'D\_unconfined={D}, D\_confined={D_confined}, momentum={momentum:.2f}, n\_steps={n_steps}, total\_time={total_time}.\n'
              f'Unconfined duration $\sim \mathcal{{N}}({unconfined_duration_mean:.2f}, {unconfined_duration_variance:.2f})$. \n'
              f'Confined duration $\sim \mathcal{{N}}({confined_duration_mean:.2f}, {unconfined_duration_variance:.2f})$.',
        X=X,
    )

    if save_plots:
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_confined_particle'
            f'_Du={D}_Dc={D_confined}_m={momentum:.2f}_n={n_steps}_T={total_time}'
            f'_u={unconfined_duration_mean:.2f},{unconfined_duration_variance:.2f}'
            f'_c={unconfined_duration_mean:.2f},{unconfined_duration_variance:.2f}'
            f'.{img_extension}'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    plot_trajectory_head_tail_distance()
    plot_trajectory_signed_speed()
    plot_trajectory_K()
    plot_trajectory_planarity()
    plot_brownian_trajectory()
    plot_active_particle_trajectory()
    plot_active_particle_trajectories()
    plot_bounded_particle_trajectory()
    plot_confined_particle_trajectory()
