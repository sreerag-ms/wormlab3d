import os
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Trial
from wormlab3d.simple_worm.estimate_k import get_K_estimates_from_args
from wormlab3d.toolkit.plot_utils import tex_mode, equal_aspect_ratio
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.brownian_particle import BrownianParticle, ActiveParticle, BoundedParticle, ConfinedParticle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.pca import get_planarity_from_args
from wormlab3d.trajectories.util import calculate_speeds, calculate_htd

animate = False
show_plots = True
save_plots = False
img_extension = 'png'
fps = 25
playback_speed = 10
n_revolutions = 2

tex_mode()


def get_trajectory(args: Namespace):
    X_slice = get_trajectory_from_args(args)
    args.trajectory_point = None
    X_full = get_trajectory_from_args(args)
    return X_full, X_slice


def make_plot(
        X_slice: np.ndarray,
        X_full: np.ndarray = None,
        title: str = None,
        filename: str = None,
        colours: np.ndarray = None,
        cmap: str = 'viridis_r',
        show_colourbar: bool = False,
        ax: Axes = None,
        draw_edges: bool = True,
        draw_worm: bool = True,
        show_axis: bool = True,
        show_ticks: bool = True,
):
    x, y, z = X_slice.T
    if X_full is None:
        draw_worm = False

    # Construct colours
    if colours is None:
        colours = np.linspace(0, 1, len(X_slice))
    cmap = plt.get_cmap(cmap)
    c = [cmap(c_) for c_ in colours]

    # Create figure if required
    return_ax = False
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.get_figure()
        return_ax = True

    # Scatter the vertices
    s = ax.scatter(x, y, z, c=c, s=100, alpha=0.4, zorder=-1)
    if show_colourbar:
        fig.colorbar(s)

    # Draw lines connecting points
    if draw_edges:
        points = X_slice[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
        ax.add_collection(lc)

    # Add worm
    if draw_worm:
        FS = FrameSequenceNumpy(x=X_full.transpose(0, 2, 1))
        fa = FrameArtist(F=FS[0])
        fa.add_midline(ax)

    # Set title
    if title is not None:
        ax.set_title(title)

    # Setup axis
    equal_aspect_ratio(ax)
    # ax.view_init(azim=170, elev=20)
    if not show_axis:
        ax.axis('off')
    if not show_ticks:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    if return_ax:
        return ax

    fig.tight_layout()

    if animate:
        # Aspects
        azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X_slice))
        ax.view_init(azim=azims[0])  # elev

        def update(frame_num: int):
            # Rotate the view.
            ax.view_init(azim=azims[frame_num])

            # Update the worm
            if draw_worm:
                fa.update(FS[frame_num])
            return ()

        idxs_mask = np.array([i % playback_speed == 0 for i in np.arange(np.round(len(X_slice)))])
        frame_nums = np.arange(len(X_slice))[idxs_mask].tolist()
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frame_nums,
            blit=True,
            interval=1 / fps
        )

    if save_plots:
        assert filename is not None
        path = LOGS_PATH / f'{START_TIMESTAMP}_{filename}'
        if animate:
            metadata = dict(
                title=title,
                artist='WormLab Leeds'
            )
            save_path = path.with_suffix('.mp4')
            logger.info(f'Saving animation to {save_path}.')
            ani.save(save_path, writer='ffmpeg', fps=fps, metadata=metadata)
        else:
            save_path = path.with_suffix(f'.{img_extension}')
            logger.info(f'Saving plot to {save_path}.')
            plt.savefig(save_path, transparent=True)

    if show_plots:
        plt.show()

    plt.close(fig)


def plot_trajectory():
    """
    Draw the trajectory coloured by the time elapsed.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    # np.savez(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_trial={args.trial}', X_full)

    make_plot(
        title=f'Trial {args.trial}.',
        filename=f'trajectory_trial={args.trial}_{args.midline3d_source}',
        X_full=X_full,
        X_slice=X_slice,
        draw_worm=False,
    )


def plot_trajectory_head_tail_distance():
    """
    Draw the trajectory coloured by the head-to-tail distance.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    htd = calculate_htd(X_full)

    make_plot(
        title=f'Head-tail distance. Trial {args.trial}.',
        filename=f'trajectory_HTD_trial={args.trial}_{args.midline3d_source}',
        X_full=X_full,
        X_slice=X_slice,
        colours=htd,
        cmap='OrRd_r',
        show_colourbar=True
    )


def plot_trajectory_signed_speed():
    """
    Draw the trajectory coloured by the signed speed.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    signed_speeds = calculate_speeds(X_full, signed=True)

    make_plot(
        title=f'Speed. Trial {args.trial}.',
        filename=f'trajectory_signed-speed_trial={args.trial}_{args.midline3d_source}',
        X_full=X_full,
        X_slice=X_slice,
        colours=signed_speeds,
        cmap='PRGn',
        show_colourbar=True
    )


def plot_trajectory_K():
    """
    Draw the trajectory coloured by the estimate of K.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    K_ests = np.log(get_K_estimates_from_args(args))

    make_plot(
        title=f'log(K\_est). Trial {args.trial}.',
        filename=f'trajectory_K-est_trial={args.trial}_{args.midline3d_source}',
        X_full=X_full,
        X_slice=X_slice,
        colours=K_ests,
        cmap='OrRd_r',
        show_colourbar=True,
        draw_worm=True
    )


def plot_trajectory_planarity():
    """
    Draw the trajectory coloured by the planarity.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    planarity = get_planarity_from_args(args)

    make_plot(
        title=f'Planarity. Trial {args.trial}. PCA window: {args.planarity_window} frames.',
        filename=f'trajectory_planarity_w={args.planarity_window}_trial={args.trial}_{args.midline3d_source}',
        X_full=X_full,
        X_slice=X_slice,
        colours=planarity,
        cmap='OrRd',
        show_colourbar=True
    )


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
        filename=f'brownian_particle_D={D}_n={n_steps}_T={total_time}',
        X_slice=X,
    )


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
        filename=f'active_particle_D={D}_m={momentum:.2f}_n={n_steps}_T={total_time}',
        X_slice=X,
    )


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
                X_slice=X,
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
        filename=f'bounded_particle_D={D}_m={momentum:.2f}_n={n_steps}_T={total_time}',
        X_slice=X,
    )


def plot_confined_particle_trajectory():
    """
    Generate and plot trajectory of a randomly generated brownian particle with momentum and confinement.
    """
    total_time = 13 * 60
    n_steps = total_time * 25
    D = 1
    momentum = 0.95
    unconfined_duration_mean = 30
    unconfined_duration_variance = 1
    confined_duration_mean = 30
    confined_duration_variance = 1
    D_confined = 0.01

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
        filename=f'confined_particle_Du={D}_Dc={D_confined}_m={momentum:.2f}_n={n_steps}_T={total_time}'
                 f'_u={unconfined_duration_mean:.2f},{unconfined_duration_variance:.2f}'
                 f'_c={unconfined_duration_mean:.2f},{unconfined_duration_variance:.2f}',
        X_slice=X,
    )


def plot_trajectory_trial_list():
    """
    Draw the trajectory coloured by the time elapsed.
    """
    args = get_args()
    args.trajectory_point = -1

    trial_ids = [65, 67, 172, 37, 64, 122, 9, 272, 76, 241, 1, 182, 106, 116, 35, 73, 124, 164, 114, 13, 103, 112, 17,
                 236, 80, 251, 168, 232, 252, 320, 239, 79, 162, 145, 256, 321, 111, 19, 121, 15, 36, 151, 167, 154,
                 127, 298, 72, 180, 279, 186, 265, 304, 110, 228, 247, 135, 123, 139, ]

    for i, trial_id in enumerate(trial_ids):
        args.trial = trial_id
        X_slice, meta = get_trajectory_from_args(args, return_meta=True)
        # np.savez(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_trial={args.trial}', X_full)
        if 'type' in meta and meta['type'] == 'tracking-only':
            src = 'tracking'
        else:
            src = args.midline3d_source

        trial = Trial.objects.get(id=trial_id)

        make_plot(
            title=f'Trial {args.trial}. '
                  f'Duration {trial.duration:%M:%S}. '
                  f'Concentration {trial.experiment.concentration:.2f}\% \n'
                  f'User {trial.experiment.user}. Source={src}.',
            filename=f'trajectory_{i:02d}_trial={args.trial:03d}_{src}',
            X_full=None,
            X_slice=X_slice,
            draw_worm=False,
        )


def plot_multiple_trajectories(
        trial_ids: list,
        plot_combined: bool = True,
        plot_individual: bool = True,
):
    """
    Draw multiple trajectories on the same scale.
    """
    args = get_args()
    args.trajectory_point = -1
    Xs = []
    metas = []
    durations_frames = []
    trials = []
    for trial_id in trial_ids:
        args.trial = trial_id
        X_slice, meta = get_trajectory_from_args(args, return_meta=True)
        Xs.append(X_slice)  # (T, 3)
        metas.append(meta)
        trial = Trial.objects.get(id=trial_id)
        trials.append(trial)
        durations_frames.append(trial.n_frames_min)

    # Calculate the common range to use
    ranges = np.zeros(3)
    for i in range(3):
        for X in Xs:
            r = np.nanmax(X[:, i]) - np.nanmin(X[:, i])
            ranges[i] = max(ranges[i], r)
    r_max = max(ranges) * 0.75

    if plot_combined:
        fig_combined = plt.figure(figsize=(20, 10))
        gs_combined = GridSpec(nrows=1, ncols=4)

    for i, trial_id in enumerate(trial_ids):
        args.trial = trial_id
        trial = trials[i]
        X_slice = Xs[i]
        meta = metas[i]
        if 'type' in meta and meta['type'] == 'tracking-only':
            src = 'tracking'
        else:
            src = args.midline3d_source
        # np.savez(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_trial={args.trial}_src={src}', X_slice)

        # Subsample for smaller svg sizes
        # X_slice = X_slice[::4]

        # Create colours consistent across all plots
        colours = np.linspace(0, durations_frames[i] / max(durations_frames), len(X_slice))

        axes = []
        if plot_individual:
            fig_individual = plt.figure(figsize=(10, 10))
            axes.append(fig_individual.add_subplot(projection='3d'))
        if plot_combined:
            axes.append(fig_combined.add_subplot(gs_combined[0, i], projection='3d'))

        for ax in axes:
            make_plot(
                title=f'Concentration {trial.experiment.concentration:.2f}\%\nDuration {trial.duration:%M:%S}',
                X_slice=X_slice,
                ax=ax,
                colours=colours,
                show_colourbar=False
            )

            # Set axis range
            for j in range(3):
                Xj_max = np.nanmax(X_slice[:, j])
                Xj_min = np.nanmin(X_slice[:, j])
                adj = (r_max - (Xj_max - Xj_min)) / 2
                getattr(ax, f'set_{"xyz"[j]}lim')(Xj_min - adj, Xj_max + adj)

        if plot_individual:
            fig_individual.tight_layout()
            if save_plots:
                plt.savefig(
                    LOGS_PATH / f'{START_TIMESTAMP}_trajectory_trial={trial.id:03d}_src={src}.{img_extension}',
                    transparent=True
                )

    if plot_combined:
        fig_combined.tight_layout()
        if save_plots:
            trial_ids = ','.join([f'{t:03d}' for t in trial_ids])
            plt.savefig(
                LOGS_PATH / f'{START_TIMESTAMP}_trajectory_trials={trial_ids}.{img_extension}',
                transparent=False
            )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    # plot_trajectory()
    # plot_trajectory_head_tail_distance()
    # plot_trajectory_signed_speed()
    # plot_trajectory_K()
    # plot_trajectory_planarity()
    # plot_brownian_trajectory()
    # plot_active_particle_trajectory()
    # plot_active_particle_trajectories()
    # plot_bounded_particle_trajectory()
    # plot_confined_particle_trajectory()

    # plot_trajectory_trial_list()
    plot_multiple_trajectories(
        trial_ids=[162, 17, 272, 37],
        plot_combined=True,
        plot_individual=True
    )
