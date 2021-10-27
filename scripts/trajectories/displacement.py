import os

import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.brownian_particle import BrownianParticle
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.displacement import calculate_displacements, plot_displacement_histograms, \
    calculate_displacement_projections, plot_displacement_projections_histograms, plot_squared_displacements_over_time, \
    calculate_transitions_and_dwells_multiple_deltas, calculate_msd, plot_msd, plot_msd_multiple

# tex_mode()

show_plots = True
save_plots = False


def displacement():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    displacements = calculate_displacements(trajectory, args.deltas, args.aggregation)
    plot_displacement_histograms(displacements)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_histograms'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_{args.aggregation}'
            f'_d={",".join([str(d) for d in args.deltas])}'
            f'_u={args.trajectory_point}'
            '.png'
        )
    if show_plots:
        plt.show()


def displacement_projections():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    displacements = calculate_displacement_projections(trajectory, args.deltas)
    plot_displacement_projections_histograms(displacements)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_histograms_projections'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={",".join([str(d) for d in args.deltas])}'
            f'_u={args.trajectory_point}'
            '.png'
        )
    if show_plots:
        plt.show()


def displacement_over_time():
    args = get_args()
    trajectory = get_trajectory_from_args(args)
    displacements = calculate_displacements(trajectory, args.deltas, args.aggregation)
    dwells = calculate_transitions_and_dwells_multiple_deltas(displacements)
    plot_squared_displacements_over_time(displacements, dwells)
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_traces'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_{args.aggregation}'
            f'_d={",".join([str(d) for d in args.deltas])}'
            f'_u={args.trajectory_point}'
            '.png'
        )
    if show_plots:
        plt.show()


def msd():
    args = get_args()
    trajectory = get_trajectory_from_args(args)

    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    print(deltas)
    msds = calculate_msd(trajectory, deltas)
    plot_msd(msds)

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_msd'
            f'_trial={args.trial}'
            f'_{args.midline3d_source}'
            f'_d={args.min_delta}-{args.max_delta}'
            f'_ds={args.delta_step}'
            f'_u={args.trajectory_point}'
            f'_p={args.projection}'
            '.svg'
        )
    if show_plots:
        plt.show()


def msd_multiple():
    args = get_args()

    us = [0.1, 0.5, 0.9]
    # us = [0.5]
    # projections = ['3D', 'x', 'y', 'z']
    projections = ['x', 'y', 'z']
    msds = {}
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    print(deltas)
    for u in us:
        args.trajectory_point = u
        msds[u] = {}
        for p in projections:
            logger.info(f'Calculating MSD for u={u} and projection={p}.')
            args.projection = p
            trajectory = get_trajectory_from_args(args)
            msds_u = calculate_msd(trajectory, deltas)
            msds[u][p] = msds_u

    plot_msd_multiple(msds, deltas)

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

        frames_str = ''
        if args.start_frame is not None or args.end_frame is not None:
            start_frame = args.start_frame if args.start_frame is not None else 0
            end_frame = args.end_frame if args.end_frame is not None else -1
            frames_str = f'_f={start_frame}-{end_frame}'

        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_msd_multiple'
            f'_trial={args.trial}' +
            frames_str +
            f'_{args.midline3d_source}'
            f'_d={args.min_delta}-{args.max_delta}'
            f'_ds={args.delta_step}'
            f'_u={",".join([str(u) for u in us])}'
            '.svg'
        )
    if show_plots:
        plt.show()


def msd_wt3d_vs_reconst():
    args = get_args()

    us = [0.1, 0.5, 0.9]
    srcs = ['reconst', 'reconst', 'WT3D']
    src_files = ['039_shelf-2.hdf5', '039_shelf-2.hdf5', None]
    frames = [(0, 20000), (449, 4999), (449, 4999)]
    projections = ['x', 'y', 'z']

    msds = {}
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    print(deltas)
    for u in us:
        args.trajectory_point = u
        msds[u] = {}
        for k, src in enumerate(srcs):
            args.midline3d_source = src
            args.midline3d_source_file = src_files[k]
            args.start_frame = frames[k][0]
            args.end_frame = frames[k][1]
            msds[u][k] = {}
            for p in projections:
                logger.info(f'Calculating MSD for src={src}, frames={frames[k]} u={u} and projection={p}.')
                args.projection = p
                trajectory = get_trajectory_from_args(args)
                msds_usp = calculate_msd(trajectory, deltas)
                msds[u][k][p] = msds_usp

    deltas = deltas / 25

    linestyles = {
        'x': 'dotted',
        'y': 'dashed',
        'z': 'dashdot'
    }
    colors = {
        0: 'red',
        1: 'green',
        2: 'blue',
    }

    for u in us:
        fig, ax = plt.subplots(1, figsize=(12, 8), sharex=True, sharey=True)

        msds_u = msds[u]
        for k, src in enumerate(srcs):
            msds_us = msds_u[k]
            for p in projections:
                msds_usp = msds_us[p]
                msd_vals = np.array(list(msds_usp.values()))
                ax.plot(deltas, msd_vals, label=f'{src}, f={frames[k]}, p={p}', linestyle=linestyles[p],
                        color=colors[k], alpha=0.5)

        ax.set_title(f'MSD u={u}')
        ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
        ax.set_xlabel('$\Delta s$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid()
        ax.legend()
        fig.tight_layout()

        if save_plots:
            os.makedirs(LOGS_PATH, exist_ok=True)

            plt.savefig(
                LOGS_PATH + '/' + START_TIMESTAMP +
                f'_msd_WT3D_vs_reconst'
                f'_trial={args.trial}'
                f'_d={args.min_delta}-{args.max_delta}'
                f'_ds={args.delta_step}'
                f'_u={u}'
                '.svg'
            )
        if show_plots:
            plt.show()


def msd_brownian():
    args = get_args()
    n_runs = 10

    # Use same brownian parameters for all simulations
    fps = 25
    total_time = 13 * 60
    n_steps = total_time * fps
    D = 1

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    deltas_s = deltas / fps

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Brownian Particle. D={D}, n_steps={n_steps}, total_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')

    for i in range(n_runs):
        # Generate the brownian trajectory
        p = BrownianParticle(D=D)
        trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        msds = calculate_msd(trajectory, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals)

    fig.tight_layout()

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_msd'
            f'_brownian_particle'
            f'_D={D}_n={n_steps}_T={total_time}'
            f'_d={args.min_delta}-{args.max_delta}'
            f'_ds={args.delta_step}'
            '.svg'
        )
    if show_plots:
        plt.show()


def msd_brownian_varying_Ds():
    args = get_args()

    # Use same time window for all simulations
    fps = 25
    total_time = 10 * 60
    n_steps = total_time * fps

    # Define deltas
    deltas = np.arange(args.min_delta, args.max_delta, step=args.delta_step)
    deltas_s = deltas / fps

    # Vary the diffusion coefficient
    Ds = [0.1, 1, 10, 100]

    # Set up plot
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.set_title(f'MSD of Brownian Particle. n_steps={n_steps}, total_time={total_time}s.')
    ax.set_ylabel('$MSD=<(x(t+\Delta)-x(t))^2>_t$')
    ax.set_xlabel('$\Delta s$')
    ax.set_yscale('log')
    ax.set_xscale('log')

    for D in Ds:
        # Generate the brownian trajectory
        p = BrownianParticle(D=D)
        trajectory = p.generate_trajectory(n_steps=n_steps, total_time=total_time)
        msds = calculate_msd(trajectory, deltas)
        msd_vals = np.array(list(msds.values()))
        ax.plot(deltas_s, msd_vals, label=f'D={D}')

    ax.legend()
    fig.tight_layout()

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(
            LOGS_PATH + '/' + START_TIMESTAMP +
            f'_msd'
            f'_brownian_particle'
            f'_D={",".join([str(D) for D in Ds])}_n={n_steps}_T={total_time}'
            f'_d={args.min_delta}-{args.max_delta}'
            f'_ds={args.delta_step}'
            '.svg'
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    displacement()
    displacement_projections()
    displacement_over_time()
    msd()
    msd_multiple()
    msd_wt3d_vs_reconst()
    msd_brownian()
    msd_brownian_varying_Ds()
