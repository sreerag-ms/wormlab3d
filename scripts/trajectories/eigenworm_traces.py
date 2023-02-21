import os
import argparse
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import generate_interactive_scatter_clip
from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.helicities import calculate_helicities, plot_helicities
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds, calculate_rotation_matrix

# tex_mode()

show_plots = True
save_plots = True
img_extension = 'svg'


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot an eigenworm basis.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--eigenworms', type=str,
                        help='Eigenworms by id.')
    parser.add_argument('--n-components', type=int, default=20,
                        help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--plot-components', type=lambda s: [int(item) for item in s.split(',')],
                        default='0,1', help='Comma delimited list of component idxs to plot.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument(
        '--vline',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add vlines on magnitude plot to show region changes",
    )
    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def traces(x_label: str = 'time'):
    args = parse_args()
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=args.reconstruction,
        n_components=args.n_components,
        regenerate=False
    )

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': 25
    }

    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    X, meta = get_trajectory(**common_args)
    N = len(X)
    if x_label == 'time':
        ts = np.linspace(0, N / reconstruction.trial.fps, N)
    else:
        ts = np.arange(N) + (args.start_frame if args.start_frame is not None else 0)

    # Speed
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=True)

    # Planarity
    logger.info('Fetching planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
    nonp = pcas.nonp

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    X_ew = ew.transform(np.array(Z))

    # Helicity
    logger.info('Calculating helicities.')
    c = calculate_helicities(X)

    # Plot
    fig, axes = plt.subplots(7, figsize=(16, 18), sharex=True)

    # Speeds
    ax = axes[0]
    ax.axhline(y=0, color='lightgrey')
    ax.plot(ts, speeds)
    ax.set_ylabel('Speed')
    ax.set_title('Speed.')
    ax.grid()

    # Planarities
    ax = axes[1]
    ax.plot(ts, nonp)
    ax.set_ylabel('Non-Planarity')
    ax.set_title('Non-Planarity.')
    ax.grid()

    # Helicities
    ax = axes[2]
    ax.plot(ts, c)
    ax.set_ylabel('Helicity')
    ax.set_title('Helicity.')
    ax.grid()

    # Eigenworms - absolute values
    ax = axes[3]
    for i in args.plot_components:
        ax.plot(ts, np.abs(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - abs.')
    ax.legend()
    ax.grid()

    # Eigenworms - arguments
    ax = axes[4]
    for i in args.plot_components:
        ax.plot(ts, np.angle(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - angle.')
    ax.legend()
    ax.grid()

    # Eigenworms - reals
    ax = axes[5]
    for i in args.plot_components:
        ax.plot(ts, np.real(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - real.')
    ax.legend()
    ax.grid()

    # Eigenworms - imag
    ax = axes[6]
    for i in args.plot_components:
        ax.plot(ts, np.imag(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - imag.')
    ax.legend()
    ax.grid()

    if x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}'
    fig.suptitle(title)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_traces_' \
                           f'r={reconstruction.id}_' \
                           f'f={args.start_frame}-{args.end_frame}_' \
                           f'nc={",".join([str(c) for c in args.plot_components])}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


def traces_condensed(x_label: str = 'time'):
    args = parse_args()
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=args.reconstruction,
        n_components=args.n_components,
        regenerate=False
    )

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': 25
    }

    y_max = 0.22
    y_min = -0.22

    regions = {
        'forwards_1': {
            'start': 13700,
            'end': 14135,
            'y1': 0.,
            'y2': y_max,
            'color': 'green'
        },
        'reversal': {
            'start': 14135,
            'end': 14185,
            'y1': y_min,
            'y2': y_max,
            'color': 'skyblue'
        },
        'backwards': {
            'start': 14185,
            'end': 14400,
            'y1': y_min,
            'y2': 0.,
            'color': 'red'
        },
        'reorientation': {
            'start': 14400,
            'end': 14550,
            'y1': y_min,
            'y2': y_max,
            'color': 'violet'
        },
        'forwards_2': {
            'start': 14550,
            'end': 14750,
            'y1': 0.,
            'y2': y_max,
            'color': 'green'
        },
    }

    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    X, meta = get_trajectory(**common_args)
    N = len(X)
    if x_label == 'time':
        ts = np.linspace(0, N / reconstruction.trial.fps, N)
    else:
        ts = np.arange(N) + (args.start_frame if args.start_frame is not None else 0)

    # Speed
    logger.info('Calculating speeds.')
    speeds = calculate_speeds(X, signed=True) * reconstruction.trial.fps

    # Planarity
    logger.info('Fetching planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
    nonp = pcas.nonp

    # Helicity
    logger.info('Calculating helicities.')
    H = calculate_helicities(X)

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    X_ew = ew.transform(np.array(Z))

    # Plot
    plt.rc('axes', labelsize=6)  # fontsize of the X label
    plt.rc('xtick', labelsize=5)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=5)  # fontsize of the legend
    plt.rc('xtick.major', pad=2, size=2)
    plt.rc('ytick.major', pad=2, size=2)

    # fig, axes = plt.subplots(2, figsize=(12, 8), sharex=True)
    fig = plt.figure(figsize=(5, 3.2))
    gs = GridSpec(
        nrows=3,
        ncols=1,
        hspace=0.05,
        left=0.07,
        right=0.94,
        top=0.99,
        bottom=0.07,
    )

    # Speeds
    ax_sp = fig.add_subplot(gs[0:2, 0])  # axes[0]

    for k, region in regions.items():
        if x_label == 'time':
            idxs = region['start'] - args.start_frame, region['end'] - args.start_frame
            x_r = [ts[idxs[0]], ts[idxs[1]]]
        else:
            x_r = [region['start'], region['end']]
        ax_sp.fill_between(
            x=x_r,
            y1=region['y1'],
            y2=region['y2'],
            color=region['color'],
            alpha=0.2,
            linewidth=0
        )

    ax_sp.axhline(y=0, color='darkgrey')
    ax_sp.plot(ts, speeds, color='black')
    ax_sp.set_ylabel('Speed (mm/s)', labelpad=1)
    ax_sp.set_xticks([])
    ax_sp.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])
    # ax.grid()
    ax_sp.set_ylim(bottom=y_min, top=y_max)
    ax_sp.set_xlim(left=ts[0], right=ts[-1])
    ax_sp.spines['right'].set_visible(False)

    # Planarities
    ax_nonp = ax_sp.twinx()
    ax_nonp.plot(ts, nonp, color='orange', alpha=0.6, linestyle='--')
    ax_nonp.set_ylabel('Non-planarity / Helicity', rotation=270, labelpad=8)
    # ax_nonp.axhline(y=0, color='darkgrey')
    ax_nonp.set_yticks([0, 0.1, 0.2])
    ax_nonp.set_yticklabels([0, 0.1, 0.2])
    ax_nonp.set_ylim(bottom=0, top=nonp.max() * 1.03)
    ax_nonp.spines['right'].set_linewidth(1.5)
    ax_nonp.spines['right'].set_linestyle((0, (3, 2)))
    ax_nonp.spines['right'].set_alpha(0.8)
    ax_nonp.spines['right'].set_color('orange')

    # Helicity
    ax_hel = ax_sp.twinx()
    ax_hel.set_yticks([])
    h_lim = np.abs(H).max() * 1.1
    ax_hel.set_ylim(bottom=-h_lim, top=h_lim)
    ax_hel.spines['right'].set_visible(False)
    # ax_hel.plot(ts, H, color='peru', linewidth=0.2)
    plot_helicities(
        ax=ax_hel,
        helicities=H,
        xs=ts,
        alpha_max=0.8,
        colour_pos='grey',
        colour_neg='grey',
    )

    # Eigenworms - absolute values
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    color_reordering = list(range(len(args.plot_components)))
    color_reordering[0:4] = [3, 1, 2, 0, 4]
    component_colours = [default_colours[i] for i in color_reordering]

    ax_lam = fig.add_subplot(gs[2, 0])
    for i in args.plot_components:
        for j, (k, region) in enumerate(regions.items()):
            idxs = region['start'] - args.start_frame, region['end'] - args.start_frame + 1
            if x_label == 'time':
                x_r = ts[idxs[0]:idxs[1]]
            else:
                x_r = np.arange(region['start'], region['end'] + 1)

            # vertical line to show start of region
            if args.vline and j > 0:
                ax_lam.axvline(x=x_r[0], color="k", alpha=0.25, linewidth=0.5)


            if k in ['forwards_1', 'forwards_2', 'backwards', 'reversal']:
                if i in [0, 1]:
                    alpha = 0.8
                    linewidth = 1.5
                    add_label = k == 'forwards_1'
                else:
                    alpha = 0.8 * 0.75
                    linewidth = 1.5 * 0.75
                    add_label = False
            elif k == 'reorientation':
                if i in [0, 1]:
                    alpha = 0.8 * 0.75
                    linewidth = 1.5 * 0.75
                    add_label = False
                else:
                    alpha = 0.8
                    linewidth = 1.5
                    add_label = True
            ax_lam.plot(
                x_r,
                np.abs(X_ew[idxs[0]:idxs[1], i]),
                color=component_colours[i],
                label=f'$\lambda_{i + 1}$' if add_label else None,
                alpha=alpha,
                linewidth=linewidth
            )

    ax_lam.set_ylabel('$|\lambda|$')
    ax_lam.legend(loc='upper right', handlelength=1, handletextpad=0.6, labelspacing=0.3,
                  borderpad=0.5)
    ax_lam.set_xlim(left=ts[0], right=ts[-1])
    ax_lam.set_yticks([0, 40, 80])
    ax_lam.set_yticklabels([0, 40, 80])

    if x_label == 'time':
        ax_lam.set_xlabel('Time (s)', labelpad=2)
    else:
        ax_lam.set_xlabel('Frame #', labelpad=2)

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_traces_condensed_' \
                           f'r={reconstruction.id}_' \
                           f'f={args.start_frame}-{args.end_frame}_' \
                           f'ew={ew.id}_' \
                           f'nc={",".join([str(c) for c in args.plot_components])}' \
                           f'.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


def heatmap():
    args = parse_args()
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=args.reconstruction,
        n_components=args.n_components,
        regenerate=False
    )
    scatter_size = 0.1
    n_mesh_points = 100

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
        'smoothing_window': 25
    }
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    X_ew = ew.transform(np.array(Z))

    # Interpolate data to make a surface
    def make_surface(x_, y_):
        xmin, xmax, ymin, ymax = min(x_), max(x_), min(y_), max(y_)
        X, Y = np.mgrid[xmin:xmax:complex(n_mesh_points), ymin:ymax:complex(n_mesh_points)]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x_, y_])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        return Z, xmin, xmax, ymin, ymax

    def plot_heatmap(ax, title, x_, y_):
        Z, x_min, x_max, y_min, y_max = make_surface(x_, y_)
        ax.set_title(title)
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max])
        ax.scatter(x_, y_, color='blue', s=scatter_size, alpha=0.5)
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 16))

    # ABS(c1) vs ABS(c2)
    x = np.abs(X_ew[:, 0])
    y = np.abs(X_ew[:, 1])
    plot_heatmap(axes[0, 0], 'ABS(c1) vs ABS(c2)', x, y)

    # ANGLE(c1) vs ANGLE(c2)
    x = np.angle(X_ew[:, 0])
    y = np.angle(X_ew[:, 1])
    plot_heatmap(axes[1, 0], 'ANGLE(c1) vs ANGLE(c2)', x, y)

    # RE(c1) vs IMAG(c1)
    x = np.real(X_ew[:, 0])
    y = np.imag(X_ew[:, 0])
    plot_heatmap(axes[0, 1], 'RE(c1) vs IMAG(c1)', x, y)

    # RE(c2) vs IMAG(c2)
    x = np.real(X_ew[:, 1])
    y = np.imag(X_ew[:, 1])
    plot_heatmap(axes[1, 1], 'RE(c2) vs IMAG(c2)', x, y)

    # RE(c1) vs IMAG(c2)
    x = np.real(X_ew[:, 0])
    y = np.imag(X_ew[:, 1])
    plot_heatmap(axes[0, 2], 'RE(c1) vs IMAG(c2)', x, y)

    # RE(c2) vs IMAG(c2)
    x = np.real(X_ew[:, 1])
    y = np.imag(X_ew[:, 0])
    plot_heatmap(axes[1, 2], 'RE(c2) vs IMAG(c1)', x, y)

    title = f'Trial={reconstruction.trial.id}. Reconstruction={reconstruction.id}.'
    if args.start_frame is not None:
        title += f'\nFrames={args.start_frame}-{args.end_frame}'
    fig.suptitle(title)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_heatmap_r={reconstruction.id}_f={args.start_frame}-{args.end_frame}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()


def heatmap_basic():
    args = parse_args()
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=args.reconstruction,
        n_components=args.n_components,
        regenerate=False
    )
    n_mesh_points = 100

    common_args = {
        'reconstruction_id': args.reconstruction,
        # 'start_frame': args.start_frame,
        # 'end_frame': args.end_frame,
        'smoothing_window': 25,
        'natural_frame': True,
        'rebuild_cache': False,
    }
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)

    # Eigenworms embeddings for entire clip
    logger.info('Fetching eigenworms embeddings for entire clip.')
    Z_all, meta = get_trajectory(**common_args)
    X_ew_all = ew.transform(np.array(Z_all))

    # Eigenworms embeddings just for requested frames
    logger.info('Fetching eigenworms embeddings for requested frames.')
    Z_traj, meta = get_trajectory(**common_args, start_frame=args.start_frame, end_frame=args.end_frame)
    X_ew_traj = ew.transform(np.array(Z_traj))

    # Construct colours
    colours = np.linspace(0, 1, len(X_ew_traj))
    cmap = plt.get_cmap('winter_r')

    # Interpolate data to make a surface
    def make_surface(x_, y_):
        xmin, xmax, ymin, ymax = min(x_), max(x_), min(y_), max(y_)
        X, Y = np.mgrid[xmin:xmax:complex(n_mesh_points), ymin:ymax:complex(n_mesh_points)]
        positions = np.vstack([X.ravel(), Y.ravel()])
        values = np.vstack([x_, y_])
        kernel = gaussian_kde(values)
        Z = np.reshape(kernel(positions).T, X.shape)
        return Z, xmin, xmax, ymin, ymax

    def plot_heatmap(ax, c, x_all_, y_all_, x_traj_, y_traj_):
        Z, x_min, x_max, y_min, y_max = make_surface(x_all_, y_all_)
        ax.set_ylabel(f'Re($\lambda_{c}$)')
        ax.set_xlabel(f'Im($\lambda_{c}$)')

        # Add the heatmap
        ax.imshow(np.rot90(Z), cmap=plt.get_cmap('YlOrRd'), extent=[x_min, x_max, y_min, y_max])

        # Overlay the trajectory
        X = np.stack([x_traj_, y_traj_], axis=1)
        points = X[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, array=colours, cmap=cmap, alpha=colours)
        ax.add_collection(lc)

        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_yticks([])
        ax.set_xticks([])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(4, 2))

    # RE(c1) vs IMAG(c1)
    logger.info('Plotting RE(c1) vs IMAG(c1).')
    x_all = np.real(X_ew_all[:, 0])
    y_all = np.imag(X_ew_all[:, 0])
    x_traj = np.real(X_ew_traj[:, 0])
    y_traj = np.imag(X_ew_traj[:, 0])
    plot_heatmap(axes[0], 1, x_all, y_all, x_traj, y_traj)

    # RE(c2) vs IMAG(c2)
    logger.info('Plotting RE(c2) vs IMAG(c2).')
    x_all = np.real(X_ew_all[:, 1])
    y_all = np.imag(X_ew_all[:, 1])
    x_traj = np.real(X_ew_traj[:, 1])
    y_traj = np.imag(X_ew_traj[:, 1])
    plot_heatmap(axes[1], 2, x_all, y_all, x_traj, y_traj)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_heatmap_basic' \
                           f'_r={reconstruction.id}' \
                           f'_f={args.start_frame}-{args.end_frame}' \
                           f'_ew={ew.id}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)

    if show_plots:
        plt.show()


def animate():
    args = parse_args()
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=args.reconstruction,
        n_components=args.n_components,
        regenerate=False
    )

    common_args = {
        'reconstruction_id': args.reconstruction,
        'start_frame': args.start_frame,
        'end_frame': args.end_frame,
    }
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    X_ew = ew.transform(np.array(Z))

    # Reconstruct from embedding
    NFs = []
    Xs = []
    nf_prev = None
    exclude_component_idxs = list(set(list(range(ew.n_components))) - set(args.plot_components))
    for i, X_embedding in enumerate(X_ew):
        X_embedding[exclude_component_idxs] = 0
        Zi = ew.inverse_transform(X_embedding)
        nf = NaturalFrame(Zi)
        NFs.append(nf)
        X = nf.X_pos - nf.X_pos.mean(axis=0)

        if nf_prev is not None:
            v1 = nf.pca.components_[0]
            v2 = NFs[0].pca.components_[0]
            R1 = calculate_rotation_matrix(v1, v2)
            R2 = calculate_rotation_matrix(-v1, v2)
            Xr1 = np.dot(X, R1.T)
            Xr2 = np.dot(X, R2.T)

            # Pick best aligned
            hhd1 = np.linalg.norm(Xr1[0] - Xs[-1][0])
            hhd2 = np.linalg.norm(Xr2[0] - Xs[-1][0])
            if hhd1 < hhd2:
                X = Xr1
            else:
                X = Xr2

        nf_prev = nf
        Xs.append(X)
    Xs = np.array(Xs).transpose(0, 2, 1)

    FS = FrameSequenceNumpy(x=Xs)
    generate_interactive_scatter_clip(FS, fps=25)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    # interactive()
    # traces(x_label='frames')
    traces_condensed(x_label='time')
    # heatmap()
    # heatmap_basic()
    # animate()
