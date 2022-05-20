import os
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
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import calculate_speeds, calculate_rotation_matrix

# tex_mode()

show_plots = False
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
    r = pcas.explained_variance_ratio.T
    nonp = r[2] / np.sqrt(r[1] * r[0])

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    X_ew = ew.transform(np.array(Z))

    # Plot
    fig, axes = plt.subplots(6, figsize=(16, 16), sharex=True)

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

    # Eigenworms - absolute values
    ax = axes[2]
    for i in args.plot_components:
        ax.plot(ts, np.abs(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - abs.')
    ax.legend()
    ax.grid()

    # Eigenworms - arguments
    ax = axes[3]
    for i in args.plot_components:
        ax.plot(ts, np.angle(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - angle.')
    ax.legend()
    ax.grid()

    # Eigenworms - reals
    ax = axes[4]
    for i in args.plot_components:
        ax.plot(ts, np.real(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('Component contribution')
    ax.set_title('Eigenworms - real.')
    ax.legend()
    ax.grid()

    # Eigenworms - imag
    ax = axes[5]
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

    regions = {
        'forwards_1': {
            'start': 13700,
            'end': 14135,
            'y1': 0.,
            'y2': 0.008 * 25,
            'color': 'green'
        },
        'reversal': {
            'start': 14135,
            'end': 14185,
            'y1': -0.008 * 25,
            'y2': 0.008 * 25,
            'color': 'skyblue'
        },
        'backwards': {
            'start': 14185,
            'end': 14375,
            'y1': -0.008 * 25,
            'y2': 0.,
            'color': 'red'
        },
        'reorientation': {
            'start': 14375,
            'end': 14550,
            'y1': -0.008 * 25,
            'y2': 0.008 * 25,
            'color': 'violet'
        },
        'forwards_2': {
            'start': 14550,
            'end': 14750,
            'y1': 0.,
            'y2': 0.008 * 25,
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
    r = pcas.explained_variance_ratio.T
    nonp = r[2] / np.sqrt(r[1] * r[0])

    # Eigenworms embeddings
    Z, meta = get_trajectory(**common_args, natural_frame=True, rebuild_cache=False)
    X_ew = ew.transform(np.array(Z))

    # Plot
    # fig, axes = plt.subplots(2, figsize=(12, 8), sharex=True)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(nrows=3, ncols=1)

    # Speeds
    ax = fig.add_subplot(gs[0:2, 0])  # axes[0]

    for k, region in regions.items():
        if x_label == 'time':
            idxs = region['start'] - args.start_frame, region['end'] - args.start_frame
            x_r = [ts[idxs[0]], ts[idxs[1]]]
        else:
            x_r = [region['start'], region['end']]
        ax.fill_between(
            x=x_r,
            y1=region['y1'],
            y2=region['y2'],
            color=region['color'],
            alpha=0.2,
            linewidth=0
        )

    ax.axhline(y=0, color='darkgrey')
    ax.plot(ts, speeds)
    ax.set_ylabel('Speed (mm/s)')
    # ax.grid()

    # ax.set_yticks([])

    # Planarities
    ax2 = ax.twinx()
    ax2.plot(ts, nonp, color='orange', alpha=0.6, linestyle='--')
    ax2.set_ylabel('Non-planarity', rotation=270, labelpad=15)
    ax2.axhline(y=0, color='darkgrey')
    ax2.set_yticks([0, 0.1, 0.2])
    ax2.set_yticklabels([0, 0.1, 0.2])

    # Eigenworms - absolute values
    prop_cycle = plt.rcParams['axes.prop_cycle']
    default_colours = prop_cycle.by_key()['color']
    component_colours = [default_colours[i] for i in range(len(args.plot_components))]

    ax = fig.add_subplot(gs[2, 0])  # axes[1]
    for i in args.plot_components:
        for j, (k, region) in enumerate(regions.items()):
            idxs = region['start'] - args.start_frame, region['end'] - args.start_frame + 1
            if x_label == 'time':
                x_r = ts[idxs[0]:idxs[1]]
            else:
                x_r = np.arange(region['start'], region['end'] + 1)

            if k in ['forwards_1', 'forwards_2', 'backwards', 'reversal']:
                if i in [0, 1]:
                    alpha = 0.8
                    linewidth = 2
                    add_label = k == 'forwards_1'
                else:
                    alpha = 0.25
                    linewidth = 1
                    add_label = False
            elif k == 'reorientation':
                if i in [0, 1]:
                    alpha = 0.25
                    linewidth = 1
                    add_label = False
                else:
                    alpha = 0.8
                    linewidth = 2
                    add_label = True
            ax.plot(
                x_r,
                np.abs(X_ew[idxs[0]:idxs[1], i]),
                color=component_colours[i],
                label=f'$\lambda_{i + 1}$' if add_label else None,
                alpha=alpha,
                linewidth=linewidth
            )

        # ax.plot(ts, np.abs(X_ew[:, i]), label=i, alpha=0.7)
    ax.set_ylabel('$|\lambda|$')
    ax.legend()
    # ax.grid()

    ax.set_yticks([0, 40, 80])
    ax.set_yticklabels([0, 40, 80])

    if x_label == 'time':
        ax.set_xlabel('Time (s)')
    else:
        ax.set_xlabel('Frame #')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_traces_condensed_' \
                           f'r={reconstruction.id}_' \
                           f'f={args.start_frame}-{args.end_frame}_' \
                           f'nc={",".join([str(c) for c in args.plot_components])}.{img_extension}'
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
        path = LOGS_PATH / f'{START_TIMESTAMP}_heatmap_basic_r={reconstruction.id}_f={args.start_frame}-{args.end_frame}.{img_extension}'
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
    from simple_worm.plot3d import interactive
    interactive()
    # traces(x_label='frames')
    # traces_condensed(x_label='time')
    # heatmap()
    heatmap_basic()
    # animate()
