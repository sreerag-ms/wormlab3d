import os
from argparse import ArgumentParser
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from simple_worm.frame import FrameSequenceNumpy
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

show_plots = True
save_plots = True
img_extension = 'png'


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


def traces():
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
    ts = np.linspace(0, N / reconstruction.trial.fps, N)

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

    ax.set_xlabel('Time (s)')

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
    from simple_worm.plot3d import interactive, generate_interactive_scatter_clip

    interactive()
    # traces()
    # heatmap()
    animate()
