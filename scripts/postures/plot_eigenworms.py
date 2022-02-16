import os
from argparse import Namespace, ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Eigenworms, Dataset
from wormlab3d.postures.cpca import load_cpca_from_file
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.cache import get_trajectory

plot_n_components = 5
show_plots = True
save_plots = True
img_extension = 'png'
eigenworm_length = 1
eigenworm_scale = 64
cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to plot an eigenworm basis.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--n-components', type=int, default=10,
                        help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--cpca-file', type=str,
                        help='Load CPCA from file.')
    args = parser.parse_args()

    targets = np.array([getattr(args, k) is not None for k in ['reconstruction', 'dataset', 'cpca_file']],
                       dtype=np.bool)
    assert targets.sum() == 1, 'One of --reconstruction, --dataset or --cpca-file must be defined.'

    print_args(args)

    return args


def _plot_eigenworms(
        eigenworms: Eigenworms,
        title: str,
        filename: str
):
    n_rows = 3
    n_cols = plot_n_components
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = GridSpec(n_rows, n_cols)

    for i in range(plot_n_components):
        component = eigenworms.components[i]
        NF = NaturalFrame(component * eigenworm_scale, length=eigenworm_length)
        N = NF.N
        ind = np.arange(N)
        fc = cmap((np.arange(N) + 0.5) / N)

        # 3D plot of eigenworm
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax = plot_natural_frame_3d(
            NF,
            show_frame_arrows=True,
            n_frame_arrows=20,
            arrow_scale=0.2,
            show_pca_arrows=False,
            ax=ax
        )
        ax.set_title(f'Component = {i}')

        # Psi polar plot
        ax = fig.add_subplot(gs[1, i], projection='polar')
        ax.set_title('$\psi=arg(m_1+i m_2)$')
        for j in range(N - 1):
            ax.plot(NF.psi[j:j + 2], ind[j:j + 2], c=fc[j])
        ax.set_rticks([])
        thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
        ax.set_xticks(thetaticks)
        ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
        ax.xaxis.set_tick_params(pad=-3)

        # Curvature plot
        ax = fig.add_subplot(gs[2, i])
        if i == 0:
            kappa_share_ax = ax
        else:
            ax.sharey(kappa_share_ax)
        ax.set_title('$|\kappa|=|m_1|+|m_2|$')
        for j in range(N - 1):
            ax.plot(ind[j:j + 2], NF.kappa[j:j + 2], c=fc[j])
        ax.set_xticks([0, ind[-1]])
        ax.set_xticklabels(['H', 'T'])

    fig.suptitle(title.replace('.\n', '. '))
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigenworms_{filename}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_eigenvalues(
        eigenworms: Eigenworms,
        title: str,
        filename: str
):
    fig, axes = plt.subplots(2)
    ax = axes[0]
    ax.set_title('Explained variance')
    ax.plot(np.cumsum(eigenworms.explained_variance))
    ax = axes[1]
    ax.set_title('Explained variance ratio')
    ax.plot(np.cumsum(eigenworms.explained_variance_ratio))

    fig.suptitle(title)
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigenvalues_{filename}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    plt.close(fig)


def _plot_reconstruction(
        eigenworms: Eigenworms,
        X_full: np.ndarray,
        idx: int,
        title: str,
        filename: str
):
    # Reconstruct a worm using the basis
    X = X_full[idx]
    NF_original = NaturalFrame(X)
    coeffs = eigenworms.transform(np.array([NF_original.mc]))
    mc = eigenworms.inverse_transform(coeffs)
    NF_reconstructed = NaturalFrame(
        mc[0],
        X0=NF_original.X_pos[0],
        T0=NF_original.T[0],
        M0=NF_original.M1[0],
    )

    NF_original_args = {'c': 'red', 'linestyle': '--', 'alpha': 0.8, 'label': 'Original'}
    NF_reconst_args = {'c': 'blue', 'linestyle': ':', 'alpha': 0.8, 'label': 'Reconstructed'}
    N = NF_original.N
    ind = np.arange(N)

    fig = plt.figure(figsize=(10, 12))
    gs = GridSpec(3, 2)
    gs2 = GridSpec(3, 2, wspace=0.25, hspace=0.2, left=0.1, right=0.95, bottom=0.05, top=0.9)

    # 3D plots of eigenworms
    for i, NF in enumerate([NF_original, NF_reconstructed]):
        ax = fig.add_subplot(gs[0, i], projection='3d')
        ax = plot_natural_frame_3d(
            NF,
            show_frame_arrows=True,
            n_frame_arrows=20,
            arrow_scale=0.2,
            show_pca_arrows=False,
            ax=ax
        )
        ax.set_title(['Original', 'Reconstruction'][i])

    # Kappa
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('$|\kappa|=|m_1|+|m_2|$')
    ax.plot(ind, NF_original.kappa, **NF_original_args)
    ax.plot(ind, NF_reconstructed.kappa, **NF_reconst_args)
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])
    ax.legend()

    # Psi
    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    ax.plot(NF_original.psi, ind, **NF_original_args)
    ax.plot(NF_reconstructed.psi, ind, **NF_reconst_args)
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    # Coefficients
    ax = fig.add_subplot(gs[2, :])
    ax.set_title('Coefficient magnitudes')
    ax.plot(np.abs(coeffs)[0])

    fig.suptitle(title + f' Idx={idx}.')
    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_eigen-reconstruction_{filename}_idx={idx}.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path)

    if show_plots:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()
    X_full = None

    if args.dataset is not None:
        ew = generate_or_load_eigenworms(
            dataset_id=args.dataset,
            n_components=args.n_components,
            regenerate=False
        )
        dataset = Dataset.objects.get(id=args.dataset)
        X_full = dataset.X_all

        title = f'Dataset {dataset.id}.'
        filename = f'ds={dataset.id}'

    elif args.reconstruction is not None:
        ew = generate_or_load_eigenworms(
            reconstruction_id=args.reconstruction,
            n_components=args.n_components,
            regenerate=False
        )
        reconstruction = Reconstruction.objects.get(id=args.reconstruction)
        X_full, meta = get_trajectory(reconstruction_id=args.reconstruction)

        title = f'Trial {reconstruction.trial.id}.\n' \
                f'Reconstruction={reconstruction.id} ({reconstruction.source}).'

        filename = f'trial={reconstruction.trial.id:03d}_' \
                   f'reconstruction={reconstruction.id}_{reconstruction.source}'

    elif args.cpca_file is not None:
        path = Path(args.cpca_file)
        fn = path.parts[-1]
        assert path.exists(), 'CPCA file not found!'
        ew = Eigenworms()
        ew.cpca = load_cpca_from_file(path)
        ew.components = ew.cpca.components_
        ew.n_samples = ew.cpca.n_samples_
        ew.n_features = ew.cpca.n_features_
        ew.n_components = ew.cpca.n_components_
        title = f'Basis: {fn}.'
        filename = f'basis={fn}'

    title += f'\nNum worms={ew.n_samples}. Num points={ew.n_features}.'
    filename += f'_M={ew.n_samples}_N={ew.n_features}'

    _plot_eigenworms(ew, title, filename)
    _plot_eigenvalues(ew, title, filename)
    if X_full is not None:
        _plot_reconstruction(ew, X_full, 500, title, filename)


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    main()
