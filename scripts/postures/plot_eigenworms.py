import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.gridspec import GridSpec

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import LOGS_PATH, logger, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction
from wormlab3d.postures.cpca import CPCA
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d
from wormlab3d.toolkit.util import parse_target_arguments
from wormlab3d.trajectories.cache import get_trajectory

plot_n_components = 4
show_plots = False
save_plots = True
img_extension = 'png'
eigenworm_length = 1
eigenworm_scale = 64
cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)


def _get_reconstruction() -> Reconstruction:
    """
    Find a reconstruction by id.
    """
    args = parse_target_arguments()
    if args.reconstruction is None:
        raise RuntimeError('This script must be run with the --reconstruction=ID argument defined.')
    return Reconstruction.objects.get(id=args.reconstruction)


def _calculate_basis(
        reconstruction: Reconstruction,
        n_components: int
) -> Tuple[CPCA, np.ndarray]:
    X_full, meta = get_trajectory(reconstruction_id=reconstruction.id)
    M = X_full.shape[0]
    N = X_full.shape[1]
    if N == 1:
        raise RuntimeError('Trajectory returned only a single point. Eigenworms requires full postures!')
    logger.info(f'Loaded {M} midlines of length {N}.')

    # Calculate the natural frame representations for all midlines
    logger.info('Calculating natural frame representations.')
    Z = []
    bad_idxs = []
    for i, X in enumerate(X_full):
        if (i + 1) % 100 == 0:
            logger.info(f'Calculating for midline {i + 1}/{M}.')
        try:
            nf = NaturalFrame(X)
            Z.append(nf.mc)
        except Exception:
            bad_idxs.append(i)
    if len(bad_idxs):
        logger.warning(f'Failed to calculate {len(bad_idxs)} midline idxs: {bad_idxs}.')

    # Calculate CPCA components
    logger.info('Calculating basis.')
    Z = np.array(Z)
    cpca = CPCA(n_components=n_components, whiten=False)
    cpca.fit(Z)

    return cpca, X_full


def _plot_eigenworms(
        cpca: CPCA,
        title: str,
        filename: str
):
    n_rows = 3
    n_cols = plot_n_components
    fig = plt.figure(figsize=(n_cols * 3, n_rows * 3))
    gs = GridSpec(n_rows, n_cols)

    for i in range(plot_n_components):
        component = cpca.components_[i]
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
        cpca: CPCA,
        title: str,
        filename: str
):
    fig, axes = plt.subplots(2)
    ax = axes[0]
    ax.set_title('Explained variance')
    ax.plot(np.cumsum(cpca.explained_variance_))
    ax = axes[1]
    ax.set_title('Explained variance ratio')
    ax.plot(np.cumsum(cpca.explained_variance_ratio_))

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
        cpca: CPCA,
        X_full: np.ndarray,
        idx: int,
        title: str,
        filename: str
):
    # Reconstruct a worm using the basis
    X = X_full[idx]
    NF_original = NaturalFrame(X)
    coeffs = cpca.transform(np.array([NF_original.mc]))
    mc = cpca.inverse_transform(coeffs)
    NF_reconstructed = NaturalFrame(
        mc[0],
        X0=NF_original.X_pos[0],
        T0=NF_original.T[0],
        M0=NF_original.M1[0],
    )

    NF1_args = {'c': 'red', 'linestyle': '--', 'alpha': 0.8}
    NF2_args = {'c': 'blue', 'linestyle': ':', 'alpha': 0.8}
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
    ax.plot(ind, NF_original.kappa, **NF1_args)
    ax.plot(ind, NF_reconstructed.kappa, **NF2_args)
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])

    # Psi
    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    ax.plot(NF_original.psi, ind, **NF1_args)
    ax.plot(NF_reconstructed.psi, ind, **NF2_args)
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
    reconstruction = _get_reconstruction()
    cpca, X_full = _calculate_basis(reconstruction, n_components=20)

    title = f'Trial {reconstruction.trial.id}.\n' \
            f'Reconstruction={reconstruction.id} ({reconstruction.source}).\n' \
            f'Num worms={cpca.n_samples_}. ' \
            f'Num points={cpca.n_features_}.'

    filename = f'trial={reconstruction.trial.id:03d}_' \
               f'reconstruction={reconstruction.id}_{reconstruction.source}_' \
               f'M={cpca.n_samples_}_N={cpca.n_features_}'

    _plot_eigenworms(cpca, title, filename)
    _plot_eigenvalues(cpca, title, filename)
    _plot_reconstruction(cpca, X_full, 0, title, filename)


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    main()
