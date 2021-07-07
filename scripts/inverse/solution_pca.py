import os
import time
from typing import List, Tuple

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.polynomial.polynomial import Polynomial
from sklearn.decomposition import PCA

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import cla, FrameArtist, interactive
from wormlab3d import LOGS_PATH
from wormlab3d.data.model import SwCheckpoint, SwRun
from wormlab3d.toolkit.util import parse_target_arguments

START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')


def get_checkpoint() -> SwCheckpoint:
    """
    Find a checkpoint by id.
    """
    args = parse_target_arguments()
    if args.sw_checkpoint is None:
        raise RuntimeError('This script must be run with the --sw-checkpoint=ID argument defined.')
    return SwCheckpoint.objects.get(id=args.sw_checkpoint)


def calculate_pca(
        runs: List[SwRun],
        canonical: bool,
        include_gamma: bool,
        variance_captured: float
) -> Tuple[PCA, np.ndarray]:
    bs = len(runs)

    # Extract and flatten controls from the runs
    psi0 = np.array([run.F0.psi for run in runs])
    alpha = np.array([run.CS.alpha for run in runs])
    beta = np.array([run.CS.beta for run in runs])
    gamma = np.array([run.CS.gamma for run in runs])
    alpha_vec = alpha.reshape((bs, -1))
    beta_vec = beta.reshape((bs, -1))
    gamma_vec = gamma.reshape((bs, -1))

    # Generate rotated "equivalent" solutions
    if include_gamma:
        solutions0 = np.concatenate([psi0, alpha_vec, beta_vec, gamma_vec], axis=1)  # default (0-rotation)
        solutions1 = np.concatenate([psi0, beta_vec, -alpha_vec, gamma_vec], axis=1)  # +pi/2 rotation
        solutions2 = np.concatenate([psi0, -alpha_vec, -beta_vec, gamma_vec], axis=1)  # +pi rotation
        solutions3 = np.concatenate([psi0, -beta_vec, alpha_vec, gamma_vec], axis=1)  # +3pi/2 rotation
    else:
        solutions0 = np.concatenate([psi0, alpha_vec, beta_vec], axis=1)  # default (0-rotation)
        solutions1 = np.concatenate([psi0, beta_vec, -alpha_vec], axis=1)  # +pi/2 rotation
        solutions2 = np.concatenate([psi0, -alpha_vec, -beta_vec], axis=1)  # +pi rotation
        solutions3 = np.concatenate([psi0, -beta_vec, alpha_vec], axis=1)  # +3pi/2 rotation

    # Combine solutions
    solutions = np.zeros((4 * bs, solutions0.shape[-1]))
    solutions[0::4] = solutions0
    if canonical:
        solutions[1::4] = solutions1
        solutions[2::4] = solutions2
        solutions[3::4] = solutions3
    else:
        solutions[1::4] = solutions0
        solutions[2::4] = solutions0
        solutions[3::4] = solutions0

    # PCA
    pca = PCA(svd_solver='full', copy=False, n_components=variance_captured)
    pca.fit(solutions)
    embeddings = pca.transform(solutions0)

    return pca, embeddings


def solution_pca(
        canonical: bool = True,
        include_gamma: bool = False,
        variance_to_capture: float = 0.99,
        show_pca: bool = True,
        save_pca: bool = False,
        show_animation: bool = False,
        save_animation: bool = False,
        fps: int = 25
):
    """
    Loads a checkpoint and does some PCA analysis on the results found across the different runs.
    """
    checkpoint = get_checkpoint()
    runs = checkpoint.get_runs()
    bs = len(runs)
    N = checkpoint.sim_params.worm_length
    T = int(checkpoint.sim_params.duration / checkpoint.sim_params.dt)
    pca, embeddings = calculate_pca(runs, canonical, include_gamma, variance_to_capture)
    save_dir = LOGS_PATH + f'/{START_TIMESTAMP}_checkpoint={checkpoint.id}_canonical={canonical}_gamma={include_gamma}_variance={variance_to_capture}'
    interactive()

    # Fit quadratic curves
    polys = []
    for i, j in [[0, 1], [0, 2], [1, 2]]:
        poly = Polynomial.fit(embeddings[:, i], embeddings[:, j], deg=2)
        polys.append(poly)

    # Plot components
    if show_pca or save_pca:
        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(3, 3)
        fig.suptitle(
            f'including gamma = {include_gamma}, canonical={canonical}, captured variance = {variance_to_capture}')
        cmap = plt.get_cmap('rainbow')

        # Plot the singular values
        ind = np.arange(pca.n_components_)
        ax = fig.add_subplot(gs[0, 0])
        ax.bar(ind, pca.singular_values_, align='center')
        ax.set_xticks(ind)
        ax.set_title('PCA on solutions')
        ax.set_xlabel('Singular value')

        # Show the distributions for the samples
        ax = fig.add_subplot(gs[0, 1])
        for i, embedding in enumerate(embeddings):
            ax.plot(embedding, color=cmap((i + 0.5) / bs), label=i)
        if bs < 10:
            ax.legend()
        ax.set_title('Sample distribution')
        ax.set_xlabel('Singular value')
        ax.set_ylabel('Contribution')

        # Show average psi on a straight midline
        x0 = np.zeros((3, N))
        x0[:][0] = np.linspace(1, 0, N, endpoint=True)
        psi = pca.mean_[:N]
        F0 = FrameNumpy(x=x0, psi=psi, calculate_components=True)
        ax = fig.add_subplot(gs[0, 2], projection='3d')
        cla(ax)

        # Add frame arrows and midline
        fa = FrameArtist(F0, arrow_scale=0.35, n_arrows=N // 2)
        fa.add_midline(ax)
        fa.add_component_vectors(ax, draw_e0=False)

        # Fix axes range
        mins, maxs = F0.get_bounding_box()
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        # Plot the components as matrices
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            ax.axis('off')
            if i >= pca.components_.shape[0]:
                continue

            component = pca.components_[i]
            c = np.r_[component[N:], np.zeros(T)].reshape((3 * T, N)).T
            ax.matshow(c, aspect='auto')
            ax.set_title(f'Component = {i}')

        # Scatter the correlations between singular values
        fc = cmap((np.arange(bs) + 0.5) / bs)
        for col_idx, (i, j) in enumerate([[0, 1], [0, 2], [1, 2]]):
            ax = fig.add_subplot(gs[2, col_idx])
            ax.scatter(embeddings[:, i], embeddings[:, j], c=fc)
            ax.set_title('Sample distribution')
            ax.set_xlabel(f'Singular value {i}')
            ax.set_ylabel(f'Singular value {j}')

            # Add fitted quadratic curve
            xs = np.linspace(min(embeddings[:, i]), max(embeddings[:, i]), 1000)
            ax.plot(xs, polys[col_idx](xs))

        fig.tight_layout()

        if show_pca:
            plt.show()
        if save_pca:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_dir + '/pca.svg')

    # Create animated plot showing transition along polynomial fitted correlation curves
    if show_animation or save_animation:
        n_points = 100
        sv0 = np.linspace(min(embeddings[:, 0]), max(embeddings[:, 0]), n_points)
        sv1 = polys[0](sv0)
        sv2 = polys[1](sv0)
        svs = np.array([sv0, sv1, sv2])

        # Inverse transform
        X = pca.inverse_transform(np.r_[svs, np.zeros((3, n_points))].T)
        C = np.c_[X[:, N:], np.zeros((n_points, T))].reshape((n_points, 3 * T, N)).transpose(0, 2, 1)
        psi = X[:, :N]

        # Calculate frames
        x0 = np.zeros((3, N))
        x0[:][0] = np.linspace(1, 0, N, endpoint=True)
        Fs = [FrameNumpy(x=x0, psi=psi[k], calculate_components=True) for k in range(n_points)]

        # Extend data so it plays in a reflecting loop
        svs = np.concatenate([svs, np.fliplr(svs)], axis=1)
        C = np.concatenate([C, np.flipud(C)])
        Fs = Fs + Fs[::-1]

        fig = plt.figure(figsize=(10, 10))
        gs = GridSpec(3, 4)

        # Plot the curves
        scats = []
        for col_idx, (i, j) in enumerate([[0, 1], [0, 2]]):
            ax = fig.add_subplot(gs[0, col_idx * 2:col_idx * 2 + 2])
            ax.set_xlabel(f'Singular value {i}')
            ax.set_ylabel(f'Singular value {j}')

            # Add fitted quadratic curve
            xs = np.linspace(min(embeddings[:, i]), max(embeddings[:, i]), 1000)
            ax.plot(xs, polys[col_idx](xs), zorder=-1)

            # Add solution indicator
            scat = ax.scatter(svs[i][0], svs[j][0], c='red', marker='x', s=50)
            scats.append(scat)

        # Plot the solution
        ax = fig.add_subplot(gs[1:3, 0:2])
        ax.axis('off')
        ax.set_title('Solution')
        sol = ax.matshow(C[0], aspect='auto')

        # Plot frame on a straight midline
        ax = fig.add_subplot(gs[1:3, 2:4], projection='3d')
        cla(ax)
        ax.set_title('Frame')
        fa = FrameArtist(Fs[0], arrow_scale=0.35, n_arrows=N // 2)
        fa.add_midline(ax)
        fa.add_component_vectors(ax, draw_e0=False)

        # Fix axes range
        mins, maxs = Fs[0].get_bounding_box()
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

        def update(k):
            for col_idx, (i, j) in enumerate([[0, 1], [0, 2]]):
                scats[col_idx].set_offsets((svs[i][k], svs[j][k]))
            sol.set_array(C[k])
            fa.update(Fs[k])
            return ()

        fig.tight_layout()

        ani = manimation.FuncAnimation(
            fig,
            update,
            frames=n_points * 2,
            blit=True,
            interval=1 / fps
        )

        if show_animation:
            plt.show()
        if save_animation:
            os.makedirs(save_dir, exist_ok=True)
            metadata = dict(
                title=f'{START_TIMESTAMP}_checkpoint={checkpoint.id}_canonical={canonical}_gamma={include_gamma}_variance={variance_to_capture}',
                artist='WormLab Leeds'
            )
            ani.save(save_dir + f'/animation_{fps}fps.mp4', writer='ffmpeg', fps=fps, metadata=metadata)


if __name__ == '__main__':
    solution_pca(
        canonical=False,
        include_gamma=True,
        show_pca=False,
        save_pca=True,
        show_animation=False,
        save_animation=True,
        fps=25
    )
