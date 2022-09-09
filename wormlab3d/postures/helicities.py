from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from numpy.lib.stride_tricks import sliding_window_view

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT, Arrow3D
from wormlab3d import N_WORKERS, logger
from wormlab3d.postures.natural_frame import NaturalFrame, normalise, EPS, orthogonalise
from wormlab3d.postures.plot_utils import plot_natural_frame_3d


def calculate_helicity(X: np.ndarray) -> float:
    """
    Calculate the helicity of a curve.
    """
    NF = NaturalFrame(X)
    return NF.helicity()


def _calculate_helicities_parallel(X: np.ndarray) -> np.ndarray:
    """
    Calculate helicities in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_helicity,
            [Xi for Xi in X]
        )
    return np.array(res)


def calculate_helicities(X: np.ndarray, parallel: bool = True) -> np.ndarray:
    """
    Calculate the helicity metric for all the postures.
    """
    assert X.ndim == 3, 'Full postures required!'
    if parallel and N_WORKERS > 1:
        h = _calculate_helicities_parallel(X)
    else:
        h = np.zeros(len(X))
        for i, Xi in enumerate(X):
            h[i] = calculate_helicity(Xi)
    return h


def calculate_trajectory_helicities(
        X: np.ndarray,
        window_size: int,
        match_length: bool = True,
        parallel: bool = True
) -> np.ndarray:
    """
    Calculate helicities along a trajectory.
    """
    logger.info(f'Calculating trajectory helicities for window size={window_size}.')
    assert X.ndim == 2, 'Only point trajectories supported!'
    Xw = sliding_window_view(X, window_size, axis=0).transpose(0, 2, 1)
    Hw = calculate_helicities(Xw, parallel=parallel)

    if match_length:
        N = len(X)
        w2 = int((window_size + 1) / 2)
        h = np.zeros(N)
        h[w2 - 1:N - window_size + w2] = Hw

        # Fill in the ends
        for i in range(w2 - 1):
            ws = window_size - i - 1
            if ws < 5:
                break
            h[w2 - 2 - i] = calculate_helicity(X[:ws])
            h[-w2 + i] = calculate_helicity(X[-ws:])
    else:
        h = Hw

    return h


def plot_helicities(
        ax: Axes,
        helicities: np.ndarray,
        xs: np.ndarray = None,
        alpha_max: int = 1,
        n_fade_lines: int = 100
):
    """
    Helper method to plot the helicities on an axes.
    """
    fade_lines_pos = np.linspace(0, helicities.max(), n_fade_lines)
    fade_lines_neg = np.linspace(0, helicities.min(), n_fade_lines)
    if xs is None:
        xs = np.arange(len(helicities))
    for i in range(n_fade_lines):
        ax.fill_between(
            xs,
            np.ones_like(helicities) * fade_lines_pos[i],
            helicities,
            where=helicities > fade_lines_pos[i],
            color='purple',
            alpha=alpha_max / n_fade_lines,
            linewidth=0,
        )
        ax.fill_between(
            xs,
            helicities,
            np.ones_like(helicities) * fade_lines_neg[i],
            where=helicities < fade_lines_neg[i],
            color='green',
            alpha=alpha_max / n_fade_lines,
            linewidth=0,
        )


def illustrate_helicity_method(NF: NaturalFrame, normalise_curve: bool = True) -> Figure:
    """
    Illustrate the method.
    """
    N = NF.N
    s = np.linspace(0, 1, N)
    if normalise_curve:
        NF = NaturalFrame(NF.X_pos / NF.length)
    X = NF.X_pos.copy()

    cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2)

    # Plot the 3d curve
    ax = fig.add_subplot(gs[0:2, 0], projection='3d')
    ax.set_title('Original curve')
    plot_natural_frame_3d(NF, ax=ax, show_frame_arrows=False)

    # Plot psi
    ax = fig.add_subplot(gs[0:2, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    for i in range(N - 1):
        ax.plot(NF.psi[i:i + 2], s[i:i + 2], c=fc[i])
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    # Plot number of twists
    ax = fig.add_subplot(gs[2, 0])
    t = NF.tau * (NF.N - 1) / (np.pi * 2)
    ax.set_title('Number of twists: $t=\\tau \dot N / 2 \pi$')
    ax.scatter(x=s, y=t, c=fc)

    # Plot number of coils
    k = NF.kappa / (np.pi * 2)
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title('Number of coils: $c=\kappa / 2 \pi$')
    ax.scatter(x=s, y=k, c=fc)

    # Plot curve speed
    u = NF.speed * (NF.N - 1) / NF.N
    ax = fig.add_subplot(gs[3, 0])
    ax.set_title('Curve speed: $u$')
    ax.scatter(x=s, y=u, c=fc)

    hx = t * k * u
    h1 = np.sum(hx)
    h = np.sign(h1) * np.log(1 + h1)
    assert h == NF.helicity()

    # Plot the cumulative sum of helicity contributions
    ax = fig.add_subplot(gs[3, 1])
    ax.set_title(f'Cumulative sum of $h=\log\\{{1 + \sum_s t c u={h:.6f}\\}}$')
    ax.scatter(x=s, y=np.sign(np.cumsum(hx)) * np.log(1 + np.cumsum(hx)), c=fc)
    ax.scatter(x=s[-1], y=[h, ], color='red', s=150, marker='x')

    fig.tight_layout()

    return fig


def illustrate_helicity_method_old(NF: NaturalFrame) -> Figure:
    """
    Illustrate the method.
    """
    N = NF.N
    s = np.linspace(0, 1, N)

    cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 6)

    # Plot the 3d curve
    ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax.set_title('Original curve')
    plot_natural_frame_3d(NF, ax=ax, show_frame_arrows=False)

    # Normalise length
    X = NF.X_pos.copy()
    X /= NF.length

    # If the first PCA component is large then use this to define the projection plane
    if NF.pca.explained_variance_ratio_[0] > 0.8:
        v1, v2, v3 = NF.pca.components_

    # Otherwise calculate a projection plane normal to the average direction
    else:
        v1 = normalise(NF.T.mean(axis=0) / (NF.T.var(axis=0) + EPS))
        v2 = normalise(orthogonalise(np.roll(v1, 1), v1))
        v3 = np.cross(v1, v2)

    # Rotate points to align with the projection space
    R = np.stack([v1, v2, v3], axis=1)
    Xt = np.einsum('ij,bj->bi', R.T, X)
    Xp = Xt[:, 1:]

    # Add average tangent vector to plot
    arrow = Arrow3D(
        origin=NF.X_pos.mean(axis=0),
        vec=normalise(v1) * NF.length / 2,
        color='purple',
        mutation_scale=20,
        arrowstyle='->',
        linewidth=3,
        alpha=0.9
    )
    ax.add_artist(arrow)

    # Plot the projection of the points onto the 2nd two components
    ax = fig.add_subplot(gs[0:2, 2:4])
    ax.set_title('Projected points')
    ax.scatter(x=Xp[:, 0], y=Xp[:, 1], c=fc)

    # Use the difference vectors between adjacent points and ignore first coordinate.
    diff = Xp[1:] - Xp[:-1]

    # Plot the difference vectors
    ax = fig.add_subplot(gs[0:2, 4:6])
    ax.set_title('Differences between adjacent projected points')
    ax.scatter(x=diff[:, 0], y=diff[:, 1], c=fc[:-1], alpha=0.7)
    for i, dv in enumerate(diff):
        ax.plot(np.linspace(0, dv[0], 100), np.linspace(0, dv[1], 100), color=fc[i], alpha=0.7)

    # Convert into polar coordinates
    r = np.linalg.norm(diff, axis=-1)
    theta = np.unwrap(np.arctan2(*diff.T))

    # Plot the rs and thetas
    ax = fig.add_subplot(gs[2, 0:3])
    ax.set_title('Polar coordinate: $\\theta$')
    ax.scatter(x=s[:-1], y=theta, c=fc[:-1])
    ax = fig.add_subplot(gs[2, 3:6])
    ax.set_title('Polar coordinate: r')
    ax.scatter(x=s[:-1], y=r, c=fc[:-1])

    # Weight the angular changes by the radii and sum to give helicity measure.
    r = (r[1:] + r[:-1]) / 2

    dtheta = theta[1:] - theta[:-1]
    h = np.sum(r * dtheta)

    assert h == NF.helicity()

    # Plot the change in theta
    ax = fig.add_subplot(gs[3, 0:3])
    ax.set_title('Change in $\\theta$: $\delta\\theta$')
    ax.scatter(x=s[:-2], y=dtheta, c=fc[:-2])

    # Plot the cumulative sum of helicity contributions
    ax = fig.add_subplot(gs[3, 3:6])
    ax.set_title(f'Cumulative sum of $h=\sum_s r \cdot \delta\\theta={h:.4f}$')
    ax.scatter(x=s[:-2], y=np.cumsum(r * dtheta), c=fc[:-2])
    ax.scatter(x=s[-3], y=[h, ], color='red', s=150, marker='x')

    fig.tight_layout()

    return fig
