from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import FrameArtist, Arrow3D, MIDLINE_CMAP_DEFAULT
from wormlab3d.postures.natural_frame import NaturalFrame


def plot_natural_frame_3d(
        NF: NaturalFrame,
        azim: float = -60.,
        elev: float = 30.,
        midline_opts: dict = None,
        show_frame_arrows: bool = True,
        n_frame_arrows: int = 30,
        arrow_opts: dict = None,
        arrow_scale: float = 0.1,
        show_pca_arrows: bool = True,
        show_pca_arrow_labels: bool = True,
        midline_cmap: str = None,
        ax: Axes = None,
        zoom: float = 1.,
        use_centred_midline: bool = True
) -> Union[Figure, Axes]:
    # Set up frame
    if use_centred_midline or np.iscomplexobj(NF.X):
        F = FrameNumpy(x=NF.X_pos.T)
    else:
        F = FrameNumpy(x=NF.X.T)
    F.e1 = NF.M1.T
    F.e2 = NF.M2.T

    # Create figure if required
    return_ax = False
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d', azim=azim, elev=elev)
    else:
        fig = ax.get_figure()
        return_ax = True
    # cla(ax)

    # Add frame arrows and midline
    if midline_opts is None:
        midline_opts = {}
    if arrow_opts is None:
        arrow_opts = {}
    arrow_opts = {**{'alpha': 0.3}, **arrow_opts}
    fa = FrameArtist(
        F,
        midline_opts=midline_opts,
        n_arrows=n_frame_arrows,
        arrow_opts=arrow_opts,
        arrow_scale=arrow_scale
    )
    if show_frame_arrows:
        fa.add_component_vectors(ax)
    fa.add_midline(ax, cmap_name=midline_cmap)

    # Add PCA component vectors
    if show_pca_arrows:
        centre = NF.X_pos.mean(axis=0)
        for i in range(2, -1, -1):
            vec = NF.pca.components_[i] * NF.pca.singular_values_[i] / 3
            if vec.sum() == 0:
                continue
            origin = centre - vec / 2
            arrow = Arrow3D(
                origin=origin,
                vec=vec,
                color=fa.arrow_colours[f'e{i}'],
                mutation_scale=20,
                arrowstyle='->',
                linewidth=3,
                alpha=0.9
            )
            ax.add_artist(arrow)

            if show_pca_arrow_labels:
                origin = origin - vec / np.linalg.norm(vec) * 0.15
                ax.text(
                    origin[0],
                    origin[1],
                    origin[2],
                    f'$v_{i}$',
                    color=fa.arrow_colours[f'e{i}'],
                    fontsize=40,
                    zorder=10,
                    horizontalalignment='center',
                    verticalalignment='center',
                )

    # Fix axes range
    mins, maxs = F.get_bounding_box(zoom=zoom)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    # ax.axis('off')

    if return_ax:
        return ax

    fig.tight_layout()

    return fig


def plot_natural_frame_components(NF: NaturalFrame) -> Figure:
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2)
    gs2 = GridSpec(2, 2, wspace=0.25, hspace=0.2, left=0.1, right=0.95, bottom=0.05, top=0.9)

    N = len(NF.m1)
    cmap = cm.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    ind = np.arange(N)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('$m_1$')
    for i in range(N - 1):
        ax.plot(ind[i:i + 2], NF.m1[i:i + 2], c=fc[i])

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.set_title('$m_2$')
    for i in range(N - 1):
        ax.plot(ind[i:i + 2], NF.m2[i:i + 2], c=fc[i])

    ax = fig.add_subplot(gs[0, 1], sharex=ax)
    ax.set_title('$|\kappa|=|m_1|+|m_2|$')
    for i in range(N - 1):
        ax.plot(ind[i:i + 2], NF.kappa[i:i + 2], c=fc[i])
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])

    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    for i in range(N - 1):
        ax.plot(NF.psi[i:i + 2], ind[i:i + 2], c=fc[i])
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    fig.tight_layout()

    return fig


def plot_component_comparison(NF1: NaturalFrame, NF2: NaturalFrame) -> Figure:
    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(2, 2)
    gs2 = GridSpec(2, 2, wspace=0.25, hspace=0.2, left=0.1, right=0.95, bottom=0.05, top=0.9)

    NF1_args = {'c': 'red', 'linestyle': '--', 'alpha': 0.8}
    NF2_args = {'c': 'blue', 'linestyle': ':', 'alpha': 0.8}

    N = len(NF1.m1)
    ind = np.arange(N)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('$m_1$')
    ax.plot(ind, NF1.m1, **NF1_args)
    ax.plot(ind, NF2.m1, **NF2_args)

    ax = fig.add_subplot(gs[1, 0], sharex=ax)
    ax.set_title('$m_2$')
    ax.plot(ind, NF1.m2, **NF1_args)
    ax.plot(ind, NF2.m2, **NF2_args)

    ax = fig.add_subplot(gs[0, 1], sharex=ax)
    ax.set_title('$|\kappa|=|m_1|+|m_2|$')
    ax.plot(ind, NF1.kappa, **NF1_args)
    ax.plot(ind, NF2.kappa, **NF2_args)
    ax.set_xticks([0, ind[-1]])
    ax.set_xticklabels(['H', 'T'])

    ax = fig.add_subplot(gs2[1, 1], projection='polar')
    ax.set_title('$\psi=arg(m_1+i m_2)$')
    ax.plot(NF1.psi, ind, **NF1_args)
    ax.plot(NF2.psi, ind, **NF2_args)
    ax.set_rticks([])
    thetaticks = np.arange(0, 2 * np.pi, np.pi / 2)
    ax.set_xticks(thetaticks)
    ax.set_xticklabels(['0', '$\pi/2$', '$\pi$', '$3\pi/2$'])
    ax.xaxis.set_tick_params(pad=-3)

    fig.tight_layout()

    return fig
