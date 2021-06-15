import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.figure import Figure

# from plot3d import generate_scatter_diff_clip
from simple_worm.controls import ControlSequenceNumpy
from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import MidpointNormalize, cla, FrameArtist
from simple_worm.util import expand_numpy


def plot_frame_components(
        F: FrameNumpy,
) -> Figure:
    """
    Plot the psi/e0/e1/e2 frame components as matrices.
    """

    # Expand the psi to two dims
    psi0 = expand_numpy(F.psi)

    # Determine common scales
    e_vmin = min(F.e0.min(), F.e1.min(), F.e2.min())
    e_vmax = max(F.e0.max(), F.e1.max(), F.e2.max())

    fig, axes = plt.subplots(1, 4, figsize=(12, 7), squeeze=False)

    Ms = [psi0, F.e0, F.e1, F.e2]

    for col_idx in range(4):
        ax = axes[0, col_idx]
        if col_idx == 0:
            # Use a cyclic colormap with a fixed scale for psi as 0=2pi
            cmap = plt.cm.twilight
            vmin = 0
            vmax = 2 * np.pi
            norm = None
        else:
            cmap = plt.cm.PRGn
            vmin = e_vmin
            vmax = e_vmax
            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

        m = ax.matshow(
            Ms[col_idx],
            cmap=cmap,
            clim=(vmin, vmax),
            norm=norm,
            aspect='auto'
        )

        ax.set_title(['$\psi$', '$e_0$', '$e_1$', '$e_2$'][col_idx])
        if col_idx in [0, 3]:
            fig.colorbar(m, ax=ax)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig


def plot_CS(
        CS: ControlSequenceNumpy,
) -> Figure:
    """
    Plot the control sequences as matrices.
    """

    # Determine common scales
    ab_vmin = min(CS.alpha.min(), CS.beta.min())
    ab_vmax = max(CS.alpha.max(), CS.beta.max())
    g_vmin = CS.gamma.min()
    g_vmax = CS.gamma.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), squeeze=False)

    Ms = [CS.alpha, CS.beta, CS.gamma]

    for col_idx, M in enumerate(Ms):
        if col_idx == 2:
            cmap = plt.cm.BrBG
            vmin = g_vmin
            vmax = g_vmax
            cbar_format = '%.4f'
        else:
            cmap = plt.cm.PRGn
            vmin = ab_vmin
            vmax = ab_vmax
            cbar_format = '%.3f'

        ax = axes[0, col_idx]
        m = ax.matshow(
            M.T,
            cmap=cmap,
            clim=(vmin, vmax),
            norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
            aspect='auto'
        )

        ax.set_title(['$\\alpha$', '$\\beta$', '$\gamma$'][col_idx])
        fig.colorbar(m, ax=ax, format=cbar_format)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig


def plot_F0_3d(
        F0: FrameNumpy
) -> Figure:
    # interactive()
    mins, maxs = F0.get_bounding_box()
    elevs = [-60, 0, 60]
    azims = [-60, 0, 60]

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3)
    for row_idx in range(3):
        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d', elev=elevs[row_idx], azim=azims[col_idx])
            cla(ax)
            fa = FrameArtist(F=F0, midline_opts={'s': 10}, arrow_scale=8, n_arrows=12)
            fa.add_component_vectors(ax, draw_e0=False)
            fa.add_midline(ax)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0)

    return fig
