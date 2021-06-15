import os
import time
from typing import Dict

import matplotlib.animation as manimation
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from simple_worm.frame import FrameNumpy, FrameSequenceNumpy
from simple_worm.plot3d import cla, FrameArtist

FPS = 5


class FrameDiffArtist:
    """
    Draw lines showing the difference between midline points.
    """
    line_opt_defaults = {
        'linewidth': 1,
        'alpha': 0.7,
        'color': 'red'
    }

    def __init__(
            self,
            F0: FrameNumpy,
            F1: FrameNumpy,
            line_opts: Dict = None,
    ):
        self.F0 = F0
        self.F1 = F1
        self.N = self.F0.x.shape[-1]
        self.arrows = {}
        if line_opts is None:
            line_opts = {}
        self.line_opts = {**FrameDiffArtist.line_opt_defaults, **line_opts}

    def add_diff_arrows(
            self,
            ax,
    ):
        arrows = []
        for i in range(self.N):
            p0 = self.F0.x[:, i]
            p1 = self.F1.x[:, i]

            arrow, = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                zs=[p0[2], p1[2]],
                **self.line_opt_defaults
            )

            aa = ax.add_artist(arrow)
            arrows.append(aa)
        self.arrows = arrows

    def update(self, F0: FrameNumpy, F1: FrameNumpy):
        self.F0 = F0
        self.F1 = F1

        # Update diff arrows
        for i in range(self.N):
            p0 = self.F0.x[:, i]
            p1 = self.F1.x[:, i]
            self.arrows[i].set_data_3d(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]]
            )


def generate_scatter_diff_clip(
        FS_target: FrameSequenceNumpy,
        FS_attempt: FrameSequenceNumpy,
        save_dir: str,
        save_fn: str = None,
        arrow_scale: int = 2,
        n_arrows: int = 12,
):
    """
    Generate a video clip showing a target sequence, an attempt and the difference.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_fn = save_fn if save_fn is not None else time.strftime('%Y-%m-%d_%H%M%S')
    save_path = save_dir + '/' + save_fn + '.mp4'
    labels = ['Target', 'Error', 'Attempt']

    # Show from 3 different perspectives
    perspectives = [
        {'elev': 30, 'azim': -60},
        {'elev': 30, 'azim': 60},
        {'elev': -30, 'azim': -45},
    ]

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(12, 12))
    gs = gridspec.GridSpec(len(perspectives), 3)
    axes = [[], [], []]
    artists = [[], [], []]
    for col_idx in range(3):
        if col_idx in [0, 2]:
            if col_idx == 0:
                FS = FS_target
            else:
                FS = FS_attempt
            mins, maxs = FS.get_bounding_box(zoom=1)
        else:
            mins_target, maxs_target = FS_attempt.get_bounding_box(zoom=1)
            mins_attempt, maxs_attempt = FS_target.get_bounding_box(zoom=1)
            mins = np.min((mins_target, mins_attempt), axis=0)
            maxs = np.max((maxs_target, maxs_attempt), axis=0)

        for row_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d', **perspectives[row_idx])
            cla(ax)

            if col_idx in [0, 2]:
                fa = FrameArtist(FS[0], arrow_scale=arrow_scale, n_arrows=n_arrows)
                fa.add_component_vectors(ax, draw_e0=False)
                fa.add_midline(ax)
            else:
                fa = FrameDiffArtist(FS_target[0], FS_attempt[0])
                fa.add_diff_arrows(ax)

            artists[row_idx].append(fa)

            if row_idx == 0:
                ax.set_title(labels[col_idx])

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            axes[row_idx].append(ax)

    def update(i):
        for col_idx in range(3):
            for row_idx in range(3):
                if col_idx in [0, 2]:
                    if col_idx == 0:
                        FS = FS_target
                    else:
                        FS = FS_attempt
                    artists[row_idx][col_idx].update(FS[i])
                else:
                    artists[row_idx][col_idx].update(FS_target[i], FS_attempt[i])
        return ()

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=FS_target.n_timesteps,
        blit=True
    )

    # Save
    metadata = dict(title=save_path, artist='WormLab Leeds')
    ani.save(save_path, writer='ffmpeg', fps=FPS, metadata=metadata)
    plt.close(fig)
