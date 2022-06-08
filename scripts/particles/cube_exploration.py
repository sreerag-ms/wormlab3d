import os
from argparse import Namespace
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from wormlab3d import START_TIMESTAMP, LOGS_PATH
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args

show_plots = False
save_plots = True
img_extension = 'png'


def make_filename(method: str, args: Namespace, excludes: List[str] = None):
    if excludes is None:
        excludes = []
    fn = START_TIMESTAMP + f'_{method}'

    for k in ['trial', 'frames', 'src', 'smoothing_window', 'smoothing_window_curvature']:
        if k in excludes:
            continue
        if k == 'trial':
            fn += f'_trial={args.trial}'
        elif k == 'frames':
            frames_str_fn = ''
            if args.start_frame is not None or args.end_frame is not None:
                start_frame = args.start_frame if args.start_frame is not None else 0
                end_frame = args.end_frame if args.end_frame is not None else -1
                frames_str_fn = f'_f={start_frame}-{end_frame}'
            fn += frames_str_fn
        elif k == 'src':
            if args.tracking_only:
                fn += f'_tracking'
            else:
                fn += f'_{args.midline3d_source}'
        elif k == 'smoothing_window' and args.smoothing_window is not None:
            fn += f'_sw={args.smoothing_window}'
        elif k == 'smoothing_window_curvature' and args.smoothing_window_curvature is not None:
            fn += f'_swc={args.smoothing_window_curvature}'

    return LOGS_PATH / (fn + '.' + img_extension)


def track_cube_exploration_for_trial():
    """
    Plot bishop frame components.
    """
    args = get_args()
    X = get_trajectory_from_args(args)
    T = len(X)
    null_radius = 0.01
    dt = 1 / 25
    ts = np.arange(len(X)) * dt
    plot_every_n_lines = 10

    timescales = np.array([1, 2, 4, 8, 20, 40])
    N = len(timescales)
    origins = np.zeros((T, N, 3))
    X2 = np.zeros((T, N, 3))
    dists = np.zeros((T, N))
    regions = np.zeros((T, N))

    # Use centre of mass of trajectory and start at the origin
    X_com = X.mean(axis=1)
    X = X_com - X_com[0]

    # Run sim
    for t in range(T):
        v = X[t] - origins[t]
        X2[t] = v.copy()
        if t < T - 1:
            origins[t + 1] = origins[t] + v / timescales[:, None] * dt
        d = np.linalg.norm(v, axis=-1)
        dists[t] = d

        vs = np.where(np.sign(v) >= 0, np.ones_like(v), np.zeros_like(v))
        r = vs.dot(1 << np.arange(2, -1, -1))
        r = np.where(d < null_radius, np.zeros_like(r), r + 1)
        regions[t] = r

    # Plot
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(5, N * 3)

    for i, timescale in enumerate(timescales):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
        ax.set_title(f'Timescale={timescale}s.')
        ax.plot(ts, dists[:, i])
        if i == 0:
            ax.set_ylabel('Distance from origin.')

        ax = fig.add_subplot(gs[1, i * 3:(i + 1) * 3])
        ax.set_title(f'Regions')
        ax.scatter(ts, regions[:, i], s=2)
        ax.set_xlabel('Time (s)')
        ax.set_yticks([0, 2, 4, 6, 8])

        ax = fig.add_subplot(gs[2, i * 3:(i + 1) * 3])
        ax.hist(regions[:, i], density=True, bins=9, rwidth=0.7)
        ax.set_xticks(np.linspace(0, 8, 19)[1::2])
        ax.set_xticklabels(np.arange(9))

        ax = fig.add_subplot(gs[3:, i * 3:(i + 1) * 3], projection='3d')
        x, y, z = X.T
        ax.scatter(x, y, z, c=ts, cmap='viridis_r', s=1, alpha=0.2, zorder=-1)
        x, y, z = origins[:, i].T
        ax.scatter(x, y, z, c=ts, cmap='viridis_r', s=1, alpha=0.2, zorder=-1)

        segments = np.stack([X[::plot_every_n_lines], origins[::plot_every_n_lines, i]], axis=1)
        lc = Line3DCollection(segments, array=dists[::plot_every_n_lines, i], cmap='Reds', zorder=5, linewidth=0.5,
                              alpha=0.5)
        ax.add_collection(lc)
        equal_aspect_ratio(ax)

    fig.tight_layout()
    if save_plots:
        plt.savefig(
            make_filename('cube_exploration', args),
            transparent=True
        )
    if show_plots:
        plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    track_cube_exploration_for_trial()
