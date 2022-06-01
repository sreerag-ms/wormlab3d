import os
from argparse import Namespace
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, Arrow3D, MidpointNormalize
from wormlab3d import LOGS_PATH, START_TIMESTAMP, logger
from wormlab3d.data.model import Dataset, Reconstruction
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, make_box_from_pca
from wormlab3d.trajectories.args import get_args
from wormlab3d.trajectories.cache import get_trajectory_from_args
from wormlab3d.trajectories.manoeuvres import get_manoeuvres, get_forward_durations
from wormlab3d.trajectories.util import calculate_speeds

animate = False
show_plots = True
save_plots = True
img_extension = 'svg'
fps_anim = 25
playback_speed = 10
n_revolutions = 0.5

arrow_colours = {
    'e0': 'red',
    'e1': 'blue',
    'e2': 'green',
}


def get_trajectory(args: Namespace):
    X_slice = get_trajectory_from_args(args)
    trajectory_point = args.trajectory_point
    args.trajectory_point = None
    X_full = get_trajectory_from_args(args)
    args.trajectory_point = trajectory_point
    return X_full, X_slice


def add_pca_arrows(X, pca):
    arrows = []
    # Add PCA component vectors
    centre = X.mean(axis=0)
    for i in range(2, -1, -1):
        vec = pca.components_[i] * pca.singular_values_[i] / 5
        if vec.sum() == 0:
            continue
        origin = centre - vec / 2
        arrow = Arrow3D(
            origin=origin,
            vec=vec,
            color=arrow_colours[f'e{i}'],
            mutation_scale=25,
            arrowstyle='->',
            linewidth=3,
            alpha=0.9
        )
        arrows.append(arrow)
    return arrows


def plot_manoeuvre_3d(
        X_slice: np.ndarray,
        X_full: np.ndarray = None,
        title: str = None,
        folder: str = None,
        filename: str = None,
        colours: np.ndarray = None,
        cmap: str = 'jet',
        show_colourbar: bool = False,
        ax: Axes = None,
        worm_idxs: Union[int, List[int]] = 0,
        arrows: List[Arrow3D] = None,
        planes: List[Poly3DCollection] = None,
        azim: int = -60,
        elev: int = 30,
):
    x, y, z = X_slice.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d', azim=azim, elev=elev)

    # Scatter the vertices
    s = ax.scatter(x, y, z, c=colours, cmap=cmap, s=10, alpha=0.4, zorder=-1, norm=MidpointNormalize(midpoint=0))
    if show_colourbar:
        cb = fig.colorbar(s, shrink=0.5)
        cb.ax.tick_params(labelsize=12)
        cb.ax.set_ylabel('Speed (mm/s)', rotation=270, fontsize=14)

    # Draw lines connecting points
    points = X_slice[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2)
    ax.add_collection(lc)

    # Add worm
    if worm_idxs != -1:
        if type(worm_idxs) != list:
            worm_idxs = [worm_idxs, ]
        for worm_idx in worm_idxs:
            if worm_idx >= len(X_full):
                worm_idx = len(X_full) - 1
            FS = FrameSequenceNumpy(x=X_full.transpose(0, 2, 1))
            fa = FrameArtist(F=FS[worm_idx], midline_opts={'zorder': 100, 's': 100})
            fa.add_midline(ax)

    # Add arrows
    if arrows is not None:
        for arrow in arrows:
            ax.add_artist(arrow)

    # Add planes
    if planes is not None:
        for plane in planes:
            ax.add_collection3d(plane)

    equal_aspect_ratio(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if title is not None:
        ax.set_title(title)
    fig.tight_layout()

    if animate:
        # Aspects
        azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X_slice))
        ax.view_init(azim=azims[0])  # elev

        def update(frame_num: int):
            # Rotate the view.
            ax.view_init(azim=azims[frame_num])

            # Update the worm
            if worm_idxs != -1:
                fa.update(FS[frame_num])
            return ()

        idxs_mask = np.array([i % playback_speed == 0 for i in np.arange(np.round(len(X_slice)))])
        frame_nums = np.arange(len(X_slice))[idxs_mask].tolist()
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frame_nums,
            blit=True,
            interval=1 / fps_anim
        )

    if save_plots:
        assert filename is not None
        if folder is not None:
            path = LOGS_PATH / f'{START_TIMESTAMP}_{folder}'
            os.makedirs(path, exist_ok=True)
            path = path / filename
        else:
            path = LOGS_PATH / f'{START_TIMESTAMP}_{filename}'
        if animate:
            metadata = dict(
                title=title,
                artist='WormLab Leeds'
            )
            save_path = path.with_suffix('.mp4')
            logger.info(f'Saving animation to {save_path}.')
            ani.save(save_path, writer='ffmpeg', fps=fps_anim, metadata=metadata)
        else:
            save_path = path.with_suffix(f'.{img_extension}')
            logger.info(f'Saving plot to {save_path}.')
            plt.savefig(save_path)

    if show_plots:
        plt.show()

    plt.close(fig)


def plot_all_manoeuvres():
    """
    Plot all manoeuvres detected in the trajectory.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    signed_speeds = calculate_speeds(X_full, signed=True)
    # plt.plot(signed_speeds)
    # plt.show()
    # exit()

    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    folder = f'manoeuvres_trial={args.trial}_{args.midline3d_source}' \
             f'_rev={args.min_reversal_frames}' \
             f'_ws={args.manoeuvre_window}' \
             f'_sw={args.smoothing_window}'

    # Loop over manoeuvres
    for i, m in enumerate(manoeuvres):
        plane_prev = make_box_from_pca(m['X_prev'], m['pca_prev'], 'orange')
        plane_next = make_box_from_pca(m['X_next'], m['pca_next'], 'green')

        # Xm = X_slice[m['start_idx']:m['end_idx']]
        # pca_all = PCA(svd_solver='full', copy=True, n_components=3)
        # pca_all.fit(Xm)
        # plane_all = get_plane(Xm, pca_all, 'blue')

        plot_manoeuvre_3d(
            title=f'Trial {args.trial}. '
                  f'Frames {m["start_idx"]}-{m["end_idx"]}. '
                  f'Reversal duration={m["reversal_duration"] / 25:.1f}s. '
                  f'Window size={args.manoeuvre_window} '
                  f'Smoothing window={args.smoothing_window}',
            X_slice=X_slice[m['start_idx']:m['end_idx']],
            X_full=X_full[m['start_idx']:m['end_idx']],
            folder=folder,
            filename=f'frames={m["start_idx"]}-{m["end_idx"]}',
            colours=signed_speeds[m['start_idx']:m['end_idx']],
            cmap='PRGn',
            show_colourbar=False,
            worm_idxs=int((m['end_idx'] - m['start_idx']) / 2) + 50,
            # planes=[plane_prev, plane_next, plane_all],
            planes=[plane_prev, plane_next],
        )
        # exit()


def plot_angles_and_durations():
    """
    Plot the angles between the planes coming into and going out of a manoeuvre.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    # Loop over manoeuvres
    angles = []
    durations = []
    for i, m in enumerate(manoeuvres):
        angles.append(m['angle'])
        durations.append(m['reversal_duration'] / 25)

    fig, axes = plt.subplots(3, figsize=(6, 6))
    fig.suptitle(f'Trial {args.trial}. '
                 f'Num reversal frames={args.min_reversal_frames}. '
                 f'Window size={args.manoeuvre_window}')

    # Plot histogram of angles
    ax = axes[0]
    ax.hist(angles, bins=50, density=True, facecolor='green', alpha=0.75)
    ax.set_ylabel('P(angle)')
    ax.set_xlabel('Angle')

    # Plot histogram of the reversal durations
    ax = axes[1]
    ax.hist(durations, bins=50, density=True, facecolor='green', alpha=0.75)
    ax.set_ylabel('P(duration)')
    ax.set_xlabel('Duration (s)')

    # Scatter
    ax = axes[2]
    ax.scatter(x=durations, y=angles)
    ax.set_ylabel('Angles')
    ax.set_xlabel('Durations (s)')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH + '/' + START_TIMESTAMP \
               + f'_angles_v_durations' \
                 f'_trial={args.trial}_{args.midline3d_source}' \
                 f'_rev={args.min_reversal_frames}' \
                 f'_ws={args.manoeuvre_window}'
        save_path = path + f'.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_angles_and_durations_varying_parameters():
    """
    Plot the angles between the planes coming into and going out of a manoeuvre.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    reversal_frames = [25, 50, 100]
    window_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000]

    fig, axes = plt.subplots(len(reversal_frames), len(window_sizes), figsize=(10, 10), sharex=True, sharey=True)
    fig.suptitle(f'Trial {args.trial}.')

    for row_idx, rf in enumerate(reversal_frames):
        for col_idx, ws in enumerate(window_sizes):
            manoeuvres = get_manoeuvres(
                X_full,
                X_slice,
                min_reversal_frames=rf,
                window_size=ws
            )

            # Loop over manoeuvres
            angles = []
            durations = []
            for i, m in enumerate(manoeuvres):
                angles.append(m['angle'])
                durations.append(m['reversal_duration'] / 25)

            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(f'Window size={ws}')
            if col_idx == 0:
                ax.set_ylabel(f'Reversal frames={rf}')

            ax.scatter(x=durations, y=angles, s=15, alpha=0.7)

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH + '/' + START_TIMESTAMP \
               + f'_angles_v_durations' \
                 f'_trial={args.trial}_{args.midline3d_source}' \
                 f'_rev={",".join([str(rf) for rf in reversal_frames])}' \
                 f'_ws={",".join([str(ws) for ws in window_sizes])}'
        save_path = path + f'.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_manoeuvre_rate():
    """
    Plot the rate of manoeuvres.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )

    # Loop over manoeuvres, recording the idxs they occurred
    idxs = [m['centre_idx'] for m in manoeuvres]
    forward_durations = np.diff(idxs) / 25
    reversal_durations = np.array([m['reversal_duration'] for m in manoeuvres]) / 25

    fig, axes = plt.subplots(2, figsize=(6, 6), sharex=True, sharey=True)
    fig.suptitle(f'Trial {args.trial}. '
                 f'Min reversal frames={args.min_reversal_frames}. ')

    # Plot histogram of the forward durations
    ax = axes[0]
    ax.hist(forward_durations, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Forward duration (s)')

    # Plot histogram of the reversal durations
    ax = axes[1]
    ax.hist(reversal_durations, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Reversal duration (s)')

    fig.tight_layout()

    if save_plots:
        path = LOGS_PATH + '/' + START_TIMESTAMP \
               + f'_locomotive_durations' \
                 f'_trial={args.trial}_{args.midline3d_source}' \
                 f'_rev={args.min_reversal_frames}'
        save_path = path + f'.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_dataset_distributions():
    """
    Plot the distributions for a dataset.
    """
    args = get_args()

    # Get dataset
    assert args.dataset is not None
    ds = Dataset.objects.get(id=args.dataset)

    # Unset trial and midline source args
    args.trial = None
    args.midline3d_source = None
    args.midline3d_source_file = None

    angles = {}
    durations_fwd = {}
    durations_bck = {}
    angles_all = []
    durations_fwd_all = []
    durations_bck_all = []

    # Loop over reconstructions
    for r_ref in ds.reconstructions:
        reconstruction = Reconstruction.objects.get(id=r_ref.id)
        args.reconstruction = reconstruction.id
        fps = reconstruction.trial.fps

        X_full, X_slice = get_trajectory(args)
        manoeuvres = get_manoeuvres(
            X_full,
            X_slice,
            min_reversal_frames=args.min_reversal_frames,
            window_size=args.manoeuvre_window
        )

        # Loop over manoeuvres
        durations_fwd_r = get_forward_durations(X_full, min_forward_frames=args.min_forward_frames) / fps
        angles_r = []
        durations_bck_r = []
        for i, m in enumerate(manoeuvres):
            angles_r.append(m['angle'])
            durations_bck_r.append(m['reversal_duration'] / fps)

        # Collate
        c = reconstruction.trial.experiment.concentration
        if c not in angles:
            angles[c] = []
        if c not in durations_fwd:
            durations_fwd[c] = []
        if c not in durations_bck:
            durations_bck[c] = []
        angles[c].extend(angles_r)
        durations_fwd[c].extend(durations_fwd_r)
        durations_bck[c].extend(durations_bck_r)
        angles_all.extend(angles_r)
        durations_fwd_all.extend(durations_fwd_r)
        durations_bck_all.extend(durations_bck_r)

    # Sort by concentration
    angles = {k: v for k, v in sorted(list(angles.items()))}
    durations_fwd = {k: v for k, v in sorted(list(durations_fwd.items()))}
    durations_bck = {k: v for k, v in sorted(list(durations_bck.items()))}
    concs = [c for c, _ in angles.items()]

    # Set up plots
    n_rows = 1 + len(angles)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, n_rows * 3))
    fig.suptitle(f'Min reversal frames={args.min_reversal_frames}. ')

    # First row shows the collated results
    ax = axes[0, 0]
    ax.hist(angles_all, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Angle')

    ax = axes[0, 1]
    ax.hist(durations_fwd_all, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Forward duration (s)')

    ax = axes[0, 2]
    ax.hist(durations_bck_all, bins=50, density=False, facecolor='green', alpha=0.75)
    ax.set_ylabel('Count')
    ax.set_xlabel('Backward duration (s)')

    for i, c in enumerate(concs):
        ax = axes[i + 1, 0]
        ax.set_title(f'Concentration = {c}')
        ax.hist(angles[c], bins=50, density=False, facecolor='green', alpha=0.75)
        ax.set_ylabel('Count')
        ax.set_xlabel('Angle')

        ax = axes[i + 1, 1]
        ax.set_title(f'Concentration = {c}')
        ax.hist(durations_fwd[c], bins=50, density=False, facecolor='green', alpha=0.75)
        ax.set_ylabel('Count')
        ax.set_xlabel('Forward duration (s)')

        ax = axes[i + 1, 2]
        ax.set_title(f'Concentration = {c}')
        ax.hist(durations_bck[c], bins=50, density=False, facecolor='green', alpha=0.75)
        ax.set_ylabel('Count')
        ax.set_xlabel('Backward duration (s)')

    fig.tight_layout()

    if save_plots:
        fn = START_TIMESTAMP \
             + f'_durations_and_angles' \
               f'_ds={args.dataset}' \
               f'_rev={args.min_reversal_frames}' \
               f'_fwd={args.min_forward_frames}'
        save_path = LOGS_PATH / (fn + f'.{img_extension}')
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_single_manoeuvre(index: int):
    """
    Plot a single manoeuvres detected in the trajectory.
    """
    args = get_args()
    X_full, X_slice = get_trajectory(args)
    signed_speeds = calculate_speeds(X_full, signed=True)

    manoeuvres = get_manoeuvres(
        X_full,
        X_slice,
        min_reversal_frames=args.min_reversal_frames,
        window_size=args.manoeuvre_window
    )
    m = manoeuvres[index]

    filename = f'manoeuvre_{index}_trial={args.trial}_{args.midline3d_source}' \
               f'_rev={args.min_reversal_frames}' \
               f'_ws={args.manoeuvre_window}' \
               f'_sw={args.smoothing_window}' \
               f'_frames={m["start_idx"]}-{m["end_idx"]}'

    plane_prev = make_box_from_pca(m['X_prev'], m['pca_prev'], 'orange', scale=(1, 2, 3))
    plane_next = make_box_from_pca(m['X_next'], m['pca_next'], 'green', scale=(1, 3, 1))

    plot_manoeuvre_3d(
        X_slice=X_slice[m['start_idx']:m['end_idx']],
        X_full=X_full[m['start_idx']:m['end_idx']],
        filename=filename,
        colours=signed_speeds[m['start_idx']:m['end_idx']],
        cmap='PRGn',
        show_colourbar=True,
        worm_idxs=[500, 900],
        planes=[plane_prev, plane_next],
        azim=55,
        elev=-5
    )


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
    # from simple_worm.plot3d import interactive
    #
    # interactive()
    # plot_all_manoeuvres()
    # plot_angles_and_durations()
    # plot_angles_and_durations_varying_parameters()
    # plot_manoeuvre_rate()
    # plot_dataset_distributions()

    plot_single_manoeuvre(index=2)
