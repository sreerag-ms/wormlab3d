import json
import os
import time
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Tuple, Callable

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from mayavi import mlab
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist, MidpointNormalize
from wormlab3d import PREPARED_IMAGES_PATH
from wormlab3d import logger, START_TIMESTAMP, LOGS_PATH
from wormlab3d.data.model import Frame, Reconstruction, Trial, Midline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.project_render_score import ProjectRenderScoreModel
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.particles.tumble_run import calculate_curvature
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.postures.helicities import calculate_helicities, plot_helicities
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio, overlay_image
from wormlab3d.toolkit.util import normalise, print_args, str2bool, to_dict
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.pca import generate_or_load_pca_cache
from wormlab3d.trajectories.util import smooth_trajectory, calculate_speeds

# Off-screen rendering
mlab.options.offscreen = True


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to generate a reconstruction video.')

    # Video
    parser.add_argument('--width', type=int, default=1200, help='Width of video in pixels.')
    parser.add_argument('--height', type=int, default=900, help='Height of video in pixels.')

    # Trajectory
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--use-valid-range', type=str2bool, help='Use valid range if available.')
    parser.add_argument('--smoothing-window-postures', type=int, help='Smoothing window for the postures.')
    parser.add_argument('--smoothing-window-components', type=int, help='Smoothing window for the components.')
    parser.add_argument('--smoothing-window-speed', type=int, default=25,
                        help='Smoothing window for the speed calculation.')

    # Trajectory
    parser.add_argument('--trajectory-use-mlab', type=str2bool, default=True,
                        help='Use mayavi mlab to render the 3D trajectory plot.')
    parser.add_argument('--trajectory-colouring', type=str, choices=['time', 'speed', 'curvature'], default='time',
                        help='Colour the 3D trajectory by time, speed or curvature.')
    parser.add_argument('--show-trajectory-colourbar', type=str2bool, default=False,
                        help='Show colourbar on the 3D trajectory plot.')
    parser.add_argument('--show-trajectory-axis', type=str2bool, default=True,
                        help='Show axis on the 3D trajectory plot.')
    parser.add_argument('--show-trajectory-grid', type=str2bool, default=True,
                        help='Show grid on the 3D trajectory plot.')
    parser.add_argument('--show-trajectory-ticks', type=str2bool, default=True,
                        help='Show ticks on the 3D trajectory plot.')
    parser.add_argument('--show-trajectory-tick-labels', type=str2bool, default=False,
                        help='Show tick labels on the 3D trajectory plot.')

    # Posture
    parser.add_argument('--posture-use-mlab', type=str2bool, default=True,
                        help='Use mayavi mlab to render the 3D posture plot.')
    parser.add_argument('--show-posture-axis', type=str2bool, default=True,
                        help='Show axis on the 3D posture plot.')
    parser.add_argument('--show-posture-grid', type=str2bool, default=True,
                        help='Show grid on the 3D posture plot.')
    parser.add_argument('--show-posture-ticks', type=str2bool, default=False,
                        help='Show ticks on the 3D posture plot.')
    parser.add_argument('--show-posture-tick-labels', type=str2bool, default=False,
                        help='Show tick labels on the 3D posture plot.')
    parser.add_argument('--revolution-rate', type=float, default=1 / 3,
                        help='Rate of 3D plot revolution in revolutions/minute.')

    # Traces
    parser.add_argument('--time-range-traces', type=float, default=5,
                        help='Time range to show on trace plots in seconds.')
    parser.add_argument('--planarity-windows', type=lambda s: [int(item) for item in s.split(',')],
                        default='1,2,5,10', help='Comma delimited list of planarity windows in seconds.')
    parser.add_argument('--rebuild-planarity-cache', type=str2bool, default=False, help='Rebuild the planarity caches.')
    parser.add_argument('--eigenworms', type=str, help='Eigenworms by id.')
    parser.add_argument('--n-components', type=int, default=20, help='Number of eigenworms to use (basis dimension).')
    parser.add_argument('--plot-components', type=lambda s: [int(item) for item in s.split(',')],
                        default='0,1', help='Comma delimited list of component idxs to plot.')
    parser.add_argument('--x-label', type=str, default='time', help='Label x-axis with time or frame number.')
    parser.add_argument('--n-helicity-fade-lines', type=int, default=100,
                        help='Filled region fade resolution in helicity plot.')

    # Lambdas
    parser.add_argument('--time-range-lambdas', type=float, default=5,
                        help='Time range to show on lambdas plots in seconds.')

    args = parser.parse_args()
    assert args.reconstruction is not None, 'This script requires setting --reconstruction=id.'

    print_args(args)

    return args


def _make_info_panel(
        width: int,
        height: int,
        trial: Trial,
        start_frame_trial: int,
        X: np.ndarray,
        lengths: np.ndarray,
) -> Tuple[Figure, Callable]:
    """
    Info panel.
    """
    logger.info('Building infos plot.')

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    ax = fig.add_subplot(111)
    ax.axis('off')

    def get_details(frame_idx: int) -> str:
        curr_time = datetime.fromtimestamp(np.floor(frame_idx / trial.fps))
        total_time = datetime.fromtimestamp(np.floor(len(X) / trial.fps))
        return f'Trial #{trial.id}.\n' \
               f'Concentration: {trial.experiment.concentration}%\n' \
               f'Frame: {frame_idx + 1:,}/{len(X):,}\n' \
               f'Time: {curr_time:%M:%S}/{total_time:%M:%S}\n' \
               f'Trial frame: {start_frame_trial + frame_idx:,}\n' \
               f'Length: {lengths[frame_idx]:.3f}'

    # Details
    text = fig.text(0.1, 0.9, get_details(0), ha='left', va='top', fontsize=12, linespacing=1.5)

    def update(frame_idx: int):
        # Update the text
        text.set_text(get_details(frame_idx))

        # Redraw the canvas
        fig.canvas.draw()

    fig.tight_layout()

    return fig, update


def _make_3d_trajectory_plot(
        width: int,
        height: int,
        trial: Trial,
        X: np.ndarray,
        speeds: np.ndarray,
        curvature: np.ndarray,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D trajectory plot with worm using matplotlib.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D trajectory plot.')
    X_trajectory = X.mean(axis=1)
    x, y, z = X_trajectory.T

    # Construct colours
    if args.trajectory_colouring == 'time':
        colours = np.linspace(0, 1, len(X))
        cmap = plt.get_cmap('viridis_r')
    elif args.trajectory_colouring == 'speed':
        colours = speeds
        cmap = plt.get_cmap('PRGn')
    elif args.trajectory_colouring == 'curvature':
        colours = curvature
        cmap = plt.get_cmap('Reds')
    c = [cmap(c_) for c_ in colours]

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    gs = GridSpec(1, 1, top=0.99, bottom=0.01, left=0.01, right=0.99)
    ax = fig.add_subplot(gs[0, 0], projection='3d')

    # Scatter the vertices
    s = ax.scatter(x, y, z, c=c, s=5, alpha=0.4, zorder=-1)
    if args.show_trajectory_colourbar:
        fig.colorbar(s)

    # Draw lines connecting points
    points = X_trajectory[:, None, :]
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2, alpha=0.3)
    ax.add_collection(lc)

    # Add worm
    FS = FrameSequenceNumpy(x=X.transpose(0, 2, 1))
    fa = FrameArtist(F=FS[0])
    fa.add_midline(ax)

    # Setup axis
    equal_aspect_ratio(ax)
    if not args.show_trajectory_axis:
        ax.axis('off')
    if args.show_trajectory_grid:
        ax.grid()
    else:
        ax.grid(False)
    if not args.show_trajectory_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if not args.show_trajectory_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Aspects
    n_revolutions = len(X) / trial.fps / 60 * args.revolution_rate
    azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X))
    ax.view_init(azim=azims[0])

    def update(frame_idx: int):
        # Rotate the view.
        ax.view_init(azim=azims[frame_idx])

        # Update the worm
        fa.update(FS[frame_idx])

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def _make_3d_trajectory_plot_mlab(
        width: int,
        height: int,
        trial: Trial,
        X: np.ndarray,
        speeds: np.ndarray,
        curvature: np.ndarray,
        curvature_postures: np.ndarray,
        lengths: np.ndarray,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a 3D trajectory plot with worm using mayavi.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D trajectory plot.')

    X_trajectory = X.mean(axis=1)
    centre = X_trajectory.mean(axis=0)

    # Construct colours
    if args.trajectory_colouring == 'time':
        s = np.linspace(0, 1, len(X))
        cmap = plt.get_cmap('viridis_r')
    elif args.trajectory_colouring == 'speed':
        s = speeds
        cmap = plt.get_cmap('PRGn')
    elif args.trajectory_colouring == 'curvature':
        s = curvature
        cmap = plt.get_cmap('Reds')
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255

    # Set up mlab figure
    fig = mlab.figure(size=(width * 2, height * 2), bgcolor=(1, 1, 1))
    if 1:
        # Doesn't really seem to make any difference
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 20
        fig.scene.anti_aliasing_frames = 20

    # Determine zoom level - make a sphere around the centre that would contain
    # a straight worm at its longest length extending from the furthest trajectory point.
    max_dist = np.linalg.norm(X - centre, axis=-1).max() + lengths.max()
    phi, theta = np.mgrid[0:2 * np.pi:12j, 0:np.pi:12j]
    r = (max_dist * 0.6) / 2
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    tmp_mesh = mlab.mesh(x, y, z)
    mlab.view(figure=fig, distance='auto')
    distance = mlab.view()[2]
    tmp_mesh.remove()

    # Render the trajectory with simple lines
    x, y, z = X_trajectory.T
    path = mlab.plot3d(x, y, z, s, opacity=0.4, tube_radius=None, line_width=8)
    path.module_manager.scalar_lut_manager.lut.table = cmaplist

    # Set up the artist and add the pieces
    NF = NaturalFrame(X[0])
    fa = FrameArtistMLab(
        NF,
        use_centred_midline=False,
        midline_opts={'opacity': 1, 'line_width': 8},
        surface_opts={'radius': 0.02 * lengths.mean()}
    )
    fa.add_midline(fig)
    fa.add_surface(fig, v_min=-curvature_postures.max(), v_max=curvature_postures.max())

    # Add box/axes
    mlab.outline(path, color=(0, 0, 0), figure=fig)
    axes = mlab.axes(color=(0, 0, 0), nb_labels=5, xlabel='', ylabel='', zlabel='')
    axes.axes.label_format = ''

    # Aspects
    n_revolutions = len(X) / trial.fps / 60 * args.revolution_rate
    azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X))
    # fig.scene._lift()
    mlab.view(figure=fig, azimuth=azims[0], distance=distance, focalpoint=centre)

    def update(frame_idx: int):
        fig.scene.disable_render = True
        NF = NaturalFrame(X[frame_idx])
        fa.update(NF)
        fig.scene.disable_render = False
        mlab.view(figure=fig, azimuth=azims[frame_idx], distance=distance, focalpoint=centre)
        fig.scene.render()

    return fig, update


def _make_3d_posture_plot(
        width: int,
        height: int,
        trial: Trial,
        X: np.ndarray,
        args: Namespace
) -> Tuple[Figure, Callable]:
    """
    Build a 3D posture plot using matplotlib.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D postures plot.')

    # Create figure
    fig = plt.figure(figsize=(width / 100, height / 100))
    gs = GridSpec(1, 1, top=0.99, bottom=0.01, left=0.01, right=0.99)
    ax = fig.add_subplot(gs[0, 0], projection='3d')

    # Add postures
    FS = FrameSequenceNumpy(x=X.transpose(0, 2, 1))
    fa = FrameArtist(F=FS[0])
    fa.add_midline(ax)

    # Setup axis
    equal_aspect_ratio(ax)
    if not args.show_posture_axis:
        ax.axis('off')
    if args.show_posture_grid:
        ax.grid()
    else:
        ax.grid(False)
    if not args.show_posture_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    if not args.show_posture_tick_labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Aspects
    n_revolutions = len(X) / trial.fps / 60 * args.revolution_rate
    azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X))
    ax.view_init(azim=azims[0])

    def update(frame_idx: int):
        # Rotate the view.
        ax.view_init(azim=azims[frame_idx])

        # Update the worm
        F = FS[frame_idx]
        fa.update(F)
        bb = F.get_range()
        for i, axis in enumerate('xyz'):
            getattr(ax, f'set_{axis}lim')(bb[0][i], bb[1][i])
        equal_aspect_ratio(ax)

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def _make_3d_posture_plot_mlab(
        width: int,
        height: int,
        trial: Trial,
        X: np.ndarray,
        curvature: np.ndarray,
        lengths: np.ndarray,
        args: Namespace
) -> Tuple[Figure, Callable]:
    """
    Build a 3D posture plot using mayavi.
    Returns an update function to call which rotates the view and updates the worm.
    """
    logger.info('Building 3D postures plot.')

    # Set up mlab figure
    fig = mlab.figure(size=(width * 2, height * 2), bgcolor=(1, 1, 1))
    if 1:
        # Doesn't really seem to make any difference
        fig.scene.render_window.point_smoothing = True
        fig.scene.render_window.line_smoothing = True
        fig.scene.render_window.polygon_smoothing = True
        fig.scene.render_window.multi_samples = 20
        fig.scene.anti_aliasing_frames = 20

    # Set up the artist and add the pieces
    NF = NaturalFrame(X[0])
    fa = FrameArtistMLab(
        NF,
        use_centred_midline=True,
        midline_opts={'opacity': 1, 'line_width': 8},
        surface_opts={'radius': 0.02 * lengths.mean()}
    )
    fa.add_midline(fig)
    fa.add_surface(fig, v_min=-curvature.max(), v_max=curvature.max())
    fa.add_outline(fig)

    # Determine zoom level - make a sphere around the average midpoint that would contain
    # a straight worm at its longest length in the clip
    max_length = lengths.max()
    phi, theta = np.mgrid[0:2 * np.pi:12j, 0:np.pi:12j]
    mp = np.ptp(fa.X, axis=0) / 2
    r = (max_length * 0.8) / 2
    x = mp[0] + r * np.cos(phi) * np.sin(theta)
    y = mp[1] + r * np.sin(phi) * np.sin(theta)
    z = mp[2] + r * np.cos(theta)
    tmp_mesh = mlab.mesh(x, y, z, opacity=0.4)
    mlab.view(figure=fig, distance='auto', focalpoint=mp)
    distance = mlab.view()[2]
    tmp_mesh.remove()

    # Aspects
    n_revolutions = len(X) / trial.fps / 60 * args.revolution_rate
    azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X))
    mlab.view(figure=fig, azimuth=azims[0], distance=distance, focalpoint=mp)

    def update(frame_idx: int):
        fig.scene.disable_render = True
        NF = NaturalFrame(X[frame_idx])
        fa.update(NF)
        fig.scene.disable_render = False
        mlab.view(figure=fig, azimuth=azims[frame_idx], distance=distance, focalpoint=np.ptp(fa.X, axis=0) / 2)
        fig.scene.render()

    return fig, update


def _make_traces_plots(
        width: int,
        height: int,
        reconstruction: Reconstruction,
        X: np.ndarray,
        X_ew: np.ndarray,
        speeds: np.ndarray,
        helicities: np.ndarray,
        curvature: np.ndarray,
        torsion: np.ndarray,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a traces plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building traces plot.')
    N = len(X)
    trial = reconstruction.trial
    if args.x_label == 'time':
        ts = np.linspace(0, N / trial.fps, N)
        t_range = args.time_range_traces
    else:
        ts = np.arange(N) + reconstruction.start_frame
        t_range = args.time_range_traces * trial.fps

    # Planarities
    common_args = {
        'reconstruction_id': reconstruction.id,
        'use_valid_range': args.use_valid_range,
        'smoothing_window': args.smoothing_window_postures,
        'rebuild_cache': args.rebuild_planarity_cache
    }
    logger.info('Fetching posture planarities.')
    pcas, meta = generate_or_load_pca_cache(**common_args, window_size=1)
    r = pcas.explained_variance_ratio.T
    nonp_postures = r[2] / np.sqrt(r[1] * r[0])

    logger.info('Fetching trajectory planarities.')
    nonp_trajectories = np.zeros((len(args.planarity_windows), N))
    for i, ws in enumerate(args.planarity_windows):
        pcas, meta = generate_or_load_pca_cache(**common_args, window_size=round(ws * trial.fps))
        r = pcas.explained_variance_ratio.T
        t0 = int(np.floor((N - len(pcas)) / 2))
        nonp_trajectories[i, t0:t0 + len(pcas)] = r[2] / np.sqrt(r[1] * r[0])

    # Plot
    fig, axes = plt.subplots(6, figsize=(width / 100, height / 100), gridspec_kw={
        'hspace': 0,
        'top': 0.98,
        'bottom': 0.07,
        'left': 0.15,
        'right': 0.86,
    })

    # Speeds
    ax_speed = axes[0]
    ax_speed.axhline(y=0, color='darkgrey')
    ax_speed.plot(ts, speeds)
    ax_speed.set_ylabel('Speed (mm/s)')
    ax_speed.set_xticklabels([])
    ax_speed.spines['bottom'].set_visible(False)
    ax_speed_marker = ax_speed.axvline(x=0, color='red')

    # Non-planarity of postures
    ax_nonpp = ax_speed.twinx()
    ax_nonpp.plot(ts, nonp_postures, color='orange', alpha=0.6, linestyle='--')
    ax_nonpp.set_ylabel('NP', rotation=270, labelpad=15)
    ax_nonpp.set_xticklabels([])
    ax_nonpp.axhline(y=0, color='darkgrey')

    # Non-planarity of trajectories
    ax_nonpt = axes[1]
    for i, ws in enumerate(args.planarity_windows):
        ax_nonpt.plot(ts, nonp_trajectories[i], alpha=0.7, label=f'$\Delta$={ws}s')
    ax_nonpt.set_ylabel('NP')
    ax_nonpt.axhline(y=0, color='darkgrey')
    ax_nonpt.legend(loc='upper left', bbox_to_anchor=(1.01, 0.99))
    ax_nonpt.set_xticklabels([])
    ax_nonpt.spines['top'].set_visible(False)
    ax_nonpt.spines['bottom'].set_visible(False)
    ax_nonpt_marker = ax_nonpt.axvline(x=0, color='red')

    # Helicity
    ax_hel = axes[2]
    ax_hel.axhline(y=0, color='darkgrey')
    plot_helicities(
        ax=ax_hel,
        helicities=helicities,
        xs=ts,
        n_fade_lines=args.n_helicity_fade_lines
    )

    label_args = dict(transform=ax_hel.transAxes, horizontalalignment='right', fontweight='bold', fontsize='large',
                      fontfamily='Symbol')
    ax_hel.text(-0.02, 0.94, '↻', verticalalignment='top', **label_args)
    ax_hel.text(-0.02, 0.05, '↺', verticalalignment='bottom', **label_args)
    ax_hel.set_yticks([0, ])
    ax_hel.set_yticklabels([])
    ax_hel.set_xticklabels([])
    ax_hel.spines['top'].set_visible(False)
    ax_hel.spines['bottom'].set_visible(False)
    ax_hel_marker = ax_hel.axvline(x=0, color='red')

    # Curvature
    ax_curvature = axes[3]
    im = ax_curvature.imshow(curvature.T, aspect='auto', cmap='Reds', origin='lower', extent=(0, ts[-1], 0, 1))
    cax = ax_curvature.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax_curvature.transAxes)
    fig.colorbar(im, ax=ax_curvature, cax=cax)
    ht_args = dict(transform=ax_curvature.transAxes, horizontalalignment='right', fontweight='bold')
    ax_curvature.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax_curvature.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax_curvature.set_ylabel('$\kappa$', fontsize=12, labelpad=10)
    ax_curvature.set_yticks([0, 1])
    ax_curvature.set_yticklabels([])
    ax_curvature.set_xticklabels([])
    ax_curvature.spines['top'].set_visible(False)
    ax_curvature.spines['bottom'].set_visible(False)
    ax_curvature_marker = ax_curvature.axvline(x=0, color='red')

    # Torsion
    ax_torsion = axes[4]
    im = ax_torsion.imshow(torsion.T, aspect='auto', cmap='PRGn', origin='lower', extent=(0, ts[-1], 0, 1),
                           norm=MidpointNormalize(midpoint=0))
    cax = ax_torsion.inset_axes([1.03, 0.1, 0.02, 0.8], transform=ax_torsion.transAxes)
    fig.colorbar(im, ax=ax_torsion, cax=cax)
    ht_args = dict(transform=ax_torsion.transAxes, horizontalalignment='right', fontweight='bold')
    ax_torsion.text(-0.02, 0.98, 'T', verticalalignment='top', **ht_args)
    ax_torsion.text(-0.02, 0.01, 'H', verticalalignment='bottom', **ht_args)
    ax_torsion.set_ylabel('$\\tau$', fontsize=12, labelpad=10)
    ax_torsion.set_yticks([0, 1])
    ax_torsion.set_yticklabels([])
    ax_torsion.set_xticklabels([])
    ax_torsion.spines['top'].set_visible(False)
    ax_torsion.spines['bottom'].set_visible(False)
    ax_torsion_marker = ax_torsion.axvline(x=0, color='red')

    # Eigenworms - absolute values
    ax_ew = axes[5]
    for i in args.plot_components:
        ax_ew.plot(
            ts,
            np.abs(X_ew[:, i]),
            label=f'$\lambda_{i + 1}$',
            alpha=0.7,
            linewidth=1
        )
    ax_ew.set_ylabel('$|\lambda|$')
    ax_ew.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax_ew.spines['top'].set_visible(False)
    ax_ew_marker = ax_ew.axvline(x=0, color='red')

    if args.x_label == 'time':
        ax_nonpt.set_xlabel('Time (s)')
        ax_ew.set_xlabel('Time (s)')
    else:
        ax_nonpt.set_xlabel('Frame #')
        ax_ew.set_xlabel('Frame #')

    def update(frame_idx: int):
        # Update the axis limits
        ax_speed.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_nonpt.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_hel.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_curvature.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_torsion.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])
        ax_ew.set_xlim([ts[frame_idx] - t_range / 2, ts[frame_idx] + t_range / 2])

        # Move the markers
        ax_speed_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_nonpt_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_hel_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_curvature_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_torsion_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])
        ax_ew_marker.set_data([ts[frame_idx], ts[frame_idx]], [0, 1])

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def _make_lambdas_plot(
        width: int,
        height: int,
        reconstruction: Reconstruction,
        X_ew: np.ndarray,
        args: Namespace,
) -> Tuple[Figure, Callable]:
    """
    Build a lambdas trace plot.
    Returns an update function to call which updates the axes.
    """
    logger.info('Building lambdas plot.')
    trial = reconstruction.trial
    t_range = int(args.time_range_lambdas * trial.fps)
    x = np.real(X_ew)
    y = np.imag(X_ew)
    traces = []

    # Construct colours
    recency = np.linspace(0, 1, t_range)
    cmap = plt.get_cmap('winter_r')

    # Set up figure
    fig, axes = plt.subplots(2, 2, figsize=(width / 100, height / 100), gridspec_kw={
        'width_ratios': [1, 1],
        'wspace': 0.0,
        'hspace': 0.0,
        'top': 0.99,
        'bottom': 0.01,
        'left': 0.01,
        'right': 0.99,
    })

    # Plot
    for i in range(2):
        for j in range(2):
            l = 2 * i + j
            ax = axes[i, j]
            ax.axvline(x=0, color='lightgrey')
            ax.axhline(y=0, color='lightgrey')
            ax.text(0.03, 0.97, f'$\lambda_{l + 1}$', verticalalignment='top', transform=ax.transAxes,
                    horizontalalignment='left', fontweight='bold', fontsize=12)
            ax.set_xlim(left=x[:, l].min(), right=x[:, l].max())
            ax.set_ylim(bottom=y[:, l].min(), top=y[:, l].max())
            ax.set_xticks([])
            ax.set_yticks([])

            if i == 0:
                ax.spines['bottom'].set_visible(False)
            else:
                ax.spines['top'].set_visible(False)
            if j == 0:
                ax.spines['right'].set_visible(False)
            else:
                ax.spines['left'].set_visible(False)

            Z = np.stack([x[:2, l], y[:2, l]], axis=1)
            points = Z[:, None, :]
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, array=recency, cmap=cmap, alpha=recency)
            lc = ax.add_collection(lc)
            traces.append(lc)

    def update(frame_idx: int):
        start_idx = max(0, frame_idx - t_range)
        for k, tr in enumerate(traces):
            Z = np.stack([x[start_idx:frame_idx, k], y[start_idx:frame_idx, k]], axis=1)
            points = Z[:, None, :]
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            tr.set_segments(segments)

        # Redraw the canvas
        fig.canvas.draw()

    return fig, update


def _generate_annotated_images(
        width: int,
        height: int,
        image_triplet: np.ndarray,
        points_2d: np.ndarray
) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    images = generate_annotated_images(image_triplet, points_2d)
    panel = np.ones((height, width, 3), dtype=np.uint8) * 255
    rh = height / images.shape[0]
    rw = width / images.shape[1]
    if images.shape[0] * rw > height:
        images = cv2.resize(images, None, fx=rh, fy=rh)
        new_width = images.shape[1]
        offset = width - new_width
        panel[:, offset:offset + new_width] = images
    else:
        images = cv2.resize(images, None, fx=rw, fy=rw)
        new_height = images.shape[0]
        offset = height - new_height
        panel[offset:offset + new_height] = images

    return panel


def generate_reconstruction_video():
    """
    Generate a reconstruction video showing a rotating 3D trajectory with reconstructed
    worm moving along it and camera images with overlaid 2D midline reprojections.
    """
    args = get_args()
    reconstruction = Reconstruction.objects.get(id=args.reconstruction)
    trial = reconstruction.trial

    # Set frame range - using valid range if set
    if args.use_valid_range and reconstruction.start_frame_valid is not None:
        r_start_frame = reconstruction.start_frame_valid
        r_end_frame = reconstruction.end_frame_valid
    else:
        r_start_frame = reconstruction.start_frame
        r_end_frame = reconstruction.end_frame
    if args.start_frame is None:
        start_frame = r_start_frame
    else:
        start_frame = max(args.start_frame, r_start_frame)
    if args.end_frame is None:
        end_frame = r_end_frame
    else:
        end_frame = min(args.end_frame, r_end_frame)

    # Fetch trajectory and postures for full sequence
    common_args = {
        'reconstruction_id': reconstruction.id,
        'start_frame': r_start_frame,
        'end_frame': r_end_frame,
    }
    X_raw, _ = get_trajectory(**common_args)
    if args.smoothing_window_postures is not None and args.smoothing_window_postures > 1:
        X = smooth_trajectory(X_raw, window_len=args.smoothing_window_postures)
    else:
        X = X_raw
    Xc = X - X.mean(axis=0)

    # Eigenworm projections
    ew = generate_or_load_eigenworms(
        eigenworms_id=args.eigenworms,
        reconstruction_id=reconstruction.id,
        n_components=args.n_components,
        regenerate=False
    )
    Z, _ = get_trajectory(**common_args, natural_frame=True, smoothing_window=args.smoothing_window_components)
    X_ew = ew.transform(Z)

    # Calculate parameters
    logger.info('Calculating/loading values.')
    if reconstruction.source == M3D_SOURCE_MF:
        ts = TrialState(reconstruction)
        points_3d = ts.get('points')
        if args.smoothing_window_postures is not None and args.smoothing_window_postures > 1:
            points_3d = smooth_trajectory(points_3d, args.smoothing_window_postures)
        points_3d_base = ts.get('points_3d_base')
        points_2d_base = ts.get('points_2d_base')
        lengths = ts.get('length', r_start_frame, r_end_frame + 1)[:, 0]
        cam_coeffs = np.concatenate([
            ts.get(f'cam_{k}')
            for k in ['intrinsics', 'rotations', 'translations', 'distortions', 'shifts', ]
        ], axis=2)
        prs = ProjectRenderScoreModel(image_size=trial.crop_size)
    else:
        lengths = np.linalg.norm(Xc[:, 1:] - Xc[:, :-1], axis=-1).sum(axis=-1)

    # Calculate speed
    if args.smoothing_window_speed is not None and args.smoothing_window_speed > 1:
        Xc_raw = X_raw - X_raw.mean(axis=0)
        Xc_smoothed = smooth_trajectory(Xc_raw, window_len=args.smoothing_window_speed)
        speeds = calculate_speeds(Xc_smoothed, signed=True) * trial.fps
    else:
        speeds = calculate_speeds(Xc, signed=True) * trial.fps

    # Calculate curvatures of postures and trajectory
    X_com = Xc.mean(axis=1)
    e0 = normalise(np.gradient(X_com, axis=0))
    e0[speeds < 0] *= -1
    curvature_traj = calculate_curvature(e0)
    curvature_postures = np.abs(Z)

    # Calculate torsions
    psi = np.unwrap(np.angle(Z), axis=1)
    torsion = np.gradient(psi, axis=1)

    # Calculate posture helicities
    helicities = calculate_helicities(Xc)

    # Build plots
    fig_info, update_info_plot = _make_info_panel(
        width=int(args.width / 4),
        height=int(args.height / 3),
        trial=trial,
        start_frame_trial=start_frame,
        X=Xc,
        lengths=lengths
    )
    traj_fn_args = dict(
        width=int(args.width / 3),
        height=int(args.height / 3 * 2),
        trial=trial,
        X=Xc,
        speeds=speeds,
        curvature=curvature_traj,
        args=args
    )
    if args.trajectory_use_mlab:
        fig_traj, update_traj_plot = _make_3d_trajectory_plot_mlab(
            curvature_postures=curvature_postures,
            lengths=lengths,
            **traj_fn_args
        )
    else:
        fig_traj, update_traj_plot = _make_3d_trajectory_plot(**traj_fn_args)
    posture_fn_args = dict(
        width=int(args.width / 12 * 3),
        height=int(args.height / 3 * 2),
        trial=trial,
        X=Xc - X_com[:, None],
        args=args
    )
    if args.posture_use_mlab:
        fig_posture, update_posture_plot = _make_3d_posture_plot_mlab(
            curvature=curvature_postures,
            lengths=lengths,
            **posture_fn_args
        )
    else:
        fig_posture, update_posture_plot = _make_3d_posture_plot(**posture_fn_args)
    fig_traces, update_traces_plot = _make_traces_plots(
        width=int(args.width) / 12 * 5,
        height=int(args.height / 3 * 2),
        reconstruction=reconstruction,
        X=Xc,
        X_ew=X_ew,
        speeds=speeds,
        helicities=helicities,
        curvature=curvature_postures,
        torsion=torsion,
        args=args
    )
    fig_lambdas, update_lambdas_plot = _make_lambdas_plot(
        width=int(args.width / 4),
        height=int(args.height / 3),
        reconstruction=reconstruction,
        X_ew=X_ew,
        args=args
    )

    # Fetch the frames
    logger.info('Querying database for frames.')
    pipeline = [
        {'$match': {
            'trial': trial.id,
            'frame_num': {
                '$gte': start_frame,
                '$lte': end_frame,
            }
        }},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
        }},
    ]
    cursor = Frame.objects().aggregate(pipeline, allowDiskUse=True)

    # Initialise ffmpeg process
    output_dir = LOGS_PATH / f'trial_{trial.id:03d}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f'{START_TIMESTAMP}_trial={trial.id}_r={reconstruction.id}_f={start_frame}-{end_frame}'
    output_args = {
        'pix_fmt': 'yuv444p',
        'vcodec': 'libx264',
        'r': trial.fps,
        'metadata:g:0': f'title=Trial {trial.id}. Reconstruction {reconstruction.id}.',
        'metadata:g:1': 'artist=Leeds Wormlab',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f'{args.width}x{args.height}')
            .output(str(output_path) + '.mp4', **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    frame_nums = []
    logger.info('Rendering frames.')
    for i, res in enumerate(cursor):
        n = res['frame_num']
        if i > 0 and i % 100 == 0:
            logger.info(f'Rendering frame {n} - {i}/{end_frame - start_frame}.')

        # Check we don't miss any frames
        if i == 0:
            assert n == start_frame
            n0 = n
        assert n == n0 + i

        # Check images are present
        img_path = PREPARED_IMAGES_PATH / f'{trial.id:03d}' / f'{n:06d}.npz'
        try:
            image_triplet = np.load(img_path)['images']
            if image_triplet.shape != (3, trial.crop_size, trial.crop_size):
                logger.warning('Prepared images are the wrong size, regeneration needed!')
                raise RuntimeError()
        except Exception:
            logger.warning('Prepared images not available, stopping here.')
            break
        frame_nums.append(n)

        # Generate the annotated images
        if reconstruction.source == M3D_SOURCE_MF:
            points_2d = prs._project_to_2d(
                cam_coeffs=torch.from_numpy(cam_coeffs[n][None, ...]),
                points_3d=torch.from_numpy(points_3d[n][None, ...]),
                points_3d_base=torch.from_numpy(points_3d_base[n][None, ...].astype(np.float32)),
                points_2d_base=torch.from_numpy(points_2d_base[n][None, ...].astype(np.float32)),
            )
            points_2d = points_2d[0].numpy().transpose(1, 0, 2)
            points_2d = np.round(points_2d).astype(np.int32)

        else:
            m3d = Midline3D.objects.get(
                frame=res['_id'],
                source=reconstruction.source,
                source_file=reconstruction.source_file,
            )

            # Get 2D projections
            points_2d = np.round(m3d.prepare_2d_coordinates(X=X[i])).astype(np.int32)
            points_2d = points_2d.transpose(1, 0, 2)

        # Prepare images
        images = _generate_annotated_images(
            width=int(args.width / 4 * 3),
            height=int(args.height / 3),
            image_triplet=image_triplet,
            points_2d=points_2d
        )

        # Update the plots and extract renders
        idx = start_frame - r_start_frame + i
        update_info_plot(idx)
        update_traj_plot(idx)
        update_posture_plot(idx)
        update_traces_plot(idx)
        update_lambdas_plot(idx)

        plot_info = np.asarray(fig_info.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_traces = np.asarray(fig_traces.canvas.renderer._renderer).take([0, 1, 2], axis=2)
        plot_lambdas = np.asarray(fig_lambdas.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        if args.posture_use_mlab:
            plot_posture = mlab.screenshot(mode='rgb', antialiased=True, figure=fig_posture)
            plot_posture = cv2.resize(plot_posture, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        else:
            plot_posture = np.asarray(fig_posture.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        if args.trajectory_use_mlab:
            plot_traj = mlab.screenshot(mode='rgb', antialiased=True, figure=fig_traj)
            plot_traj = cv2.resize(plot_traj, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        else:
            plot_traj = np.asarray(fig_traj.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        # Overlay plot info on top of images panel
        images = overlay_image(images, plot_info, x_offset=0, y_offset=0)

        # Join plots and images and write to stream
        frame = np.concatenate([
            np.concatenate([images, plot_lambdas], axis=1),
            np.concatenate([plot_posture, plot_traj, plot_traces], axis=1),
        ], axis=0)
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    # Write meta data
    meta = to_dict(args)
    meta['created'] = START_TIMESTAMP
    with open(output_path.with_suffix('.meta'), 'w') as f:
        json.dump(meta, f, indent=2, separators=(',', ': '))

    logger.info(f'Generated video for frames {frame_nums[0]}-{frame_nums[-1]} ({frame_nums[-1] - frame_nums[0]}). '
                f'Total frames in reconstruction = {reconstruction.n_frames}.')


if __name__ == '__main__':
    os.makedirs(LOGS_PATH, exist_ok=True)
    generate_reconstruction_video()
