import os
from argparse import Namespace
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib import transforms
from mayavi import mlab
from scipy.stats import norm
from tvtk.tools import visual

from simple_worm.frame import FrameNumpy, FRAME_COMPONENT_KEYS
from simple_worm.plot3d import cla, FrameArtist, MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, CAMERA_IDXS, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Midline3D, Reconstruction, Frame
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d_mlab, FrameArtistMLab, \
    plot_arrow
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import parse_target_arguments
from wormlab3d.trajectories.util import fetch_reconstruction, smooth_trajectory

img_extension = 'svg'
show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
invert = True

if save_plots:
    os.makedirs(LOGS_PATH, exist_ok=True)


def get_midline(
        args: Namespace = None
) -> Midline3D:
    """
    Find a midline for a given trial and frame num, or by id.
    """
    if args is None:
        args = parse_target_arguments()
    frame = None
    reconstruction = None
    if args.midline3d is not None:
        midline = Midline3D.objects.get(id=args.midline3d)
        frame = midline.frame
        reconstruction = midline.get_reconstruction()

    # Fetch reconstruction and trial
    if reconstruction is None:
        reconstruction = fetch_reconstruction(
            reconstruction_id=args.reconstruction,
            trial_id=args.trial,
            midline_source=args.midline3d_source,
            midline_source_file=args.midline3d_source_file,
        )

    # Fetch frame
    if frame is None:
        if args.frame_num is None:
            frame_num = np.random.randint(reconstruction.start_frame, reconstruction.end_frame)
            logger.info(f'Selected frame num = {frame_num} at random.')
        else:
            frame_num = args.frame_num
        frame = reconstruction.trial.get_frame(frame_num)

    # Fetch points
    if reconstruction.source == M3D_SOURCE_MF:
        ts = TrialState(reconstruction)
        n = frame.frame_num
        X = ts.get('points', n, n + 1)[0].copy()
        points_2d = ts.get('points_2d', n, n + 1)[0].copy()
        points_2d = np.round(points_2d).astype(np.int32)

    else:
        m3d = Midline3D.objects.get(
            frame=frame.id,
            source=reconstruction.source,
            source_file=reconstruction.source_file,
        )
        X = m3d.X
        points_2d = np.round(m3d.get_prepared_2d_coordinates()).astype(np.int32)
        points_2d = points_2d.transpose(1, 0, 2)

    return reconstruction, frame, X, points_2d


def plot_3d(
        reconstruction: Reconstruction,
        frame: Frame,
        X: np.ndarray
):
    """
    3D plot of a midline.
    """
    trial = reconstruction.trial
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(
        f'Trial: {trial.id}. \n'
        f'Frame: {frame.frame_num}'
    )

    F = FrameNumpy(x=X.T)
    fa = FrameArtist(F=F)
    fa.add_midline(ax)
    equal_aspect_ratio(ax)
    cla(ax)
    fig.tight_layout()

    if save_plots:
        save_path = LOGS_PATH / f'{START_TIMESTAMP}' \
                                f'_trial={trial.id}' \
                                f'_frame={frame.frame_num}' \
                                f'_reconstruction={reconstruction.id}' \
                                f'_3D.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_3d_mlab(
        reconstruction: Reconstruction,
        frame: Frame,
        X: np.ndarray,
        interactive: bool = True,
        transparent_bg: bool = True,
):
    """
    3D plot of a midline using mayavi.
    """
    trial = reconstruction.trial
    NF = NaturalFrame(X)

    # 3D plot of eigenworm
    fig = plot_natural_frame_3d_mlab(
        NF,
        azimuth=155,
        elevation=165,
        roll=155,
        distance=1.1,
        midline_opts={'line_width': 18, 'opacity': 1, 'tube_radius': 0.003},
        mesh_opts={'opacity': 0.4},
        show_frame_arrows=False,
        show_pca_arrows=False,
        show_outline=False,
        show_axis=False,
        offscreen=not interactive,
    )

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}' \
                           f'_trial={trial.id}' \
                           f'_frame={frame.frame_num}' \
                           f'_reconstruction={reconstruction.id}' \
                           f'_3D.{img_extension}'
        logger.info(f'Saving plot to {path}.')

        if not transparent_bg:
            mlab.savefig(str(path), figure=fig)
        else:
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
            img.save(path)

    if show_plots:
        if interactive:
            mlab.show()
        else:
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            fig_mpl = plt.figure(figsize=(10, 10))
            ax = fig_mpl.add_subplot()
            ax.imshow(img)
            ax.axis('off')
            fig_mpl.tight_layout()
            plt.show()
            plt.close(fig_mpl)

    mlab.clf(fig)
    mlab.close()


def plot_3d_with_points_mlab(
        reconstruction: Reconstruction,
        frame: Frame,
        X: np.ndarray,
        interactive: bool = True,
        transparent_bg: bool = True,
        n_points: int = 33
):
    """
    3D plot of a midline using mayavi.
    """
    trial = reconstruction.trial
    NF = NaturalFrame(X)

    # 3D plot of worm
    logger.info('Building 3D plot.')
    fig = plot_natural_frame_3d_mlab(
        NF,
        azimuth=155,
        elevation=165,
        roll=155,
        distance=1.1,
        mesh_opts={'opacity': 0.4},
        show_midline=False,
        show_surface=True,
        show_frame_arrows=False,
        show_frame_e0=True,
        n_frame_arrows=1,
        arrow_opts={
            'opacity': 0.9,
            'radius_shaft': 0.02,
            'radius_cone': 0.1,
            'length_cone': 0.2
        },
        arrow_scale=0.1,
        show_pca_arrows=False,
        show_outline=False,
        show_axis=False,
        offscreen=not interactive,
    )

    # Show a subset of the vertices
    logger.info('Adding vertex points')
    if 0 < n_points < NF.N:
        idxs = np.round(np.linspace(0, NF.N - 1, n_points)).astype(int)
    else:
        idxs = range(NF.N)
    X_subset = NF.X_pos[idxs]

    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
    x, y, z = X_subset.T
    t = np.linspace(0, 1, len(idxs))
    points = mlab.points3d(x, y, z, scale_factor=0.02, figure=fig)
    points.glyph.scale_mode = 'scale_by_vector'
    points.mlab_source.dataset.point_data.scalars = t
    points.module_manager.scalar_lut_manager.lut.table = cmaplist

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}' \
                           f'_trial={trial.id}' \
                           f'_frame={frame.frame_num}' \
                           f'_reconstruction={reconstruction.id}' \
                           f'_3D.{img_extension}'
        logger.info(f'Saving plot to {path}.')

        if not transparent_bg:
            mlab.savefig(str(path), figure=fig)
        else:
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
            img.save(path)

    if show_plots:
        if interactive:
            mlab.show()
        else:
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            fig_mpl = plt.figure(figsize=(10, 10))
            ax = fig_mpl.add_subplot()
            ax.imshow(img)
            ax.axis('off')
            fig_mpl.tight_layout()
            plt.show()
            plt.close(fig_mpl)

    mlab.clf(fig)
    mlab.close()


def plot_3d_construction(
        reconstruction: Reconstruction,
        frame: Frame,
        X: np.ndarray,
        n_points: int = 33,
        start_idx: int = 32,
        n_stages: int = 3,
        curvature_smoothing: int = 5,
):
    """
    3D plots of the midline construction process.
    """
    trial = reconstruction.trial
    NF = NaturalFrame(X)

    # Get vertex idxs
    if 0 < n_points < NF.N:
        vertex_idxs = np.round(np.linspace(0, NF.N - 1, n_points)).astype(int)
    else:
        vertex_idxs = range(NF.N)
    start_idx = vertex_idxs[np.argmin(np.abs(vertex_idxs - start_idx))]
    vertex_idxs_h = vertex_idxs[vertex_idxs <= start_idx]
    vertex_idxs_t = vertex_idxs[vertex_idxs >= start_idx]
    frame_idxs_h = vertex_idxs_h[np.round(np.linspace(0, 1, n_stages) * (len(vertex_idxs_h) - 1)).astype(int)][::-1]
    frame_idxs_t = vertex_idxs_t[np.round(np.linspace(0, 1, n_stages) * (len(vertex_idxs_t) - 1)).astype(int)]

    # Plot options
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    cmaplist = np.array([cmap(i) for i in range(cmap.N)]) * 255
    mlab.options.offscreen = not show_plots
    m1_colour = 'deepskyblue'
    m2_colour = 'mediumseagreen'

    # Configure save path
    save_dir = LOGS_PATH / f'{START_TIMESTAMP}' \
                           f'_trial={trial.id}' \
                           f'_frame={frame.frame_num}' \
                           f'_reconstruction={reconstruction.id}' \
                           f'_3D_construction'
    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # Plot curvatures
    plt.rc('axes', labelsize=6)  # fontsize of the axis labels
    plt.rc('xtick', labelsize=6)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=1, size=2)
    plt.rc('legend', fontsize=5.5)  # fontsize of the legend

    fig, ax = plt.subplots(1, figsize=(1.2, 0.8), gridspec_kw={
        'left': 0.22,
        'right': 0.96,
        'top': 0.95,
        'bottom': 0.21,
    })
    ax.spines['top'].set_visible(False)
    ax.plot(smooth_trajectory(NF.m1, window_len=curvature_smoothing), color=m1_colour, label='$m^1$')
    ax.plot(smooth_trajectory(NF.m2, window_len=curvature_smoothing), color=m2_colour, label='$m^2$')

    # Set up x-axis
    ax.set_xticks([])
    ax.set_xlim(left=0, right=NF.N - 1)
    ax.set_xticks([0, NF.N - 1])
    ax.set_xticklabels(['H', 'T'])

    # Add n0 label
    if start_idx != 0:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(start_idx, -0.08, '$n_0$', color=cmaplist[start_idx] / 255, fontsize=7, fontweight='bold',
                horizontalalignment='center', verticalalignment='top', transform=trans)

    # Set up y-axis
    ylim = max(np.concatenate([np.abs(NF.m1), np.abs(NF.m1)])) * 1.05
    ax.set_ylim(bottom=-ylim, top=ylim)
    ax.set_yticks([-5, 0, 5])
    if start_idx != 0:
        ax.axvline(x=start_idx, ymin=-0.1, ymax=0.92, linestyle=':', color='grey')
    ax.set_ylabel('Curvature (mm$^{-1}$)', labelpad=-1)

    # Legend
    ax.legend(loc='lower right', ncol=2)

    if save_plots:
        path = save_dir / f'curvatures.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()

    def _make_plot(htf: str, stage: int = 0, frame_idx: int = 0):
        # 3D plot of worm
        logger.info('Building 3D plot.')
        fig = mlab.figure(size=(2000, 1100), bgcolor=(1, 1, 1))
        if 1:
            # Doesn't really seem to make any difference
            fig.scene.render_window.point_smoothing = True
            fig.scene.render_window.line_smoothing = True
            fig.scene.render_window.polygon_smoothing = True
            fig.scene.render_window.multi_samples = 20
            fig.scene.anti_aliasing_frames = 20
        visual.set_viewer(fig)

        # Set up the artist and add the pieces
        fa = FrameArtistMLab(
            NF,
            midline_opts={
                'opacity': 0.7,
                'tube_radius': 0.003,
            },
            mesh_opts={'opacity': 0.3},
            arrow_opts={
                'opacity': 0.9,
                'radius_shaft': 0.03,
                'radius_cone': 0.13,
                'length_cone': 0.22
            },
            arrow_colours={
                'e0': 'grey',
                'e1': m1_colour,
                'e2': m2_colour,
            },
            arrow_scale=0.15,
        )

        # Add the midline and surface
        fa.add_midline(fig)
        fa.add_surface(fig)

        # Add the component vectors
        if htf != 'f':
            for k in FRAME_COMPONENT_KEYS:
                vec, colours = fa.get_vectors_and_colours(k)
                plot_arrow(
                    fig=fig,
                    origin=fa.X[frame_idx],
                    vec=vec[frame_idx],
                    color=colours[frame_idx],
                    **fa.arrow_opts
                )

        # Show a subset of the vertices
        logger.info('Adding vertex points')
        if htf == 'h':
            show_idxs = vertex_idxs_h[vertex_idxs_h >= frame_idx]
            t = np.linspace(frame_idx / NF.N, start_idx / NF.N, len(show_idxs))
        elif htf == 't':
            show_idxs = vertex_idxs_t[vertex_idxs_t <= frame_idx]
            t = np.linspace(start_idx / NF.N, frame_idx / NF.N, len(show_idxs))
        else:
            show_idxs = vertex_idxs
            t = np.linspace(0, 1, len(show_idxs))

        X_subset = NF.X_pos[show_idxs]
        points = mlab.points3d(*X_subset.T, scale_factor=0.025, figure=fig)
        points.glyph.scale_mode = 'scale_by_vector'
        points.mlab_source.dataset.point_data.scalars = t
        points.module_manager.scalar_lut_manager.lut.table = cmaplist

        # Render and save
        centre = fa.X.min(axis=0) + fa.X.ptp(axis=0) / 2
        mlab.view(
            azimuth=-90,
            elevation=80,
            roll=-115,
            distance=1.1,
            focalpoint=centre
        )

        if save_plots:
            if htf == 'f':
                path = save_dir / 'finished.png'
            else:
                path = save_dir / f'{htf}_{stage}_{frame_idx:03d}.png'
            logger.info(f'Saving plot to {path}.')
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
            img.save(path)

        if show_plots:
            mlab.show()

        mlab.clf(fig)

    # Plot the different stages
    for i in range(n_stages):
        _make_plot('h', i, frame_idxs_h[i])
        _make_plot('t', i, frame_idxs_t[i])

    # Plot the completed curve
    _make_plot('f')


def plot_n0_distribution(
        start_idx: int = 32
):
    """
    Plot the distribution from which we sample the n0 start point.
    """
    N = 128
    N_sample_points = N
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    ind = np.arange(N)

    dist = norm(loc=N / 2, scale=N / 8)
    x = np.linspace(0, N, N_sample_points)
    vals = dist.pdf(x)

    # Plot curvatures
    plt.rc('axes', labelsize=6)  # fontsize of the y label
    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=1, size=2)

    fig, ax = plt.subplots(1, figsize=(1.1, 0.8), gridspec_kw={
        'left': 0.2,
        'right': 0.96,
        'top': 0.98,
        'bottom': 0.21,
    })
    ax.spines['top'].set_visible(False)
    for n in range(N - 1):
        ax.plot(ind[n:n + 2], vals[n:n + 2], c=fc[n])

    # Set up x-axis
    ax.set_xticks([])
    ax.set_xlim(left=0, right=N - 1)
    ax.set_xticks([0, N - 1])
    ax.set_xticklabels(['H', 'T'])

    # Set up y-axis
    ax.set_ylim(bottom=0, top=vals.max() * 1.1)
    ax.set_yticks([0, 0.01, 0.02])
    ax.set_yticklabels([])
    ax.set_ylabel('Density')
    ax.vlines(x=start_idx, ymin=-0.1, ymax=vals[start_idx], linestyle=':', color='grey')

    # Add n0 label
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(start_idx, -0.08, '$n_0$', color=fc[start_idx], fontsize=7, fontweight='bold',
            horizontalalignment='center', verticalalignment='top', transform=trans)

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}_n0_dist.{img_extension}'
        logger.info(f'Saving plot to {path}.')
        plt.savefig(path, transparent=True)
    if show_plots:
        plt.show()


def plot_3d_with_pca(
        reconstruction: Reconstruction,
        frame: Frame,
        X: np.ndarray,
        interactive: bool = True,
        transparent_bg: bool = True,
):
    """
    3D plot of a midline with pca arrows.
    """
    trial = reconstruction.trial
    NF = NaturalFrame(X)

    # 3D plot of midline
    fig = plot_natural_frame_3d_mlab(
        NF,
        azimuth=-150,
        elevation=100,
        roll=60,
        distance=1.65,
        show_frame_arrows=False,
        show_pca_arrows=True,
        show_pca_arrow_labels=False,
        show_midline=True,
        show_outline=True,
        show_axis=False,
        midline_opts={'tube_radius': 0.01, 'opacity': 1},
        surface_opts={'radius': 0.04},
        mesh_opts={'opacity': 0.5},
        offscreen=not interactive,
    )

    if save_plots:
        path = LOGS_PATH / f'{START_TIMESTAMP}' \
                           f'_trial={trial.id}' \
                           f'_frame={frame.frame_num}' \
                           f'_reconstruction={reconstruction.id}' \
                           f'_3D_pca.{img_extension}'
        logger.info(f'Saving plot to {path}.')

        if not transparent_bg:
            mlab.savefig(str(path), figure=fig)
        else:
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
            img.save(path)
            mlab.clf(fig)
            mlab.close()

    if show_plots:
        if interactive:
            mlab.show()
        else:
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            mlab.clf(fig)
            mlab.close()
            fig_mpl = plt.figure(figsize=(10, 10))
            ax = fig_mpl.add_subplot()
            ax.imshow(img)
            ax.axis('off')
            fig_mpl.tight_layout()
            plt.show()
            plt.close(fig_mpl)


def plot_3d_pca_sequence(
        frame_nums: List[int]
):
    """
    Plot a series of 3D worms with pca arrows
    """
    args = parse_target_arguments()

    for frame_num in frame_nums:
        args.frame_num = frame_num
        reconstruction, frame, X, points_2d = get_midline(args)
        plot_3d_with_pca(
            reconstruction=reconstruction,
            frame=frame,
            X=X,
            interactive=False,
        )


def plot_reprojections(
        reconstruction: Reconstruction,
        frame: Frame,
        points_2d: np.ndarray,
):
    """
    Plot the 3D midline reprojected back down on top of the images.
    """
    trial = reconstruction.trial
    if len(frame.images) != 3:
        frame.generate_prepared_images()
        frame.save()

    fig, axes = plt.subplots(3, figsize=(6, 8))
    fig.suptitle(
        f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
        f'Frame: {frame.frame_num}'
    )

    images = generate_annotated_images(frame.images, points_2d)
    images = np.split(images, 3, axis=1)

    for c in CAMERA_IDXS:
        ax = axes[c]
        ax.set_title(f'Camera {c}')
        ax.imshow(images[c], cmap='gray', vmin=0, vmax=1)

    fig.tight_layout()

    if save_plots:
        save_path = LOGS_PATH / f'{START_TIMESTAMP}' \
                                f'_trial={trial.id}' \
                                f'_frame={frame.frame_num}' \
                                f'_reconstruction={reconstruction.id}' \
                                f'_2D.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_reprojection_singles(
        reconstruction: Reconstruction,
        frame: Frame,
        points_2d: np.ndarray,
        with_image: bool = True,
        with_midline: bool = True,
        n_points: int = -1,
        point_radius: int = 3,
        point_alpha: float = 0.6
):
    """
    Show/save the 3D midline reprojected back down on top of the images.
    """
    if len(frame.images) != 3:
        frame.generate_prepared_images()
        frame.save()
    trial = reconstruction.trial

    N = points_2d.shape[0]
    if 0 < n_points < N:
        idxs = np.round(np.linspace(0, N - 1, n_points)).astype(int)
    else:
        idxs = range(N)

    cmap_midline = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = np.array([cmap_midline(i) for i in np.linspace(0, 1, len(idxs))])
    colours = np.round(colours * 255).astype(np.uint8)
    for c, img_array in enumerate(frame.images):
        if not with_image:
            img_array.fill(0)
        if invert:
            img_array = 1 - img_array
        z = (img_array * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGRA)

        if not with_image:
            z[..., -1].fill(0)

        # Overlay 2d projection
        if with_midline:
            p2d = points_2d[idxs, c]
            for j, p in enumerate(p2d):
                col = colours[j].tolist()

                # Draw markers and connecting lines
                if n_points == -1:
                    img = cv2.drawMarker(
                        img,
                        p,
                        color=col,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=2,
                        thickness=1,
                        line_type=cv2.LINE_AA
                    )
                    if j > 0:
                        cv2.line(img, p2d[j - 1], p2d[j], color=col, thickness=1, lineType=cv2.LINE_AA)

                # Draw points
                else:
                    if invert:
                        tmp = np.ones_like(z) * 255
                    else:
                        tmp = np.zeros_like(z)
                    tmp[..., -1].fill(0)
                    tmp = cv2.circle(tmp, p, point_radius, color=col, thickness=-1, lineType=cv2.LINE_AA)
                    tmpz = cv2.addWeighted(z, 1 - point_alpha, tmp, point_alpha, 0)
                    m = (tmp[..., -1] > 10)[..., None]
                    z = z * ~m + tmpz * m
                    z = cv2.circle(z, p, point_radius, color=col, thickness=1, lineType=cv2.LINE_AA)

        # Convert to PIL image
        img = Image.fromarray(z, 'RGBA')

        if save_plots:
            save_path = LOGS_PATH / (f'{START_TIMESTAMP}'
                                     f'_trial={trial.id}'
                                     f'_frame={frame.frame_num}'
                                     f'_reconstruction={reconstruction.id}'
                                     + ('_no_midline' if not with_midline else '')
                                     + ('_no_image' if not with_image else '')
                                     + f'_2D_c={c}.png')
            logger.info(f'Saving image to {save_path}.')
            img.save(save_path)

        if show_plots:
            img.show()


if __name__ == '__main__':
    # from wormlab3d.toolkit.plot_utils import interactive_plots
    # interactive_plots()
    reconstruction_, frame_, X_, points_2d_ = get_midline()
    # plot_3d(reconstruction_, frame_, X_)
    # plot_3d_mlab(reconstruction_, frame_, X_, interactive=False, transparent_bg=True)
    # plot_3d_with_points_mlab(reconstruction_, frame_, X_, interactive=False, transparent_bg=True, n_points=33)
    plot_3d_construction(reconstruction_, frame_, X_, n_points=33, start_idx=0, n_stages=4, curvature_smoothing=11)
    # plot_n0_distribution(start_idx=40)
    # plot_3d_with_pca(reconstruction_, frame_, X_)
    # plot_reprojections(reconstruction_, frame_, points_2d_)
    # plot_reprojection_singles(reconstruction_, frame_, points_2d_)
    # plot_reprojection_singles(reconstruction_, frame_, points_2d_, with_image=False, n_points=33, point_radius=5, point_alpha=0.7)
    # plot_reprojection_singles(reconstruction_, frame_, points_2d_, with_midline=False)

    # plot_3d_pca_sequence(frame_nums=[13788,13854,13931])  # (azim, elev, roll) = (10, 100, -10)
    # plot_3d_pca_sequence(frame_nums=[14378, 14474, 14533])
    # plot_3d_pca_sequence(frame_nums=[13931,])
