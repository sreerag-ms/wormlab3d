import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from mayavi import mlab

from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import cla, FrameArtist, MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, CAMERA_IDXS, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Midline3D, Reconstruction, Frame
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.midlines3d.util import generate_annotated_images
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import plot_natural_frame_3d, plot_natural_frame_3d_mlab
from wormlab3d.toolkit.plot_utils import equal_aspect_ratio
from wormlab3d.toolkit.util import parse_target_arguments
from wormlab3d.trajectories.util import fetch_reconstruction

img_extension = 'png'
show_plots = False
save_plots = True
invert = True

if save_plots:
    os.makedirs(LOGS_PATH, exist_ok=True)


def get_midline() -> Midline3D:
    """
    Find a midline for a given trial and frame num, or by id.
    """
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


def plot_3d_with_pca(
        reconstruction: Reconstruction,
        frame: Frame,
        X: np.ndarray,
):
    """
    3D plot of a midline with pca arrows.
    """
    trial = reconstruction.trial
    NF = NaturalFrame(X)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d', azim=-110, elev=40)  # azim=-60, elev=30)
    ax = plot_natural_frame_3d(
        NF,
        ax=ax,
        show_frame_arrows=False,
        show_pca_arrows=True,
        show_pca_arrow_labels=False,
    )
    ax.axis('off')

    if save_plots:
        save_path = LOGS_PATH / f'{START_TIMESTAMP}' \
                                f'_trial={trial.id}' \
                                f'_frame={frame.frame_num}' \
                                f'_reconstruction={reconstruction.id}' \
                                f'_3D_pca.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path, transparent=True)

    if show_plots:
        plt.show()


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
    # plot_3d_with_pca(reconstruction_, frame_, X_)
    # plot_reprojections(reconstruction_, frame_, points_2d_)
    # plot_reprojection_singles(reconstruction_, frame_, points_2d_)
    plot_reprojection_singles(reconstruction_, frame_, points_2d_, with_image=False, n_points=33)
    # plot_reprojection_singles(reconstruction_, frame_, points_2d_, with_midline=False, n_points=33)
