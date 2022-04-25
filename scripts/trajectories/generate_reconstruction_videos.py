import os
import time
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from simple_worm.frame import FrameSequenceNumpy
from simple_worm.plot3d import FrameArtist
from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, PREPARED_IMAGES_PATH
from wormlab3d.data.model import Frame, Reconstruction, Trial
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF, Midline3D
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.toolkit.plot_utils import tex_mode, equal_aspect_ratio
from wormlab3d.trajectories.cache import get_trajectory

tex_mode()
n_revolutions = 2
cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)


def get_reconstruction_ids() -> List[str]:
    parser = ArgumentParser()
    parser.add_argument('--reconstruction', type=str, help='Reconstruction by id.')
    args = parser.parse_args()
    if args.reconstruction is not None:
        reconstruction_ids = [args.reconstruction]
    else:
        reconstruction_ids = [reconstruction.id for reconstruction in Reconstruction.objects]
    return reconstruction_ids


def _make_3d_plot(
        trial: Trial,
        width: int,
        height: int,
        X_slice: np.ndarray,
        X_full: np.ndarray,
        lengths: np.ndarray,
        colours: np.ndarray = None,
        cmap: str = 'viridis_r',
        show_colourbar: bool = False,
        draw_edges: bool = True,
        show_axis: bool = True,
        show_ticks: bool = True
):
    """
    Build a 3D trajectory plot with worm.
    Returns an update function to call which rotates the view and updates the worm.
    """
    x, y, z = X_slice.T

    # Construct colours
    if colours is None:
        colours = np.linspace(0, 1, len(X_slice))
    cmap = plt.get_cmap(cmap)
    c = [cmap(c_) for c_ in colours]

    # Create figure
    fig = plt.figure(figsize=(int(width / 100), int(height * 2 / 3 / 100)))
    ax = fig.add_subplot(projection='3d')

    # Scatter the vertices
    s = ax.scatter(x, y, z, c=c, s=5, alpha=0.4, zorder=-1)
    if show_colourbar:
        fig.colorbar(s)

    # Draw lines connecting points
    if draw_edges:
        points = X_slice[:, None, :]
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(segments, array=colours, cmap=cmap, zorder=-2, alpha=0.3)
        ax.add_collection(lc)

    # Add worm
    FS = FrameSequenceNumpy(x=X_full.transpose(0, 2, 1))
    fa = FrameArtist(F=FS[0])
    fa.add_midline(ax)

    # Setup axis
    equal_aspect_ratio(ax)
    if not show_axis:
        ax.axis('off')
    if not show_ticks:
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Aspects
    azims = np.linspace(start=0, stop=360 * n_revolutions, num=len(X_slice))
    ax.view_init(azim=azims[0])  # elev

    def get_details(frame_idx: int) -> str:
        curr_time = datetime.fromtimestamp(np.floor(frame_idx / trial.fps))
        total_time = datetime.fromtimestamp(np.floor(len(X_slice) / trial.fps))
        return f'Frame {frame_idx + 1}/{len(X_slice)}\n' \
               f'Time {curr_time:%M:%S}/{total_time:%M:%S}\n' \
               f'Length: {lengths[frame_idx]:.3f}'

    # Details
    text = fig.text(0.01, 0.98, get_details(0), ha='left', va='top')

    def update(frame_idx: int):
        # Rotate the view.
        ax.view_init(azim=azims[frame_idx])

        # Update the worm
        fa.update(FS[frame_idx])

        # Update the text
        text.set_text(get_details(frame_idx))

        # Redraw the canvas
        fig.canvas.draw()

    fig.tight_layout()

    return fig, update


def _generate_annotated_images(image_triplet: np.ndarray, points_2d: np.ndarray, colours: np.ndarray) -> np.ndarray:
    """
    Prepare images with overlaid midlines as connecting lines between vertices.
    """
    images = []
    for i, img_array in enumerate(image_triplet):
        z = ((1 - img_array) * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGR)

        # Overlay 2d projection
        p2d = points_2d[:, i]
        for j, p in enumerate(p2d):
            z = cv2.drawMarker(
                z,
                p,
                color=colours[j].tolist(),
                markerType=cv2.MARKER_CROSS,
                markerSize=3,
                thickness=1,
                line_type=cv2.LINE_AA
            )
            if j > 0:
                cv2.line(
                    z,
                    p2d[j - 1],
                    p2d[j],
                    color=colours[j].tolist(),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

        images.append(z)

    return np.concatenate(images).transpose(1, 0, 2)


def generate_reconstruction_video(reconstruction_id: int, missing_only: bool = True):
    """
    Generate a reconstruction video showing a rotating 3D trajectory with reconstructed
    worm moving along it and camera images with overlaid 2D midline reprojections.
    """
    reconstruction = Reconstruction.objects.get(id=reconstruction_id)
    trial = reconstruction.trial

    # Video dimensions
    width = trial.crop_size * 3
    height = trial.crop_size * 3

    # Check if trajectory video is already present
    if missing_only and reconstruction.has_video:
        logger.info(f'Video already exists at: {reconstruction.video_filename}. Skipping.')
        return

    if reconstruction.source == M3D_SOURCE_MF:
        D = reconstruction.mf_parameters.depth  # todo - different depth videos
        D_min = reconstruction.mf_parameters.depth_min
        ts = TrialState(reconstruction=reconstruction)
        from_idx = 2**(D - 1) - 2**D_min
        to_idx = from_idx + 2**(D - 1)

        # Get 3D postures
        all_points = ts.get('points')
        X_full = all_points[:, from_idx:to_idx]

        # Add tracking points to get trajectory
        X_base = ts.get('points_3d_base')
        X_full = X_full + X_base[:, None, :]

        # Get 2D projections
        all_projections = ts.get('points_2d')  # (M, N, 3, 2)
        points_2d = np.round(all_projections[:, from_idx:to_idx]).astype(np.int32)

        # Get lengths
        lengths = ts.get('length')[:, 0]

        # Colour map
        colours = np.array([cmap(d) for d in np.linspace(0, 1, 2**(D - 1))])
        colours = np.round(colours * 255).astype(np.uint8)

    else:
        X_full, _ = get_trajectory(reconstruction_id=reconstruction.id)
        lengths = np.linalg.norm(X_full[:, 1:] - X_full[:, :-1], dim=-1).sum(dim=-1)

    # Get trajectory from centre of mass of reconstruction
    X_slice = X_full.mean(axis=1)

    # Build plot
    fig, update_plot = _make_3d_plot(
        trial=trial,
        width=width,
        height=height,
        X_slice=X_slice,
        X_full=X_full,
        lengths=lengths,
        draw_edges=True
    )

    # Fetch the images
    logger.info('Querying database.')
    pipeline = [
        {'$match': {
            'trial': trial.id,
            'frame_num': {
                '$gte': reconstruction.start_frame,
                '$lte': reconstruction.end_frame,
            }
        }},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
        }},
    ]
    cursor = Frame.objects().aggregate(pipeline, allowDiskUse=True)

    # Delete existing video
    if reconstruction.has_video:
        logger.info(f'Video already exists at: {reconstruction.video_filename}. Deleting.')
        os.remove(reconstruction.video_filename)

    # Initialise ffmpeg process
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
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(str(reconstruction.video_filename), **output_args)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    frame_nums = []
    logger.info('Rendering frames.')
    for i, res in enumerate(cursor):
        n = res['frame_num']
        if i > 0 and i % 100 == 0:
            logger.info(f'Rendering frame {i}/{reconstruction.n_frames}.')

        # Check we don't miss any frames
        if i == 0:
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
            images = _generate_annotated_images(image_triplet, points_2d[n], colours)

        else:
            m3d = Midline3D.objects.get(
                frame=res['_id'],
                source=reconstruction.source,
                source_file=reconstruction.source_file,
            )

            # Get 2D projections
            points_2d = np.round(m3d.get_prepared_2d_coordinates(regenerate=True)).astype(np.int32)
            points_2d = points_2d.transpose(1, 0, 2)

            # Colour map
            colours = np.array([cmap(i) for i in np.linspace(0, 1, points_2d.shape[0])])
            colours = np.round(colours * 255).astype(np.uint8)

            # Prepare images
            images = _generate_annotated_images(image_triplet, points_2d, colours)

        # Update the 3D plot and extract rendered image
        update_plot(i)
        plot = np.asarray(fig.canvas.renderer._renderer).take([0, 1, 2], axis=2)

        # Join plot to images and write to stream
        frame = np.concatenate([images, plot], axis=0)
        process.stdin.write(frame.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    logger.info(f'Generated video for frames {frame_nums[0]}-{frame_nums[-1]} ({frame_nums[-1] - frame_nums[0]}). '
                f'Total frames in reconstruction = {reconstruction.n_frames}.')


def generate_reconstruction_videos(missing_only: bool = True):
    reconstruction_ids = get_reconstruction_ids()

    for reconstruction_id in reconstruction_ids:
        logger.info(f'------ Generating reconstruction video for reconstruction id={reconstruction_id}.')

        # Soft fail on errors
        try:
            generate_reconstruction_video(reconstruction_id, missing_only)
        except Exception as e:
            raise
            logger.error(str(e))


if __name__ == '__main__':
    generate_reconstruction_videos(missing_only=False)
