import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from mayavi import mlab
from tvtk.tools import visual

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Frame
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.mf_render_wrapper import RenderWrapper
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.postures.plot_utils import FrameArtistMLab
from wormlab3d.toolkit.util import to_numpy
from wormlab3d.trajectories.util import fetch_reconstruction

show_plots = False
save_plots = True
# show_plots = True
# save_plots = False
img_extension = 'svg'


def _get_targets() -> Tuple[Reconstruction, Reconstruction, Frame]:
    """
    Resolve the reconstructions and frame.
    """
    parser = ArgumentParser(description='Wormlab3D script to plot the effect of centre-shifting.')
    parser.add_argument('--rec1', type=str, help='First reconstruction by id.')
    parser.add_argument('--rec2', type=str, help='Second reconstruction by id.')
    parser.add_argument('--frame-num', type=int, help='Frame number.')
    args = parser.parse_args()

    # Fetch reconstructions
    assert args.rec1 is not None, 'First reconstruction must be specified!'
    assert args.rec2 is not None, 'Second reconstruction must be specified!'
    rec1 = fetch_reconstruction(reconstruction_id=args.rec1)
    rec2 = fetch_reconstruction(reconstruction_id=args.rec2)
    assert rec1.source == M3D_SOURCE_MF, 'Only MF reconstructions work for this!'
    assert rec2.source == M3D_SOURCE_MF, 'Only MF reconstructions work for this!'
    assert rec1.trial.id == rec2.trial.id, 'Reconstructions should be for the same trial!'

    # Fetch frame
    if args.frame_num is None:
        frame_num = np.random.randint(rec1.start_frame, rec1.end_frame)
        logger.info(f'Selected frame num = {frame_num} at random.')
    else:
        frame_num = args.frame_num
    frame = rec1.trial.get_frame(frame_num)

    return rec1, rec2, frame


def _find_score_midpoint(scores: np.ndarray) -> int:
    """
    Find the midpoint of the scores.
    """
    N = scores.shape[0]
    scores_aa = (scores > (scores.max() - scores.min()) / 2).astype(np.float64)
    centroid = (np.arange(N) * scores_aa).sum() / scores_aa.sum()
    centroid_idx = np.ceil(centroid).astype(np.int32)
    return centroid_idx


def plot_scores(
        rec1: Reconstruction,
        rec2: Reconstruction,
        frame: Frame,
        save_dir: Path
):
    """
    Plot the scores.
    """
    N = rec1.mf_parameters.n_points_total
    rw1 = RenderWrapper(rec1, frame)
    rw2 = RenderWrapper(rec2, frame)
    scores1 = rw1.get_scores()
    scores2 = rw2.get_scores()

    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    fc = cmap((np.arange(N) + 0.5) / N)
    ind = np.arange(N)

    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=5)  # fontsize of the y tick labels
    plt.rc('xtick.major', pad=2)
    plt.rc('ytick.major', pad=1, size=2)

    for i in range(2):
        fig, ax = plt.subplots(1, figsize=(1.15, 0.55), gridspec_kw={
            'left': 0.19,
            'right': 0.95,
            'top': 0.98,
            'bottom': 0.28,
        })
        ax.spines['top'].set_visible(False)

        if i == 0:
            v = scores1
        else:
            v = scores2

        for n in range(N - 1):
            ax.plot(ind[n:n + 2], v[n:n + 2], c=fc[n])

        mp = _find_score_midpoint(v)
        ax.vlines(x=mp, ymin=0, ymax=v[mp], linestyle=':', color='grey')

        # Set up x-axis
        ax.set_xticks([])
        ax.set_xlim(left=0, right=N - 1)
        ax.set_xticks([0, N - 1])
        ax.set_xticklabels(['H', 'T'])

        # Set up y-axis
        ax.set_ylim(bottom=0, top=max(scores1.max(), scores2.max()) + 100)
        ax.set_yticks([0, 2000])

        if save_plots:
            path = save_dir / f'rec{i}_scores.{img_extension}'
            logger.info(f'Saving scores plot to {path}.')
            plt.savefig(path, transparent=True)
        if show_plots:
            plt.show()


def plot_3d(
        rec1: Reconstruction,
        rec2: Reconstruction,
        frame: Frame,
        save_dir: Path
):
    """
    Plot the 3d postures.
    """
    frame_num = frame.frame_num
    N = rec1.mf_parameters.n_points_total
    rw1 = RenderWrapper(rec1, frame)
    rw2 = RenderWrapper(rec2, frame)
    NF1 = NaturalFrame(rw1.ts.get('points', frame_num, frame_num + 1)[0])
    NF2 = NaturalFrame(rw2.ts.get('points', frame_num, frame_num + 1)[0])
    scores1 = rw1.get_scores()
    scores2 = rw2.get_scores()
    c1 = _find_score_midpoint(scores1)
    c2 = _find_score_midpoint(scores2)
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    mlab.options.offscreen = not show_plots

    # Set up the artists
    fa_args = dict(
        midline_opts={
            'opacity': 0.7,
            'tube_radius': 0.003,
        },
        mesh_opts={'opacity': 0.7},
        use_centred_midline=False
    )
    fa1 = FrameArtistMLab(NF1, **fa_args)
    fa2 = FrameArtistMLab(NF2, **fa_args)
    X_combined = np.concatenate([fa1.X, fa2.X])
    centre = X_combined.min(axis=0) + X_combined.ptp(axis=0) / 2

    for i in range(2):
        fig = mlab.figure(size=(2000, 1100), bgcolor=(1, 1, 1))
        if 1:
            # Doesn't really seem to make any difference
            fig.scene.render_window.point_smoothing = True
            fig.scene.render_window.line_smoothing = True
            fig.scene.render_window.polygon_smoothing = True
            fig.scene.render_window.multi_samples = 20
            fig.scene.anti_aliasing_frames = 20
        visual.set_viewer(fig)

        # Add the midlines and surfaces
        if i == 0:
            fa1.mesh_opts['opacity'] = 0.4
            fa2.mesh_opts['opacity'] = 0.2

            fa1.add_midline(fig)
            fa1.add_surface(fig)
            fa2.add_surface(fig)

            x = fa1.X[c1]
            x_col = cmap(c1 / N)[:3]

        else:
            fa1.mesh_opts['opacity'] = 0.2
            fa2.mesh_opts['opacity'] = 0.4

            fa2.add_midline(fig)
            fa2.add_surface(fig)
            fa1.add_surface(fig)

            x = fa2.X[c2]
            x_col = cmap(c2 / N)[:3]

        # Add the midpoint
        mlab.points3d(*x, scale_factor=0.04, color=x_col, figure=fig)

        # Render and save
        mlab.view(
            azimuth=110,
            elevation=80,
            roll=125,
            distance=1.,
            focalpoint=centre
        )

        # # Useful for getting the view parameters when recording from the gui:
        # scene = mlab.get_engine().scenes[0]
        # scene.scene.camera.position = [-0.38944838595029707, 0.8715967356589394, 0.15633139993768805]
        # scene.scene.camera.focal_point = [-0.03450489044189453, -0.05100369453430176, 0.005256712436676025]
        # scene.scene.camera.view_angle = 30.0
        # scene.scene.camera.view_up = [-0.1423623793905549, -0.21305155758042005, 0.9666136698530684]
        # scene.scene.camera.clipping_range = [0.23884424151380584, 1.9536354859666436]
        # scene.scene.camera.compute_view_plane_normal()
        # scene.scene.render()
        # print(mlab.view())  # (azimuth, elevation, distance, focalpoint)
        # print(mlab.roll())
        # exit()

        if save_plots:
            path = save_dir / f'rec{i}_3D.png'
            logger.info(f'Saving plot to {path}.')
            fig.scene._lift()
            img = mlab.screenshot(figure=fig, mode='rgba', antialiased=True)
            img = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
            img.save(path)

        if show_plots:
            mlab.show()

        mlab.clf(fig)


def plot_2d(
        rec1: Reconstruction,
        rec2: Reconstruction,
        frame: Frame,
        save_dir: Path,
        midpoint_radius: int = 3,
        midline_width: int = 1,
        crop_size: int = -1
):
    """
    Plot the 2d projections.
    """
    N = rec1.mf_parameters.n_points_total
    rw1 = RenderWrapper(rec1, frame)
    rw2 = RenderWrapper(rec2, frame)
    scores1 = rw1.get_scores()
    scores2 = rw2.get_scores()
    c1 = _find_score_midpoint(scores1)
    c2 = _find_score_midpoint(scores2)

    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = cmap(np.linspace(0, 1, N))
    colours = np.round(colours * 255).astype(np.uint8)

    # Get the masked input images
    dm = rw2.get_detection_masks()
    masked_images = (frame.images * dm).clip(min=0, max=1)

    if save_plots:
        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            midpoint_radius=midpoint_radius,
            midline_width=midline_width,
            crop_size=crop_size
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for i in range(2):
        for c, img_array in enumerate(masked_images):
            z = (img_array * 255).astype(np.uint8)
            z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGRA)

            # Overlay 2d projection
            if i == 0:
                p2d = rw1.points_2d[0, :, c]
            else:
                p2d = rw2.points_2d[0, :, c]
            p2d = np.round(to_numpy(p2d)).astype(np.int32)

            for j, p in enumerate(p2d):
                col = colours[j].tolist()

                # Draw markers and connecting lines
                z = cv2.drawMarker(
                    z,
                    p,
                    color=col,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=2,
                    thickness=1,
                    line_type=cv2.LINE_AA
                )
                if j > 0:
                    cv2.line(z, p2d[j - 1], p2d[j], color=col, thickness=midline_width, lineType=cv2.LINE_AA)

            # Draw midpoint
            idx = [c1, c2][i]
            z = cv2.circle(z, p2d[idx], midpoint_radius, color=colours[idx].tolist(), thickness=-1,
                           lineType=cv2.LINE_AA)

            # Convert to PIL image
            img = Image.fromarray(z, 'RGBA')

            # Crop
            if crop_size > -1 and crop_size < img.size[0]:
                m = int(img.size[0] - crop_size) / 2  # margin to remove
                img = img.crop(box=(m, m, img.size[0] - m, img.size[1] - m))

            if save_plots:
                save_path = save_dir / f'rec{i}_2D_c{c}.png'
                logger.info(f'Saving image to {save_path}.')
                img.save(save_path)

            if show_plots:
                img.show()


if __name__ == '__main__':
    rec1_, rec2_, frame_ = _get_targets()
    save_dir_ = LOGS_PATH / (f'{START_TIMESTAMP}'
                             f'_trial={rec1_.trial.id}'
                             f'_frame={frame_.frame_num}'
                             f'_rec1={rec1_.id}'
                             f'_rec2={rec2_.id}')
    if save_plots:
        os.makedirs(save_dir_, exist_ok=True)

    plot_scores(
        rec1=rec1_,
        rec2=rec2_,
        frame=frame_,
        save_dir=save_dir_,
    )

    plot_3d(
        rec1=rec1_,
        rec2=rec2_,
        frame=frame_,
        save_dir=save_dir_,
    )

    plot_2d(
        rec1=rec1_,
        rec2=rec2_,
        frame=frame_,
        save_dir=save_dir_,
        midpoint_radius=7,
        midline_width=2,
        crop_size=140
    )
