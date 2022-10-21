import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from simple_worm.plot3d import MIDLINE_CMAP_DEFAULT
from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Reconstruction, Frame
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.mf_render_wrapper import RenderWrapper
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
    parser = ArgumentParser(description='Wormlab3D script to plot the effect of masking.')
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


def plot_2d(
        rec1: Reconstruction,
        rec2: Reconstruction,
        frame: Frame,
        save_dir: Path,
        midline_width: int = 1,
        crop_size: int = -1
):
    """
    Plot the 2d projections.
    """
    N = rec1.mf_parameters.n_points_total
    rw1 = RenderWrapper(rec1, frame)
    rw2 = RenderWrapper(rec2, frame)

    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = cmap(np.linspace(0, 1, N))
    colours = np.round(colours * 255).astype(np.uint8)

    if save_plots:
        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            midline_width=midline_width,
            crop_size=crop_size
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    for i in range(2):
        for c, img_array in enumerate(frame.images):
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

            # Convert to PIL image
            img = Image.fromarray(z, 'RGBA')

            # Crop
            if -1 < crop_size < img.size[0]:
                m = int(img.size[0] - crop_size) / 2  # margin to remove
                img = img.crop(box=(m, m, img.size[0] - m, img.size[1] - m))

            if save_plots:
                save_path = save_dir / f'rec{i}_2D_c{c}.png'
                logger.info(f'Saving image to {save_path}.')
                img.save(save_path)

            if show_plots:
                img.show()


def plot_2d_progression(
        rec1: Reconstruction,
        rec2: Reconstruction,
        frame: Frame,
        save_dir: Path,
        midline_width: int = 1,
        crop_size: int = -1,
        window_size: int = 5,
        frame_skip: int = 5,
):
    """
    Plot 2d projections over a window.
    """
    trial = rec1.trial
    N = rec1.mf_parameters.n_points_total
    cmap = plt.get_cmap(MIDLINE_CMAP_DEFAULT)
    colours = cmap(np.linspace(0, 1, N))
    colours = np.round(colours * 255).astype(np.uint8)

    if save_plots:
        # Save layout spec
        spec = dict(
            created=START_TIMESTAMP,
            midline_width=midline_width,
            crop_size=crop_size,
            window_size=window_size
        )
        with open(save_dir / 'spec.yml', 'w') as f:
            yaml.dump(spec, f)

    start_frame = int(frame.frame_num - window_size * frame_skip / 2)
    end_frame = start_frame + window_size * frame_skip
    frame_nums = np.arange(start_frame, end_frame, frame_skip)
    logger.info(f'Frame progression: {frame_nums}')

    if crop_size == -1:
        crop_size = rec1.trial.crop_size

    for c in range(3):
        out = np.zeros((crop_size * 2, crop_size * window_size, 4), dtype=np.uint8)

        for i, frame_num in enumerate(frame_nums):
            frame_w = trial.get_frame(frame_num)
            rw2 = RenderWrapper(rec2, frame_w)
            rw1 = RenderWrapper(rec1, frame_w)
            img_array = frame_w.images[c]

            for j in range(2):
                z = (img_array * 255).astype(np.uint8)
                z = cv2.cvtColor(z, cv2.COLOR_GRAY2BGRA)

                # Overlay 2d projection
                if j == 0:
                    p2d = rw1.points_2d[0, :, c]
                else:
                    p2d = rw2.points_2d[0, :, c]
                p2d = np.round(to_numpy(p2d)).astype(np.int32)

                for k, p in enumerate(p2d):
                    col = colours[k].tolist()

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
                    if k > 0:
                        cv2.line(z, p2d[k - 1], p2d[k], color=col, thickness=midline_width, lineType=cv2.LINE_AA)

                # Crop
                img = Image.fromarray(z, 'RGBA')
                if -1 < crop_size < img.size[0]:
                    m = int(img.size[0] - crop_size) / 2  # margin to remove
                    img = img.crop(box=(m, m, img.size[0] - m, img.size[1] - m))

                out[crop_size * j:crop_size * (j + 1), crop_size * i:crop_size * (i + 1)] = img

        out_img = Image.fromarray(out, 'RGBA')
        out_img_r1 = Image.fromarray(out[:crop_size], 'RGBA')
        out_img_r2 = Image.fromarray(out[crop_size:], 'RGBA')
        if save_plots:
            save_path = save_dir / f'progression_frames[{",".join([str(n) for n in frame_nums])}]_c{c}.png'
            logger.info(f'Saving image to {save_path}.')
            out_img.save(save_path)
            out_img_r1.save(save_path.with_name(save_path.stem + '_r1.png'))
            out_img_r2.save(save_path.with_name(save_path.stem + '_r2.png'))

        if show_plots:
            out_img.show()


if __name__ == '__main__':
    rec1_, rec2_, frame_ = _get_targets()
    save_dir_ = LOGS_PATH / (f'{START_TIMESTAMP}'
                             f'_trial={rec1_.trial.id}'
                             f'_frame={frame_.frame_num}'
                             f'_rec1={rec1_.id}'
                             f'_rec2={rec2_.id}')
    if save_plots:
        os.makedirs(save_dir_, exist_ok=True)

    # plot_2d(
    #     rec1=rec1_,
    #     rec2=rec2_,
    #     frame=frame_,
    #     save_dir=save_dir_,
    #     midline_width=3,
    #     crop_size=-1
    # )

    plot_2d_progression(
        rec1=rec1_,
        rec2=rec2_,
        frame=frame_,
        save_dir=save_dir_,
        midline_width=3,
        crop_size=200,
        window_size=5,
        frame_skip=5
    )
