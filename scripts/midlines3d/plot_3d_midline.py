import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from simple_worm.frame import FrameNumpy
from simple_worm.plot3d import cla, FrameArtist
from wormlab3d import logger, CAMERA_IDXS, LOGS_PATH, START_TIMESTAMP
from wormlab3d.data.model import Trial, Midline3D
from wormlab3d.toolkit.util import parse_target_arguments

MAX_ATTEMPTS = 10

img_extension = 'svg'
show_plots = True
save_plots = True
invert = True

if save_plots:
    os.makedirs(LOGS_PATH, exist_ok=True)


def get_midline(attempt: int = 1) -> Midline3D:
    """
    Find a midline for a given trial and frame num, or by id.
    """
    args = parse_target_arguments()
    if args.midline3d:
        return Midline3D.objects.get(id=args.midline3d)

    if args.trial is None:
        raise RuntimeError('Trial id or midline3d id must be specified.')

    trial = Trial.objects.get(id=args.trial)
    if args.frame_num is None:
        frame_num = np.random.randint(trial.n_frames_min)
        logger.info(f'Selected frame num = {frame_num} at random.')
    else:
        frame_num = args.frame_num

    frame = trial.get_frame(frame_num)
    midlines = frame.get_midlines3d({'source': 'reconst'})
    logger.info(f'Found {len(midlines)} 3d midlines for trial_id = {trial.id}, frame_num = {frame_num}')
    if len(midlines) > 1:
        logger.info('Picking at random..')
        midline = midlines[np.random.randint(len(midlines))]
        logger.info(f'Source={midline.source}')
    elif len(midlines) == 1:
        midline = midlines[0]
    elif attempt < MAX_ATTEMPTS:
        logger.warning(f'No midlines found for frame_num={frame_num}, trying again...')
        return get_midline(attempt + 1)
    else:
        raise ValueError('No midlines found, try specifying a frame number of midline id.')

    return midline


def plot_3d(midline: Midline3D):
    """
    3D plot of a midline.
    """
    frame = midline.frame
    trial = frame.trial
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(
        f'Trial: {trial.id}. \n'
        f'Frame: {frame.frame_num}'
    )

    F = FrameNumpy(x=midline.X.T)
    fa = FrameArtist(F=F)
    fa.add_midline(ax)
    cla(ax)
    fig.tight_layout()

    if save_plots:
        save_path = LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_frame={frame.frame_num}_midline={midline.id}_3D.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_reprojections(midline: Midline3D):
    """
    Plot the 3D midline reprojected back down on top of the images.
    """
    frame = midline.frame
    if len(frame.images) != 3:
        frame.generate_prepared_images()
        frame.save()
    trial = frame.trial
    images = frame.images

    fig, axes = plt.subplots(3, figsize=(6, 8))
    fig.suptitle(
        f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
        f'Frame: {midline.frame.frame_num}'
    )

    masks = midline.get_segmentation_masks(blur_sigma=1)
    for c in CAMERA_IDXS:
        ax = axes[c]
        ax.set_title(f'Camera {c}')
        ax.imshow(images[c], cmap='gray', vmin=0, vmax=1)
        alphas = masks[c].copy()
        alphas[alphas < 0.1] = 0
        alphas[alphas > 0.2] = 1
        ax.imshow(masks[c], vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

    fig.tight_layout()

    if save_plots:
        save_path = LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_frame={frame.frame_num}_midline={midline.id}_2D.{img_extension}'
        logger.info(f'Saving plot to {save_path}.')
        plt.savefig(save_path)

    if show_plots:
        plt.show()


def plot_reprojection_singles(midline: Midline3D):
    """
    Show/save the 3D midline reprojected back down on top of the images.
    """
    frame = midline.frame
    if len(frame.images) != 3:
        frame.generate_prepared_images()
        frame.save()
    trial = frame.trial
    images = frame.images
    points_2d = np.array(midline.get_prepared_2d_coordinates())
    points_2d = np.round(points_2d).astype(np.int32)

    # Colour map
    cmap = plt.get_cmap('plasma')
    colours = np.array([cmap(i) for i in np.linspace(0, 1, points_2d.shape[1])])
    colours = np.round(colours * 255).astype(np.uint8)

    for c in CAMERA_IDXS:
        img = images[c]
        if invert:
            img = 1 - img
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Overlay 2d projection
        p2d = points_2d[c]
        for j, p in enumerate(p2d):
            img = cv2.drawMarker(
                img,
                p,
                color=colours[j].tolist(),
                markerType=cv2.MARKER_CROSS,
                markerSize=2,
                thickness=1,
                line_type=cv2.LINE_AA
            )
            if j > 0:
                cv2.line(img, p2d[j - 1], p2d[j], color=colours[j].tolist(), thickness=1, lineType=cv2.LINE_AA)

        # Convert to PIL image
        img = Image.fromarray(img, 'RGB')

        if save_plots:
            save_path = LOGS_PATH / f'{START_TIMESTAMP}_trial={trial.id}_frame={frame.frame_num}_midline={midline.id}_2D_c={c}.png'
            logger.info(f'Saving image to {save_path}.')
            img.save(save_path)

        if show_plots:
            img.show()


if __name__ == '__main__':
    # from wormlab3d.toolkit.plot_utils import interactive_plots
    # interactive_plots()
    mid = get_midline()
    # plot_3d(mid)
    # plot_reprojections(mid)
    plot_reprojection_singles(mid)
