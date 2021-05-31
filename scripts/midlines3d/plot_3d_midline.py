import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import logger, CAMERA_IDXS
from wormlab3d.data.model import Trial, Midline3D
from wormlab3d.toolkit.plot_utils import interactive_plots
from wormlab3d.toolkit.util import parse_target_arguments

MAX_ATTEMPTS = 10


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
    midlines = frame.get_midlines3d()
    logger.info(f'Found {len(midlines)} 3d midlines for trial_id = {trial.id}, frame_num = {frame_num}')
    if len(midlines) > 1:
        logger.info('Picking at random..')
        midline = midlines[np.random.randint(len(midlines))]
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
    interactive_plots()
    frame = midline.frame
    trial = frame.trial
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_title(
        f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
        f'Frame: {frame.frame_num}'
    )
    x, y, z = (midline.X[:, j] for j in range(3))
    ax.scatter(x, y, z, s=50, marker='x', alpha=0.9)
    fig.tight_layout()
    plt.show()


def plot_reprojections(midline: Midline3D):
    """
    Plot the 3D midline reprojected back down on top of the images.
    """
    interactive_plots()

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
    plt.show()


if __name__ == '__main__':
    mid = get_midline()
    plot_3d(mid)
    plot_reprojections(mid)
