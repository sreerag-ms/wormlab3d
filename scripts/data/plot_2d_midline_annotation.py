import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import logger
from wormlab3d.data.model import Trial
from wormlab3d.data.model.cameras import CAMERA_IDXS
from wormlab3d.data.model.frame import PREPARED_IMAGE_SIZE
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.toolkit.plot_utils import interactive_plots


def get_midline(trial_id: int, frame_num: int = None, camera_idx: int = None) -> Midline2D:
    """
    Find a midline for a given trial, frame and camera.
    """
    trial = Trial.objects.get(id=trial_id)
    midlines = Midline2D.objects(frame__in=trial.get_frames())
    logger.info(f'Found {len(midlines)} 2d midlines for trial_id = {trial_id}')
    logger.debug('Available for (frame_num, cam): ' + ','.join([f'{m.frame.frame_num, m.camera}' for m in midlines]))

    # Find a midline which matches the frame number and camera
    cam_idxs = CAMERA_IDXS if camera_idx is None else [camera_idx]
    midline = None
    if frame_num is not None:
        for m in midlines:
            if m.frame.frame_num == frame_num and m.camera in cam_idxs:
                midline = m
                break
        if midline is None:
            raise ValueError(
                f'No midlines found for frame_num={frame_num}'
                + (f' and camera_idx={camera_idx}' if camera_idx is not None else '')
            )
    else:
        # Or if nothing specified then pick one at random
        midline = midlines[np.random.randint(len(midlines))]

    return midline


def plot_2d_midline_annotation(midline2d_id: str = None):
    """
    Plot a 2d midline annotation
    """
    interactive_plots()
    midline = Midline2D.objects.get(id=midline2d_id)
    trial = midline.frame.trial
    image_original = midline.get_image()
    image_prepped = midline.get_prepared_image()
    n_plots = 1 if image_prepped is None else 2

    fig, axes = plt.subplots(n_plots)
    fig.suptitle(
        f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
        f'Video: {trial.videos[midline.camera]}. \n'
        f'Frame: {midline.frame.frame_num}'
    )

    if image_prepped is None:
        ax = axes
    else:
        ax = axes[0]
    ax.imshow(image_original, cmap='gray', vmin=0, vmax=255)
    ax.scatter(x=midline.X[:, 0], y=midline.X[:, 1], color='red', s=10, alpha=0.8)

    if image_prepped is not None:
        ax = axes[1]
        ax.set_title('Prepared image')
        ax.imshow(image_prepped, cmap='gray', vmin=0, vmax=1)

        centre_2d = midline.frame.centre_3d.reprojected_points_2d[midline.camera]

        X = midline.X.copy()
        X[:, 0] = X[:, 0] - centre_2d[0] + PREPARED_IMAGE_SIZE[0] / 2
        X[:, 1] = X[:, 1] - centre_2d[1] + PREPARED_IMAGE_SIZE[1] / 2

        ax.scatter(x=X[:, 0], y=X[:, 1], color='red', s=10, alpha=0.8)

    plt.show()


if __name__ == '__main__':
    mid = get_midline(trial_id=4, frame_num=5820, camera_idx=1)
    plot_2d_midline_annotation(midline2d_id=mid.id)
