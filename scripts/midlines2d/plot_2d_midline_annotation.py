import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import logger, CAMERA_IDXS
from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.data.model.trial import Trial
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


def plot_2d_midline_annotation(midline2d_id: str, draw_mode: str):
    """
    Plot a 2d midline annotation
    """
    interactive_plots()
    midline = Midline2D.objects.get(id=midline2d_id)
    trial = midline.frame.trial
    image_original = midline.get_image()
    image_prepped = midline.get_prepared_image()
    n_plots = 1 if image_prepped is None else 4

    fig, axes = plt.subplots(n_plots, figsize=(10, 10))
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
        X = midline.get_prepared_coordinates()
        ax.scatter(x=X[:, 0], y=X[:, 1], color='red', s=10, alpha=0.8)

        # Plot mask
        ax = axes[2]
        ax.set_title(f'Segmentation mask. Draw mode={draw_mode}.')
        mask = midline.get_segmentation_mask(draw_mode=draw_mode)
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)

        # Plot fattened mask
        blur_sigma = 5
        ax = axes[3]
        ax.set_title(f'Segmentation mask. Blur_sigma={blur_sigma}.')
        mask = midline.get_segmentation_mask(draw_mode=draw_mode, blur_sigma=blur_sigma)
        ax.imshow(mask, cmap='gray', vmin=0, vmax=1)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # mid = get_midline(trial_id=4, frame_num=0, camera_idx=1)

    # # Lots of 2d points
    # trial_id=301
    # frame_num=79

    # Broken
    # trial_id = 114
    # frame_num = 0

    # Sparsely defined midline
    trial_id = 232
    frame_num = 6983

    mid = get_midline(trial_id=trial_id, frame_num=frame_num, camera_idx=0)
    plot_2d_midline_annotation(midline2d_id=mid.id, draw_mode='line_aa')
