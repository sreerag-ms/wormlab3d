import matplotlib.pyplot as plt
import numpy as np
from wormlab3d import logger, CAMERA_IDXS
from wormlab3d.data.model import SegmentationMasks
from wormlab3d.data.model.trial import Trial
from wormlab3d.toolkit.plot_utils import interactive_plots


def plot_segmentation_masks(
        masks_id: str = None,
        trial_id: int = None,
        frame_id: int = None,
        frame_num: int = 0,
):
    """
    Plot a triplet of generated segmentation masks.
    """
    interactive_plots()
    masks: SegmentationMasks

    # Fetch masks by id
    if masks_id is not None:
        masks = SegmentationMasks.objects.get(id=masks_id)
    else:
        if frame_id is not None:
            filters = {'frame': frame_id}
        elif trial_id is not None:
            trial = Trial.objects.get(id=trial_id)
            frame = trial.get_frame(frame_num=frame_num)
            filters = {'frame': frame.id}
        else:
            raise RuntimeError('Either a mask, frame or trial id must be specified.')

        masks = SegmentationMasks.objects(**filters)
        if masks.count() > 1:
            logger.info(f'Found {len(masks)} in database, picking at random.')
            masks = masks[np.random.randint(masks.count())]
        elif masks.count() > 0:
            logger.info(f'Found {len(masks)} in database.')
            masks = masks[0]
        else:
            raise RuntimeError('No masks found in database!')
        logger.info(f'Loaded mask id = {masks.id}.')

    trial = masks.trial
    frame = masks.frame

    images = masks.get_images()
    segs = masks.X

    fig, axes = plt.subplots(3, 3)
    fig.suptitle(
        f'{trial.date:%Y%m%d} #{trial.trial_num}. \n'
        f'Frame: {frame.frame_num}'
    )

    for c in CAMERA_IDXS:
        ax = axes[0, c]
        ax.set_title(f'Camera {c}')
        ax.imshow(images[c], cmap='gray', vmin=0, vmax=1)

        ax = axes[1, c]
        ax.set_title('Combined')
        ax.imshow(images[c], vmin=0, vmax=1, cmap='gray', aspect='equal')
        alphas = segs[c].copy()
        alphas[alphas < 0.1] = 0
        alphas[alphas > 0.2] = 1
        ax.imshow(segs[c], vmin=0, vmax=1, cmap='Reds', aspect='equal', alpha=alphas)

        ax = axes[2, c]
        ax.set_title('Mask')
        ax.imshow(segs[c], vmin=0, vmax=1, cmap='Blues', aspect='equal')

    plt.show()


if __name__ == '__main__':
    plot_segmentation_masks(
        # masks_id='607ff754f782c04c8abd026d',
        trial_id=3,
        # frame_num=3343,
    )
