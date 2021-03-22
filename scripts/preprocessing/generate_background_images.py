import os

import cv2

from scripts.preprocessing.util import process_args
from wormlab3d import logger
from wormlab3d.data.util import fix_path, DATA_PATH_PLACEHOLDER


def ensure_output_dir_exists():
    output_dir = f'{DATA_PATH_PLACEHOLDER}/backgrounds'
    actual_dir = fix_path(output_dir)
    os.makedirs(actual_dir, exist_ok=True)


def generate_background_images(
        experiment_id=None,
        trial_id=None,
        camera_idx=None,
):
    """
    Generates a fixed background image from the input video using a low pass filter.
    """
    trials, cam_idxs = process_args(experiment_id, trial_id, camera_idx)
    ensure_output_dir_exists()

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        for ci in cam_idxs:
            logger.info(f'Processing camera idx={ci}')

            # Load the video and generate the background
            video = trial.get_video_reader(camera_idx=ci)
            bg = video.get_background()

            # Use placeholder in path
            output_path = f'{DATA_PATH_PLACEHOLDER}/backgrounds/{trial.id}_{ci}.png'
            actual_path = fix_path(output_path)

            # Save image to disk
            saved = cv2.imwrite(actual_path, bg, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if saved:
                logger.info(f'Saving to {output_path} ({actual_path})')
                setattr(trial, f'camera_{ci}_background', output_path)
            else:
                logger.error(f'Error saving to {actual_path}')

        # Update database
        trial.save()


if __name__ == '__main__':
    generate_background_images(
        trial_id='605473024e0295b2796d03d1',
        camera_idx=1
    )
