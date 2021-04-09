import os
from collections import OrderedDict
from subprocess import CalledProcessError

import cv2

from scripts.preprocessing.util import process_args
from wormlab3d import logger
from wormlab3d.data.annex import is_annexed_file, fetch_from_annex
from wormlab3d.data.util import fix_path, DATA_PATH_PLACEHOLDER


def ensure_output_dir_exists():
    output_dir = f'{DATA_PATH_PLACEHOLDER}/backgrounds'
    actual_dir = fix_path(output_dir)
    os.makedirs(actual_dir, exist_ok=True)


def generate_background_images(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        missing_only: bool = True
):
    """
    Generates a fixed background image from the input video using a low pass filter.
    """
    trials, cam_idxs = process_args(experiment_id, trial_id, camera_idx)
    ensure_output_dir_exists()

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}. {len(trial.backgrounds)} backgrounds in database.')
        backgrounds = OrderedDict()
        for ci in cam_idxs:
            logger.info(f'Processing camera idx={ci}.')

            # Check for any existing background
            if missing_only and len(trial.backgrounds) > 0:
                bg_existing = None
                bg_path = fix_path(trial.backgrounds[ci])
                if bg_path is not None:
                    if is_annexed_file(bg_path):
                        try:
                            fetch_from_annex(bg_path)
                        except CalledProcessError as e:
                            logger.warning(f'Could not fetch existing background image from annex: {e}')
                    bg_existing = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
                if bg_existing is not None:
                    logger.info('Background image available, skipping.')
                    continue

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
                backgrounds[ci] = output_path
            else:
                logger.error(f'Error saving to {actual_path}')
                break

        # Save backgrounds, replacing or updating as needed
        if len(backgrounds) > 0:
            if len(backgrounds) == 3:
                trial.backgrounds = list(backgrounds.values())
            else:
                if len(trial.backgrounds) > 0:
                    bgs = trial.backgrounds
                else:
                    bgs = ['', '', '']
                for ci, bg in backgrounds.items():
                    bgs[ci] = bg
                trial.backgrounds = bgs

        else:
            logger.error('No backgrounds generated for trial!')

        # Update database
        trial.save()


if __name__ == '__main__':
    generate_background_images()
