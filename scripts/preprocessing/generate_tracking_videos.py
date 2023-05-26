import os
import time

import ffmpeg
import numpy as np

from wormlab3d import logger, PREPARED_IMAGES_PATH
from wormlab3d.data.model import Trial, Frame
from wormlab3d.toolkit.util import resolve_targets


def generate_tracking_video_trial(trial: Trial, invert: bool = False):
    if trial.fps == 0:
        logger.warning('Trial FPS=0. Cannot create video.')
        return

    # Video dimensions
    width = trial.crop_size * 3
    height = trial.crop_size

    # Fetch the images
    logger.info('Querying database.')
    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
        }},
        # {'$sort': {'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline, allowDiskUse=True)

    # Delete existing video
    video_filename = trial.tracking_video_path
    if video_filename.exists():
        logger.info(f'Video already exists at: {video_filename}. Deleting.')
        os.remove(video_filename)

    # Initialise ffmpeg process
    output_args = {
        'pix_fmt': 'gray',
        'vcodec': 'libx264',
        'r': trial.fps,
        'metadata:g:0': f'title=Trial {trial.id}',
        'metadata:g:1': 'artist=Leeds Wormlab',
        'metadata:g:2': f'year={time.strftime("%Y")}',
    }
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(width, height))
        .output(str(video_filename), **output_args)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    frame_nums = []
    logger.info('Fetching images.')
    for i, res in enumerate(cursor):
        n = res['frame_num']
        if i % 100 == 0:
            logger.debug(f'Fetching images for frame {n}/{trial.n_frames_min}.')

        # Check we don't miss any frames
        if i == 0:
            n0 = n
        assert n == n0 + i

        # Check images are present and the right size
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

        # Invert colours if required
        if invert:
            image_triplet = 1 - image_triplet

        # Stack image triplet and convert
        image_triplet = np.floor(np.concatenate(image_triplet) * 255)
        image_triplet = image_triplet.astype(np.uint8).T
        process.stdin.write(image_triplet.tobytes())

    # Flush video
    process.stdin.close()
    process.wait()

    # Abort if less than 1 second of images available
    if len(frame_nums) < 25:
        logger.warning('Fewer than 25 contiguous frames found with images.')
        return

    logger.info(f'Generated video for frames {frame_nums[0]}-{frame_nums[-1]}. '
                f'(Total frames in database = {trial.n_frames_min}).')


def generate_tracking_videos(missing_only: bool = True, invert: bool = False):
    trials, _ = resolve_targets()
    trial_ids = [trial.id for trial in trials]
    for trial_id in trial_ids:
        logger.info(f'------ Generating tracking video for trial id={trial_id}.')
        trial = Trial.objects.get(id=trial_id)

        # Check if tracking video is already present and the right size
        if missing_only and trial.has_tracking_video:
            metadata = ffmpeg.probe(trial.tracking_video_path)
            width = metadata['streams'][0]['width']
            height = metadata['streams'][0]['height']
            if width == height * 3 and height == trial.crop_size:
                logger.info(f'Video already exists at: {trial.tracking_video_path}. Skipping.')
                continue

        # Soft fail on errors
        try:
            generate_tracking_video_trial(trial, invert)
        except Exception as e:
            logger.error(str(e))


if __name__ == '__main__':
    generate_tracking_videos(missing_only=False, invert=True)
