import ffmpeg
import numpy as np

from wormlab3d import logger, PREPARED_IMAGE_SIZE, TRACKING_VIDEOS_PATH
from wormlab3d.data.model import Trial, Frame
from wormlab3d.toolkit.util import resolve_targets

width = PREPARED_IMAGE_SIZE[0] * 3
height = PREPARED_IMAGE_SIZE[1]


def generate_tracking_video_trial(trial_id: int):
    trial = Trial.objects.get(id=trial_id)
    frame_nums = []

    # Fetch the images
    logger.info('Querying database.')
    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
            'images': 1
        }},
        # {'$sort': {'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline, allowDiskUse=True)

    # Initialise ffmpeg process
    video_filename = str(TRACKING_VIDEOS_PATH / f'{trial.id:03d}.mp4')
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='gray', s='{}x{}'.format(width, height))
            .output(video_filename, pix_fmt='gray', vcodec='libx264', r=trial.fps)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )

    logger.info('Fetching images.')
    for i, res in enumerate(cursor):
        n = res['frame_num']
        if i % 100 == 0:
            logger.debug(f'Fetching images for frame {n}/{trial.n_frames_min}.')

        # Check we don't miss any frames
        if i == 0:
            n0 = n
        assert n == n0 + i

        # Check images are present
        if 'images' not in res or len(res['images']) != 3:
            logger.warning('Prepared images not available, stopping here.')
            break
        frame_nums.append(n)

        # Stack image triplet and convert
        image_triplet = Frame.images.to_python(res['images'])
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

    logger.info(f'Generating video for frames {frame_nums[0]}-{frame_nums[-1]}. '
                f'(Total frames in database = {trial.n_frames_min}).')


def generate_tracking_videos(missing_only: bool = True):
    trials, _ = resolve_targets()
    trial_ids = [trial.id for trial in trials]
    for trial_id in trial_ids:
        logger.info(f'------ Generating tracking video for trial id={trial_id}.')

        # Check if tracking video is already present
        if missing_only:
            video_filename = TRACKING_VIDEOS_PATH / f'{trial_id:03d}.mp4'
            if video_filename.exists():
                logger.info(f'Video already exists at: {video_filename}. Skipping.')
                continue
        generate_tracking_video_trial(trial_id)
        break


if __name__ == '__main__':
    generate_tracking_videos(missing_only=True)
