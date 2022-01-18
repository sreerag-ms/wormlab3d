import os

import numpy as np
from wormlab3d import logger, PREPARED_IMAGES_PATH, PREPARED_IMAGE_SIZE
from wormlab3d.data.model import Trial, Frame
from wormlab3d.toolkit.util import resolve_targets

dry_run = False
shape = (3, *PREPARED_IMAGE_SIZE)
db = Trial._get_db()


def move_images_to_disk_trial(trial_id: int):
    logger.info(f'------ Moving images to disk for trial id={trial_id}.')
    trial = Trial.objects.get(id=trial_id)

    # Set up destination directory
    dest = PREPARED_IMAGES_PATH / f'{trial.id:03d}'
    os.makedirs(dest, exist_ok=True)

    # Fetch the frame ids
    logger.info('Fetching the frame ids.')
    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$project': {'_id': 1, 'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)
    frame_ids = []
    for res in cursor:
        frame_ids.append(res['_id'])
    logger.info(f'Fetched {len(frame_ids)} frame ids.')

    # Loop over frames
    missing = []
    for i, frame_id in enumerate(frame_ids):
        if (i + 1) % 10 == 0:
            logger.info(f'Processing frame {i + 1}/{len(frame_ids)}')
        frame = Frame.objects.get(id=frame_id)
        n = frame.frame_num
        images = np.array(frame.images)
        if images.shape != shape:
            missing.append(n)
            continue
        np.savez_compressed(dest / f'{n:06d}.npz', images=images)

        # Remove images from database!
        if not dry_run:
            frame.images = None
            frame.save()

    if len(missing) > 0:
        logger.warning(f'Found {len(missing)} frames without prepared images.')
        logger.debug(f'Frame nums={missing}.')

    # Free up the disk space
    logger.info('Compacting frames table to reclaim disk space.')
    res = db.command('compact', 'frame')
    logger.info(f'Result={res}')


def move_images_to_disk():
    trials, _ = resolve_targets()
    trial_ids = [trial.id for trial in trials]
    for trial_id in trial_ids:
        move_images_to_disk_trial(trial_id)


if __name__ == '__main__':
    move_images_to_disk()
