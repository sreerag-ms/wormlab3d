import os

from pims import PyAVReaderTimed

os.environ['LOG_LEVEL'] = 'INFO'

from wormlab3d import logger
from wormlab3d.data.model import Trial


def fix_fps():
    trials = Trial.objects.only('id')
    trial_ids = [trial.id for trial in trials]

    for trial_id in trial_ids:
        trial = Trial.objects.get(id=trial_id)
        vtr = trial.get_video_triplet_reader()

        fps = []
        for r in vtr.readers:
            video: PyAVReaderTimed = r.video
            try:
                fps.append(video.frame_rate)
            except:
                logger.warning('Error getting frame rate.')
                break

        logger.info(f'Trial {trial_id}: {fps}')

        if len(fps) != 3 or any([r == 0 for r in fps]):
            logger.warning('Could not get FPS values for all videos! Setting FPS=0.')
            trial.fps = 0
            trial.save()
            continue

        # Check the rates are close enough
        mean = sum(fps) / 3
        rdtm = [abs(r - mean) / mean for r in fps]
        if any([r > 0.05 for r in rdtm]):
            logger.warning('Rates are too different! Setting FPS=0.')
            trial.fps = 0
            trial.save()
            continue

        # Update the average if different
        if mean != trial.fps:
            old_fps = f'{trial.fps:.2f}' if trial.fps is not None else '???'
            logger.info(f'Updating FPS from {old_fps} -> {mean:.2f}.')
            trial.fps = mean
            trial.save()
        else:
            logger.info('Database value matches.')


if __name__ == '__main__':
    fix_fps()
