from scripts.preprocessing.util import process_args
from wormlab3d import logger


def triangulate_2d(
        experiment_id=None,
        trial_id=None,
        camera_idx=None,
):
    """
    Find the centre-points of any objects in every frame of each camera's video.
    Ideally this should just return a single coordinate corresponding to the worm's location,
    but occasionally it might find multiple in which case we store all and resolve with triangulate_3d.
    Note - the background images must be available for this to work.
    """

    trials, cam_idxs = process_args(experiment_id, trial_id, camera_idx)

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        readers = {}
        for ci in cam_idxs:
            readers[ci] = trial.get_video_reader(camera_idx=ci)

        # Iterate over the frames
        frames = trial.get_frames()
        for frame in frames:
            for ci in cam_idxs:
                readers[ci].set_frame_num(frame.frame_num)
                centres = readers[ci].find_objects()
                logger.debug(f'Frame #{frame.frame_num}. Found {len(centres)} objects.')
                setattr(frame, f'centres_cam_{ci}', centres)
            frame.save()


def triangulate_3d(
        experiment_id=None,
        trial_id=None
):
    """
    Find a unique 3d centre-point for the worm.
    This uses the 2d coordinates found in each of the 3 camera views to resolve any uncertainties.
    Note - background images and 2d centre points must be available for this to work.
    """

    trials, cam_idxs = process_args(experiment_id, trial_id)

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')

        # Iterate over the frames
        frames = trial.get_frames()
        for frame in frames:
            c1 = frame.centres_cam_1
            c2 = frame.centres_cam_2
            c3 = frame.centres_cam_3
            # todo - triangulate
            # frame.save()


if __name__ == '__main__':
    triangulate_2d(
        trial_id='605473024e0295b2796d03d1',
        camera_idx=1
    )
