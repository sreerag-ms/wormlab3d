from scripts.preprocessing.util import process_args
from wormlab3d import logger

cached_readers = {}


def generate_centres_2d(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None,
        missing_only: bool = True
):
    """
    Find the centre-points of any objects in every frame of each camera's video.
    Ideally this should just return a single coordinate corresponding to the worm's location,
    but often it finds multiple in which case we store all and resolve later with generate_centres_3d.
    Note - the background images must be available.
    """
    trials, cam_idxs = process_args(experiment_id, trial_id, camera_idx, frame_num)
    if missing_only:
        logger.info(f'Generating any missing 2D centre points for {len(trials)} trials.')
    else:
        logger.info(f'(Re)generating ALL 2D centre points for {len(trials)} trials.')

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        if len(trial.backgrounds) != 3:
            logger.debug(f'Background images unavailable, skipping.')
            continue

        if trial_id in cached_readers:
            logger.debug('Using cached video readers.')
            readers = cached_readers[trial_id]
        else:
            readers = {}
            for c in cam_idxs:
                readers[c] = trial.get_video_reader(camera_idx=c)
            cached_readers[trial_id] = readers

        # Iterate over the frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            if missing_only:
                filters = {'__raw__': {
                    '$or': [
                        {'centres_2d': {'$size': 0}},
                        {'centres_2d': {'$elemMatch': {'$size': 0}}},
                    ]
                }}
            else:
                filters = None
            frames = trial.get_frames(filters)

        for frame in frames:
            log_prefix = f'Frame #{frame.frame_num}/{trial.num_frames} (id={frame.id}). '

            # Due to the large number of frames in each trial, we need to reload from the
            # database to catch changes since we fetched the results.
            frame.reload()
            if missing_only and frame.centres_2d_available():
                logger.info(log_prefix + 'Has 2D points, skipping.')
                continue

            # Lock the frame, or if we can't then skip it.
            if not frame.get_lock():
                logger.info(log_prefix + 'LOCKED, skipping.')
                continue

            if len(frame.centres_2d) == 0:
                frame.centres_2d = [[]] * 3
            for c in cam_idxs:
                try:
                    readers[c].set_frame_num(frame.frame_num)
                    centres = readers[c].find_objects()
                    logger.info(log_prefix + f'Cam={c}. Found {len(centres)} objects.')
                    frame.centres_2d[c] = centres
                except Exception as e:
                    logger.error(log_prefix + f'Cam={c}. Failed to find objects: {e}')

            # Unlock the frame when finished
            frame.release_lock_and_save()


def generate_centres_3d(
        experiment_id: int = None,
        trial_id: int = None,
        frame_num: int = None,
        missing_only: bool = True
):
    """
    Find a unique 3d centre-point for the worm.
    This uses the 2d coordinates found in each of the 3 camera views to resolve any uncertainties.
    Note - background images and 2d centre points must be available for this to work.
    """
    trials, cam_idxs = process_args(experiment_id, trial_id, None, frame_num)
    if missing_only:
        logger.info(f'Generating any missing 3D centre points for {len(trials)} trials.')
    else:
        logger.info(f'(Re)generating ALL 3D centre points for {len(trials)} trials.')

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        prev_point = None

        # Iterate over the frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            if missing_only:
                filters = {'centre_3d__exists': False}
            else:
                filters = None
            frames = trial.get_frames(filters)

        for frame in frames:
            log_prefix = f'Frame #{frame.frame_num}/{trial.num_frames} (id={frame.id}). '

            # Due to the large number of frames in each trial, we need to reload from the
            # database to catch changes since we fetched the results.
            frame.reload()
            if missing_only and frame.centre_3d is not None:
                logger.info(log_prefix + 'Has 3D point, skipping.')
                continue

            # 3D triangulation requires 2D points in all 3 views
            # This check is done in frame.generate_centre_3d also, but we use the above method
            # to take advantage of locks and cached video readers - better for bulk processing.
            if not frame.centres_2d_available():
                logger.warning(log_prefix + '2D centre points not available, generating now.')
                generate_centres_2d(trial_id=trial.id, frame_num=frame.frame_num)
                frame = frame.reload()
                assert frame.centres_2d_available()

            # Lock the frame, or if we can't then skip it.
            if not frame.get_lock():
                logger.info(log_prefix + 'LOCKED, skipping.')
                continue

            logger.info(log_prefix + 'Triangulating...')
            try:
                frame.generate_centre_3d(x0=prev_point)
                logger.info(log_prefix + f'Found 3D centre point.')
            except Exception as e:
                logger.error(log_prefix + f'Failed to find 3D centre: {e}')

            # Unlock the frame when finished
            frame.release_lock_and_save()
            if frame.centre_3d is not None:
                prev_point = frame.centre_3d.point_3d


if __name__ == '__main__':
    # # Poor error, spot obscured in one view
    # trial_id=186
    # frame_num=823

    # Lots of 2d points
    trial_id = 301
    frame_num = 79

    # 0-length video
    # trial_id=258
    # frame_num=559

    generate_centres_2d(
        # trial_id=trial_id,
        # frame_num=frame_num
    )
    generate_centres_3d(
        # trial_id=trial_id,
        # frame_num=frame_num
    )
