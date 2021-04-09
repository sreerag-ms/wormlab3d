from wormlab3d import logger
from wormlab3d.toolkit.util import resolve_targets

cached_readers = {}


def generate_centres_2d(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None
):
    """
    Find the centre-points of any objects in every frame of each camera's video.
    Ideally this should just return a single coordinate corresponding to the worm's location,
    but occasionally it might find multiple in which case we store all and resolve with triangulate_3d.
    Note - the background images must be available for this to work well.
    """
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, camera_idx, frame_num)
    logger.info(f'Generating 2d centre points for {len(trials)} trials.')

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')

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
            frames = trial.get_frames()
        for frame in frames:
            if len(frame.centres_2d) == 0:
                frame.centres_2d = [[]] * 3
            for c in cam_idxs:
                readers[c].set_frame_num(frame.frame_num)
                centres = readers[c].find_objects()
                logger.debug(f'Frame #{frame.frame_num}/{trial.num_frames}. Found {len(centres)} objects.')
                frame.centres_2d[c] = centres
            frame.save()


def generate_centres_3d(
        experiment_id: int = None,
        trial_id: int = None,
        frame_num: int = None
):
    """
    Find a unique 3d centre-point for the worm.
    This uses the 2d coordinates found in each of the 3 camera views to resolve any uncertainties.
    Note - background images and 2d centre points must be available for this to work.
    """
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, None, frame_num)
    logger.info(f'Generating 3d centre points for {len(trials)} trials.')

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        prev_point = None

        # Iterate over the frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            frames = trial.get_frames()
        for frame in frames:
            logger.info(f'Frame #{frame.frame_num}/{trial.num_frames} (id={frame.id}).')
            if not frame.centres_2d_available():
                logger.warning('Frame does not have 2d centre points available for all views, generating now.')
                generate_centres_2d(
                    trial_id=trial.id,
                    frame_num=frame.frame_num
                )
                frame = frame.reload()
                assert frame.centres_2d_available()

            frame.generate_centre_3d(x0=prev_point)
            frame.save()
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
        trial_id=trial_id,
        frame_num=frame_num
    )
    generate_centres_3d(
        trial_id=trial_id,
        frame_num=frame_num
    )
