import gc

from wormlab3d import logger
from wormlab3d.data.model import Frame, Trial
from wormlab3d.preprocessing.contour import MIN_REQ_THRESHOLD, CONT_THRESH_RATIO_DEFAULT
from wormlab3d.toolkit.util import resolve_targets

cached_readers = {}


def generate_centres_2d(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None,
        missing_only: bool = True,
        contour_threshold_ratio: float = CONT_THRESH_RATIO_DEFAULT,
        min_brightness: int = None,
):
    """
    Find the centre-points of any objects in every frame of each camera's video.
    Ideally this should just return a single coordinate corresponding to the worm's location,
    but often it finds multiple in which case we store all and resolve later with generate_centres_3d.
    Note - the background images must be available.
    """
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, camera_idx, frame_num)
    if missing_only:
        logger.info(f'Generating any missing 2D centre points for {len(trials)} trials.')
    else:
        logger.info(f'(Re)generating ALL 2D centre points for {len(trials)} trials.')

    if min_brightness is None:
        min_brightness = MIN_REQ_THRESHOLD / contour_threshold_ratio
    logger.debug(f'Minimum brightness required = {min_brightness:.2f}.')

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')
        if len(trial.backgrounds) != 3:
            logger.warning(f'Background images unavailable, skipping.')
            continue

        # Filter frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            # Match each cam if it is in range, bright enough and (optionally) centres are not computed
            matches = []
            if len(trial.n_frames) != 3:
                trial.n_frames = [0] * 3

            for c in cam_idxs:
                if missing_only:
                    missing_cond = {
                        '$or': [
                            {'centres_2d': {'$size': 0}},
                            {'$and': [
                                {f'centres_2d.{c}': {'$size': 0}},
                                {f'centres_2d_thresholds.{c}': {'$gt': MIN_REQ_THRESHOLD}},
                            ]},
                            {'centres_2d_thresholds': None},
                            {'centres_2d_thresholds': {'$size': 0}},
                        ]
                    }
                else:
                    missing_cond = {}

                matches.append(
                    {'$and': [
                        {'frame_num': {'$lte': trial.n_frames[c]}},
                        {f'max_brightnesses.{c}': {'$gte': min_brightness}},
                        missing_cond
                    ]}
                )

            filters = {'__raw__': {'$or': matches}}
            frames = trial.get_frames(filters).no_dereference()
        frame_ids = [f.id for f in frames]

        if len(frame_ids) == 0:
            logger.debug('No frames missing 2d centre points!')
            continue

        # Get video readers
        readers = {}
        for c in cam_idxs:
            readers[c] = trial.get_video_reader(camera_idx=c)

        # Iterate over the frames
        for frame_id in frame_ids:
            frame = Frame.objects.get(id=frame_id)
            log_prefix = f'Frame #{frame.frame_num}/{trial.n_frames_max} (id={frame.id}). '

            # Due to the large number of frames in each trial, we need to reload from the
            # database to catch changes since we fetched the results.
            frame.reload()
            if missing_only and frame.centres_2d_available() and len(frame.centres_2d_thresholds) == 3:
                logger.info(log_prefix + 'Has 2D points, skipping.')
                continue

            # Only process cameras which need it and for which the frames can be read.
            cam_idxs_to_process = []
            errors = {}
            for c in cam_idxs:
                if missing_only and len(frame.centres_2d) == 3 and len(frame.centres_2d[c]) > 0 \
                        and len(frame.centres_2d_thresholds) == 3 and frame.centres_2d_thresholds[c] > 0:
                    errors[c] = 'Centres 2D already exist.'
                    continue
                if frame.frame_num > trial.n_frames[c]:
                    errors[c] = 'Frame out of range.'
                    continue
                if frame.max_brightnesses[c] < min_brightness:
                    errors[c] = 'Not bright enough.'
                    continue
                if len(frame.centres_2d_thresholds) == 3 \
                        and frame.centres_2d_thresholds[c] <= MIN_REQ_THRESHOLD:
                    errors[c] = 'Threshold already at minimum allowed.'
                    continue
                cam_idxs_to_process.append(c)
            if len(cam_idxs_to_process) == 0:
                logger.debug(log_prefix + f'No cameras to process: {errors}')
                continue

            # Lock the frame, or if we can't then skip it.
            if not frame.get_lock():
                logger.info(log_prefix + 'LOCKED, skipping.')
                continue

            # Generate the centres
            if len(frame.centres_2d) == 0:
                frame.centres_2d = [[]] * 3
            if len(frame.centres_2d_thresholds) == 0:
                frame.centres_2d_thresholds = [0] * 3
            for c in cam_idxs_to_process:
                log_prefix_c = log_prefix + f'Cam={c}. '
                try:
                    readers[c].set_frame_num(frame.frame_num)
                    centres, final_thresholds = readers[c].find_objects(contour_threshold_ratio)
                    logger.info(log_prefix_c + f'Found {len(centres)} objects.')
                    if centres != frame.centres_2d[c]:
                        frame.centres_2d[c] = centres
                        frame.centre_3d = None
                        frame.images = None
                    frame.centres_2d_thresholds[c] = final_thresholds
                except Exception as e:
                    logger.error(log_prefix_c + f'Failed to find objects: {e}')

                    # Catch any errors which won't log properly and re-raise
                    if str(e) == '':
                        frame.release_lock_and_save()
                        raise e

            # Unlock the frame when finished
            frame.release_lock_and_save()
            gc.collect()

        for c in cam_idxs:
            readers[c].close()
        gc.collect()


def generate_centres_3d(
        experiment_id: int = None,
        trial_id: int = None,
        frame_num: int = None,
        error_threshold: float = 50,
        missing_only: bool = True,
        fix_missing_2d: bool = False,
        store_bad_results: bool = True,
        ratio_adj_orig: float = 0,
        ratio_adj_exp: float = 0,
        ratio_adj_all: float = 0,
):
    """
    Find a unique 3d centre-point for the worm.
    This uses the 2d coordinates found in each of the 3 camera views to resolve any uncertainties.
    Note - background images and 2d centre points must be available for this to work.
    """
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, None, frame_num)
    trial_ids = [t.id for t in trials]
    if missing_only:
        logger.info(f'Generating any missing 3D centre points for {len(trials)} trials.')
    else:
        logger.info(f'(Re)generating ALL 3D centre points for {len(trials)} trials.')

    # Iterate over matching trials
    for trial_id in trial_ids:
        logger.info(f'Processing trial id={trial_id}')
        trial = Trial.objects.get(id=trial_id)
        prev_point = None

        # Filter frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            if missing_only:
                filters = {'__raw__': {
                    '$and': [
                        {'centre_3d': None},
                        {'centres_2d': {'$size': 3}},
                        {'centres_2d.0.0': {'$exists': True}},
                        {'centres_2d.1.0': {'$exists': True}},
                        {'centres_2d.2.0': {'$exists': True}},
                    ]
                }}
            else:
                filters = None
            frames = trial.get_frames(filters).only('id')

        if frames.count() == 0:
            logger.debug('No frames missing 3D centre points!')
            continue
        frame_ids = [f.id for f in frames]
        logger.debug(f'Found {len(frame_ids)} frames to process.')

        # Check if the trial has cameras
        has_exp_cameras = trial.get_cameras() is not None

        # Iterate over the frames
        for frame_id in frame_ids:
            frame = Frame.objects.get(id=frame_id)
            log_prefix = f'Frame #{frame.frame_num}/{trial.n_frames_min} (id={frame.id}). '

            # Due to the large number of frames in each trial, we need to reload from the
            # database to catch changes since we fetched the results.
            if missing_only and frame.centre_3d is not None:
                logger.info(log_prefix + 'Has 3D point, skipping.')
                continue

            # Assign the readers from the trial as reloading from the database will break this reference
            frame.trial = trial

            # 3D triangulation requires 2D points in all 3 views
            # This check is done in frame.generate_centre_3d also, but we use the above method
            # to take advantage of locks and cached video readers - better for bulk processing.
            if not frame.centres_2d_available():
                log_prefix += '2D centre points not available, '
                if fix_missing_2d:
                    logger.warning(log_prefix + 'generating now.')
                    generate_centres_2d(trial_id=trial.id, frame_num=frame.frame_num)
                    frame = frame.reload()
                else:
                    logger.warning(log_prefix + 'skipping.')
                    continue
            assert frame.centres_2d_available()

            # Lock the frame, or if we can't then skip it.
            if not frame.get_lock():
                logger.info(log_prefix + 'LOCKED, skipping.')
                continue

            logger.info(log_prefix + 'Triangulating...')
            try:
                res = frame.generate_centre_3d(
                    x0=prev_point,
                    error_threshold=error_threshold,
                    try_experiment_cams=has_exp_cameras,
                    try_all_cams=True,
                    only_replace_if_better=True,
                    store_bad_result=store_bad_results,
                    ratio_adj_orig=ratio_adj_orig,
                    ratio_adj_exp=ratio_adj_exp,
                    ratio_adj_all=ratio_adj_all,
                )
                if not res:
                    logger.error(log_prefix + f'Failed to find 3D centre.')
            except Exception as e:
                logger.error(log_prefix + f'Failed to find 3D centre: {e}')
                # raise

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
    # Frame.unlock_all()
    # Frame.reset_centres()

    generate_centres_2d(
    # trial_id=trial_id,
    # frame_num=frame_num
        missing_only=True
    )
    generate_centres_3d(
        missing_only=True,
        ratio_adj_orig=0.1,
        ratio_adj_exp=0,
        ratio_adj_all=0.1,
    )
