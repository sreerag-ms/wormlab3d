import gc

from wormlab3d import logger, CAMERA_IDXS
from wormlab3d.data.model import Trial, Frame
from wormlab3d.toolkit.util import resolve_targets


def generate_frames(
        experiment_id: int = None,
        trial_id: int = None,
        frame_num: int = None,
        remove_surplus_frames: bool = True,
        add_missing_frames: bool = True,
        fix_brightnesses: bool = True,
        fix_brightnesses_missing_only: bool = True,
):
    """
    Checks all the trials in the database to get the actual frame counts according to what is readable from the videos,
    populates the Frame collection with entries to hold the per-frame information (either adding missing or removing surplus),
    and calculates the max brightness per-camera per-frame.
    """
    trials, _ = resolve_targets(experiment_id, trial_id)
    trial_ids = [t.id for t in trials]

    # Iterate over matching trials
    for trial_id in trial_ids:
        logger.info(f'Checking trial id={trial_id}.')
        trial = Trial.objects.get(id=trial_id)

        # Get video readers and check frame counts
        reader = trial.get_video_triplet_reader()
        n_frames = reader.get_frame_counts()
        max_n_frames = max(n_frames)
        matching = [False] * 3
        for c in CAMERA_IDXS:
            if len(trial.n_frames) == 3 and trial.n_frames[c] == n_frames[c]:
                matching[c] = True

        # Check frame counts
        if sum(matching) != 3:
            logger.warning(f'{3 - sum(matching)} frame counts incorrect. Fixing.')
            trial.n_frames = n_frames
            trial.n_frames_min = min(n_frames)
            trial.n_frames_max = max(n_frames)
            trial.save()
        elif trial.n_frames_min != min(n_frames) or trial.n_frames_max != max(n_frames):
            logger.warning(f'Missing/incorrect min/max frame counts. Fixing.')
            trial.n_frames_min = min(n_frames)
            trial.n_frames_max = max(n_frames)
            trial.save()
        else:
            logger.info('Frame counts OK.')

        # Check for surplus frames
        surplus = trial.get_frames({'frame_num__gt': max_n_frames})
        if surplus.count() > 0:
            log_prefix = f'Found {surplus.count()} frames in database without corresponding/readable video frames. '
            if remove_surplus_frames:
                logger.warning(log_prefix + 'Removing.')
                Frame.objects(
                    id__in=[f.id for f in surplus]
                ).delete()
            else:
                logger.warning(log_prefix + 'Ignoring.')

        # Check for missing frames
        existing = trial.get_frames({'frame_num__lte': max_n_frames})
        if existing.count() < max_n_frames:
            log_prefix = f'Found {existing.count()}/{max_n_frames} frames in database. '
            if add_missing_frames:
                logger.warning(log_prefix + f'Adding {max_n_frames - existing.count()} missing frames.')
                existing_frame_nums = [f.frame_num for f in existing]

                # Add missing frames
                frames = []
                for i in range(max_n_frames):
                    if i in existing_frame_nums:
                        continue
                    frame = Frame()
                    frame.trial = trial
                    frame.experiment = trial.experiment
                    frame.frame_num = i
                    frames.append(frame)
                Frame.objects.insert(frames)
            else:
                logger.warning(log_prefix + 'Ignoring.')

        # Find the brightnesses for each frame
        if fix_brightnesses:
            # Iterate over the frames
            if frame_num is not None:
                frames = [trial.get_frame(frame_num)]
            else:
                if fix_brightnesses_missing_only:
                    filters = {'__raw__': {
                        '$or': [
                            {'max_brightnesses': None},
                            {'max_brightnesses': {'$size': 0}},
                        ]
                    }}
                else:
                    filters = {}
                frames = trial.get_frames(filters).no_dereference()

            frame_ids = [f.id for f in frames]
            for frame_id in frame_ids:
                frame = Frame.objects.get(id=frame_id)
                log_prefix = f'Frame #{frame.frame_num}/{trial.n_frames_max} (id={frame.id}). '

                # Check to see if the brightnesses have been calculated already.
                if fix_brightnesses_missing_only and len(frame.max_brightnesses) == 3:
                    logger.info(log_prefix + 'Brightness levels already set, skipping.')
                    continue

                # Lock the frame, or if we can't then skip it.
                if not frame.get_lock():
                    logger.info(log_prefix + 'LOCKED, skipping.')
                    continue

                # Read the images from all 3 cameras and determine max brightness for each.
                reader.set_frame_num(frame.frame_num)
                images = reader.get_images(invert=True, subtract_background=True)
                max_brightnesses = [0] * 3
                matching = [False] * 3
                for c in CAMERA_IDXS:
                    if c in images:
                        max_brightnesses[c] = images[c].max()
                    if len(frame.max_brightnesses) == 3 and frame.max_brightnesses[c] == max_brightnesses[c]:
                        matching[c] = True

                # Update database if needed
                if sum(matching) != 3:
                    logger.warning(log_prefix + f'{3 - sum(matching)} frame max brightnesses incorrect. Fixing.')
                    frame.max_brightnesses = max_brightnesses
                else:
                    logger.info(log_prefix + 'Brightnesses OK.')
                frame.release_lock_and_save()

        reader.close()
        gc.collect()


if __name__ == '__main__':
    generate_frames()
