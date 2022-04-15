import numpy as np

from wormlab3d import logger
from wormlab3d.data.model import Trial, Frame
from wormlab3d.data.model.trial import TrialQualityChecks, TRIAL_QUALITY_BEST, TRIAL_QUALITY_BROKEN, \
    TRIAL_QUALITY_VIDEO_ISSUES, TRIAL_QUALITY_TRACKING_ISSUES, TRIAL_QUALITY_MINOR_ISSUES, TRIAL_QUALITY_GOOD
from wormlab3d.toolkit.util import resolve_targets

dry_run = False
duration_min_frames = 100
duration_range_threshold = 0.1
min_brightness = 20
max_brightness = 220
bad_frames_threshold = 0.1


def _check_fps(trial: Trial) -> bool:
    """
    FPS is set to 0 if it cannot be read from the video or if there is too much discrepancy between the videos
    See scripts/data/fix_fps.py script for more details
    """
    return trial.fps != 0


def _check_durations(trial: Trial) -> bool:
    """
    Durations must all be close and long enough.
    """
    n_frames = np.array(trial.n_frames)
    if (n_frames < duration_min_frames).any():
        return False
    mean = n_frames.mean()
    rdtm = [abs(n - mean) / mean for n in n_frames]
    if any([r > duration_range_threshold for r in rdtm]):
        return False
    return True


def _check_brightnesses(trial: Trial) -> bool:
    """
    Check the brightnesses for all frames are defined and
    """

    # Check how many frames missing brightness values
    pipeline = [
        {'$match': {
            'trial': trial.id,
            '$or': [
                {'max_brightnesses': None},
                {'max_brightnesses': {'$size': 0}},
            ]
        }},
        {'$count': 'n_frames'}
    ]
    cursor = Frame.objects().aggregate(pipeline)
    try:
        n_frames = list(cursor)[0]['n_frames']
    except IndexError:
        n_frames = 0
    if n_frames > trial.n_frames_max * bad_frames_threshold:
        return False

    # Check brightness values
    ors = []
    for i in range(3):
        ors.append({f'max_brightnesses.{i}': {'$lte': min_brightness}})
        ors.append({f'max_brightnesses.{i}': {'$gte': max_brightness}})

    pipeline = [
        {'$match': {
            'trial': trial.id,
            '$or': ors
        }},
        {'$project': {
            '_id': 0,
            'max_brightnesses': 1,
        }},
        {'$count': 'n_frames'}
    ]
    cursor = Frame.objects().aggregate(pipeline)
    try:
        n_frames = list(cursor)[0]['n_frames']
    except IndexError:
        n_frames = 0
    if n_frames > trial.n_frames_max * bad_frames_threshold:
        return False

    return True


def _check_triangulations(trial: Trial) -> bool:
    """
    Check that triangulations are present for a sufficient proportion of frames.
    """
    pipeline = [
        {'$match': {
            'trial': trial.id,
            'centre_3d': {'$exists': True},
        }},
        {'$count': 'n_frames'}
    ]
    cursor = Frame.objects().aggregate(pipeline)
    try:
        n_frames = list(cursor)[0]['n_frames']
    except IndexError:
        n_frames = 0
    if n_frames < trial.n_frames_max * (1 - bad_frames_threshold):
        return False

    return True


def _check_fixed_triangulations(trial: Trial) -> bool:
    """
    Check that fixed triangulations are present for a sufficient proportion of frames.
    """
    pipeline = [
        {'$match': {
            'trial': trial.id,
            'centre_3d_fixed': {'$exists': True},
        }},
        {'$count': 'n_frames'}
    ]
    cursor = Frame.objects().aggregate(pipeline)
    try:
        n_frames = list(cursor)[0]['n_frames']
    except IndexError:
        n_frames = 0
    if n_frames < trial.n_frames_max * (1 - bad_frames_threshold):
        return False

    return True


def _check_tracking_video_exists(trial: Trial) -> bool:
    """
    Check that a tracking video exists.
    """
    return trial.has_tracking_video


def check_trial(trial_id: int, missing_only: bool = True):
    """
    Run all the checks on a trial and update the database.
    """
    logger.info(f'------ Checking trial id={trial_id}.')
    trial = Trial.objects.get(id=trial_id)

    if trial.quality_checks is not None:
        if missing_only:
            logger.info('Quality checks already present, skipping.')
            return
        qc = trial.quality_checks
    else:
        qc = TrialQualityChecks()

    # Run checks
    qc.fps = _check_fps(trial)
    qc.durations = _check_durations(trial)
    qc.brightnesses = _check_brightnesses(trial)
    qc.triangulations = _check_triangulations(trial)
    qc.triangulations_fixed = _check_fixed_triangulations(trial)
    qc.tracking_video = _check_tracking_video_exists(trial)

    # Check that if we have fixed triangulations then the other checks are passing, otherwise something has gone wrong.
    if qc.triangulations_fixed and not (qc.fps and qc.durations and qc.brightnesses and qc.triangulations):
        logger.warning('Fixed triangulations but other issues present!')

    # Calculate overall quality score
    if not qc.fps or not qc.durations:
        q = TRIAL_QUALITY_BROKEN
    elif not qc.brightnesses:
        q = TRIAL_QUALITY_VIDEO_ISSUES
    elif not qc.triangulations or not qc.triangulations_fixed:
        q = TRIAL_QUALITY_TRACKING_ISSUES
    elif not qc.tracking_video or not qc.syncing or not qc.crop_size:
        q = TRIAL_QUALITY_MINOR_ISSUES
    elif not qc.verified:
        q = TRIAL_QUALITY_GOOD
    else:
        q = TRIAL_QUALITY_BEST

    # Report
    logger.info(
        f'FPS={int(qc.fps)}, '
        f'durations={int(qc.durations)}, '
        f'brightnesses={int(qc.brightnesses)}, '
        f'triangulations={int(qc.triangulations)}, '
        f'triangulations_fixed={int(qc.triangulations_fixed)}, '
        f'tracking_video={int(qc.tracking_video)}, '
        f'syncing={int(qc.syncing)}, '
        f'crop_size={int(qc.crop_size)}, '
        f'verified={int(qc.verified)}'
    )
    logger.info(f'Overall quality = {q}')

    if not dry_run:
        trial.quality = q
        trial.quality_checks = qc
        trial.save()


def check_trials(missing_only: bool = True):
    trials, _ = resolve_targets()
    trial_ids = [trial.id for trial in trials]
    for trial_id in trial_ids:
        check_trial(trial_id, missing_only=missing_only)


if __name__ == '__main__':
    check_trials(missing_only=False)
