import gc
from collections import OrderedDict
from subprocess import CalledProcessError

import cv2

from wormlab3d import logger
from wormlab3d.data.annex import is_annexed_file, fetch_from_annex
from wormlab3d.data.model import Trial
from wormlab3d.data.util import fix_path
from wormlab3d.toolkit.util import resolve_targets


def check_frames(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        check_frame_counts: bool = False,
        check_backgrounds: bool = False,
        check_brightnesses: bool = True,
        check_centres: bool = True,
        check_images: bool = True,
):
    """
    Runs a sweep of the trials and frames in the database and checks backgrounds, centres, images and brightness levels.
    """
    assert any([check_frame_counts, check_backgrounds, check_brightnesses, check_centres, check_images])
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, camera_idx)
    trial_ids = [t.id for t in trials]

    frame_counts = {}
    trials_incorrect_counts = OrderedDict()
    trials_missing_backgrounds = []
    trials_missing_centres_2d = OrderedDict()
    trials_missing_centres_3d = OrderedDict()
    trials_missing_images = OrderedDict()
    trials_missing_brightnesses = OrderedDict()

    # Iterate over matching trials
    for trial_id in trial_ids:
        logger.info(f'Checking trial id={trial_id}.')
        trial = Trial.objects.get(id=trial_id)
        if len(trial.n_frames) == 0:
            trial.n_frames = 0, 0, 0
        frame_counts[trial_id] = list(trial.n_frames)

        # Check frame counts
        if check_frame_counts:
            counts = {c: 0 for c in cam_idxs}
            for c in cam_idxs:
                reader = trial.get_video_reader(camera_idx=c)
                counts[c] = len(reader)
                reader.close()
            for c in cam_idxs:
                if trial.n_frames[c] != counts[c]:
                    trials_incorrect_counts[trial.id] = counts
                    break

        # Check backgrounds
        if check_backgrounds:
            bgs_ok = {c: False for c in cam_idxs}
            for c in cam_idxs:
                # Check for any existing background
                if len(trial.backgrounds) > 0:
                    bg_existing = None
                    bg_path = fix_path(trial.backgrounds[c])
                    if bg_path is not None:
                        if is_annexed_file(bg_path):
                            try:
                                fetch_from_annex(bg_path)
                            except CalledProcessError:
                                pass
                        bg_existing = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
                    if bg_existing is not None:
                        bgs_ok[c] = True
            if not all(bgs_ok.values()):
                trials_missing_backgrounds.append(trial.id)

        # Check for pre-computed max-brightness values
        if check_brightnesses:
            filters = {
                '__raw__': {
                    '$or': [
                        {'max_brightnesses': None},
                        {'max_brightnesses': {'$size': 0}},
                    ]
                },
                'frame_num__lt': max(trial.n_frames),
            }
            n_frames_missing_brightnesses = trial.get_frames(filters).count()
            if n_frames_missing_brightnesses > 0:
                trials_missing_brightnesses[trial.id] = n_frames_missing_brightnesses

        if check_centres:
            # 2D centres - check present in each camera
            cam_conds = []
            for c in cam_idxs:
                cam_conds.append(
                    {'$and': [
                        {f'centres_2d.{c}.0': {'$exists': False}},  # no centres for this camera
                        {'frame_num': {'$lt': trial.n_frames[c]}},  # frame is in range
                    ]}
                )

            filters = {'__raw__': {
                '$or': [
                    {'centres_2d': None},
                    {'centres_2d': {'$size': 0}},
                    *cam_conds
                ]
            }}
            n_frames_missing_centres_2d = trial.get_frames(filters).count()

            # 3D centres
            filters = {
                'centre_3d': None,
                'frame_num__lt': min(trial.n_frames),
            }
            n_frames_missing_centres_3d = trial.get_frames(filters).count()

            if n_frames_missing_centres_2d > 0:
                trials_missing_centres_2d[trial.id] = n_frames_missing_centres_2d
            if n_frames_missing_centres_3d > 0:
                trials_missing_centres_3d[trial.id] = n_frames_missing_centres_3d

        if check_images:
            filters = {
                '__raw__': {
                    '$or': [
                        {'images': None},
                        {'images': {'$size': 0}},
                    ]
                },
                'frame_num__lt': min(trial.n_frames),
            }
            n_frames_missing_images = trial.get_frames(filters).count()
            if n_frames_missing_images > 0:
                trials_missing_images[trial.id] = n_frames_missing_images

        gc.collect()

    # Avoid divide-by-zero problems
    for trial_id, f_counts in frame_counts.items():
        for c, count in enumerate(f_counts):
            if count == 0:
                frame_counts[trial_id][c] = 1

    log = ['\n\n']

    if check_backgrounds:
        log.append('------------- BACKGROUNDS -------------')
        if len(trials_missing_backgrounds) == 0:
            log.append('All present and readable!')
        else:
            trials_missing_backgrounds = OrderedDict(sorted(trials_missing_backgrounds.items()))
            log.append(
                f'{len(trials_missing_backgrounds)}/{len(trial_ids)} trials with missing or unreadable backgrounds:')
            log.append(trials_missing_backgrounds)
        log.append('\n\n')

    if check_frame_counts:
        log.append('------------- COUNTS -------------')
        if len(trials_incorrect_counts) == 0:
            log.append('All match up!')
        else:
            log.append(f'{len(trials_incorrect_counts)}/{len(trial_ids)} trials with inconsistent frame counts!')
            log.append('id: \t database:cam0/cam1/cam2 \t videos:cam0/cam1/cam2 \t %/%/%')

            for trial_id, counts in trials_incorrect_counts.items():
                log.append(
                    f'{trial_id}'
                    f'\t {frame_counts[trial_id][0]} / {frame_counts[trial_id][1]} / {frame_counts[trial_id][2]} '
                    f'\t {counts[0]} / {counts[1]} / {counts[2]} '
                    f'\t ({counts[0] / frame_counts[trial_id][0] * 100:.2f}%'
                    f' / {counts[1] / frame_counts[trial_id][0] * 100:.2f}%'
                    f' / {counts[2] / frame_counts[trial_id][0] * 100:.2f}%)')
        log.append('\n\n')

    def make_log_section(title, missing_counts, n_frames_minmax):
        log.append(f'------------- {title} -------------')
        if len(missing_counts) == 0:
            log.append('All present in database!')
        else:
            missing_counts = OrderedDict(sorted(missing_counts.items(), key=lambda item: -item[1]))
            log.append(f'{len(missing_counts)}/{len(trial_ids)} trials with missing {title}:')
            log.append('id: \t missing/total \t %')
            minmax = min if n_frames_minmax == 'min' else max
            for trial_id, count in missing_counts.items():
                n_trial_frames = minmax(frame_counts[trial_id])
                log.append(
                    f'{trial_id}'
                    f'\t {count}/{n_trial_frames} '
                    f'\t ({count / n_trial_frames * 100:.2f}%)')
        log.append('\n\n')

    if check_brightnesses:
        make_log_section('MAX BRIGHTNESSES', trials_missing_brightnesses, 'max')
    if check_centres:
        make_log_section('CENTRES 2D', trials_missing_centres_2d, 'max')
        make_log_section('CENTRES 3D', trials_missing_centres_3d, 'min')
    if check_images:
        make_log_section('PREPARED IMAGES', trials_missing_images, 'min')

    logger.info('\n'.join(log))


if __name__ == '__main__':
    check_frames()
