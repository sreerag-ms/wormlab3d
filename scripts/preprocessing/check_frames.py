import gc
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
        check_frame_counts: bool = True,
        check_backgrounds: bool = True,
        check_centres: bool = True,
        check_images: bool = True
):
    """
    Runs a sweep of the trials and frames in the database and checks backgrounds, centres and images.
    """
    assert any([check_frame_counts, check_backgrounds, check_centres, check_images])
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, camera_idx)
    trial_ids = [t.id for t in trials]

    frame_counts = {}
    trials_incorrect_counts = {}
    trials_missing_backgrounds = []
    trials_missing_centres_2d = {}
    trials_missing_centres_3d = {}
    trials_missing_images = {}

    # Iterate over matching trials
    for trial_id in trial_ids:
        logger.info(f'Checking trial id={trial_id}.')
        trial = Trial.objects.get(id=trial_id)
        frame_counts[trial.id] = trial.n_frames

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
            log.append('All present and correct!')
        else:
            log.append(f'{len(trials_missing_backgrounds)} trials with missing or unreadable backgrounds:')
            log.append(trials_missing_backgrounds)
        log.append('\n\n')

    if check_frame_counts:
        log.append('------------- COUNTS -------------')
        if len(trials_incorrect_counts) == 0:
            log.append('All match up!')
        else:
            log.append(f'{len(trials_incorrect_counts)} trials with inconsistent frame counts!')
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

    if check_centres:
        log.append('------------- CENTRES 2D -------------')
        if len(trials_missing_centres_2d) == 0:
            log.append('All present and correct!')
        else:
            log.append(f'{len(trials_missing_centres_2d)} trials with missing 2D centres:')
            log.append('id: \t missing/total \t %')
            for trial_id, count in trials_missing_centres_2d.items():
                log.append(
                    f'{trial_id}'
                    f'\t {count}/{max(frame_counts[trial_id])} '
                    f'\t ({count / max(frame_counts[trial_id]) * 100:.2f}%)')
        log.append('\n\n')

        log.append('------------- CENTRES 3D -------------')
        if len(trials_missing_centres_3d) == 0:
            log.append('All present and correct!')
        else:
            log.append(f'{len(trials_missing_centres_3d)} trials with missing 3D centres:')
            log.append('id: \t missing/total \t %')
            for trial_id, count in trials_missing_centres_3d.items():
                log.append(
                    f'{trial_id}'
                    f'\t {count}/{min(frame_counts[trial_id])} '
                    f'\t ({count / min(frame_counts[trial_id]) * 100:.2f}%)')
        log.append('\n\n')

    if check_images:
        log.append('------------- PREPARED IMAGES -------------')
        if len(trials_missing_images) == 0:
            log.append('All present and correct!')
        else:
            log.append(f'{len(trials_missing_images)} trials with missing prepared images:')
            log.append('id: \t missing/total \t %')
            for trial_id, count in trials_missing_images.items():
                log.append(
                    f'{trial_id}'
                    f'\t {count}/{min(frame_counts[trial_id])} '
                    f'\t ({count / min(frame_counts[trial_id]) * 100:.2f}%)')

    logger.info('\n'.join(log))


if __name__ == '__main__':
    check_frames()
