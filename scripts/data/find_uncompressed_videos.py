from pathlib import Path

import yaml

from wormlab3d import logger, UOL_STORE_PATH, LOGS_PATH
from wormlab3d.data.model import Trial
from wormlab3d.data.util import UOL_PATH_PLACEHOLDER
from wormlab3d.toolkit.util import resolve_targets

dry_run = True
path_cache_path = LOGS_PATH / 'path_cache.yml'
conc_dirs = {
    '0.25': '0.25%_A',
    '0.75': '0.75%_A',
    '1.00': '1%_A',
    '1.25': '1.25%_A',
    '1.50': '1.5%_A',
    '1.75': '1.75%_A',
    '2.00': '2%_A',
    '2.25': '2.25%_A',
    '2.50': '2.5%_A',
    '2.75': '2.75%_A',
    '3.00': '3%_A',
    '3.50': '3.5%_A',
    '3.75': '3.75%_A',
    '4.00': '4%_A',
}
paths = {c: [] for c in conc_dirs.keys()}


def scan_uol_store(use_cache: bool = True):
    """
    Scan the UoL store recursively to locate all video files.
    """
    global paths
    if use_cache:
        try:
            with open(path_cache_path, 'r') as f:
                paths = yaml.load(f, Loader=yaml.FullLoader)
            paths = {c: [Path(p) for p in paths_c] for c, paths_c in paths.items()}
            logger.info(f'Using path cache: {path_cache_path}')
            return
        except Exception as e:
            logger.warning(f'Could not load path cache: {e}')

    logger.info(f'Scanning the UoL store: {UOL_STORE_PATH}.')
    for c, c_dir in conc_dirs.items():
        logger.info(f'Scanning concentration folder: {UOL_STORE_PATH / c_dir}.')
        for path in (UOL_STORE_PATH / c_dir).rglob('*'):
            if path.is_dir() or path.suffix not in ['.avi', '.seq']:
                continue
            paths[c].append(path)
        logger.info(f'Found {len(paths[c])} videos.')

    if len(paths) == 0:
        raise RuntimeError('Failed to find any files. Is the path correct and mounted?')

    with open(path_cache_path, 'w') as f:
        yaml.dump({c: [str(p) for p in paths_c] for c, paths_c in paths.items()}, f)


def find_trial_videos(trial_id: int, missing_only: bool = False):
    """
    Find the videos corresponding to this trial in the UoL store.
    """
    trial = Trial.objects.get(id=trial_id)
    if missing_only and len(trial.videos_uncompressed) == 3:
        return
    logger.info(f'Looking for videos for trial={trial.id}.')

    # Check the paths for the concentration of the trial
    c = f'{trial.experiment.concentration:.2f}'
    matching_paths = [
        str(p).replace(str(UOL_STORE_PATH), UOL_PATH_PLACEHOLDER) for p in paths[c]
        if p.stem in [trial.legacy_data['legacy_id'] + f'_cam{c}' for c in range(4)]
    ]

    # Check other concentrations if we didn't find anything
    if len(matching_paths) == 0:
        matching_paths = {c: [
            str(p).replace(str(UOL_STORE_PATH), UOL_PATH_PLACEHOLDER) for p in paths[c]
            if p.stem in [trial.legacy_data['legacy_id'] + f'_cam{c}' for c in range(4)]
        ] for c in conc_dirs.keys()}
        matching_paths = {c: mp for c, mp in matching_paths.items() if len(mp) > 0}

        if len(matching_paths):
            logger.warning(f'Found videos in different concentrations: {matching_paths}')
        else:
            logger.warning(f'Couldn\'t find any videos for trial!')
        return

    # Prefer .seq over .avi
    if len(matching_paths) > 3:
        filtered_paths = []
        for p1 in matching_paths:
            ignore = False
            p1p = Path(p1)
            for p2 in matching_paths:
                p2p = Path(p2)
                if p1p.stem == p2p.stem and p1p.suffix == '.avi' and p2p.suffix == '.seq':
                    ignore = True
            if not ignore:
                filtered_paths.append(p1)
        matching_paths = filtered_paths

    # Abort if we didn't match 3 videos
    if len(matching_paths) != 3:
        err = f'Didn\'t find 3 videos for trial: {matching_paths}'
        if dry_run:
            logger.warning(err)
            return
        else:
            raise RuntimeError(err)

    # Update videos in database
    matching_paths.sort()
    if trial.videos_uncompressed == matching_paths:
        logger.info('Videos match existing.')
    else:
        if trial.videos_uncompressed is None or trial.videos_uncompressed == []:
            logger.info(f'Adding new paths to database: {matching_paths}')
        else:
            logger.warning(f'Changing paths! Old: {trial.videos_uncompressed}. New: {matching_paths}.')

        if not dry_run:
            trial.videos_uncompressed = matching_paths
            trial.save()


def find_videos(missing_only: bool = False):
    """
    Loop over all trials and find the matching videos.
    """
    trials, _ = resolve_targets()
    trial_ids = [trial.id for trial in trials]
    scan_uol_store()
    for trial_id in trial_ids:
        find_trial_videos(trial_id, missing_only)


if __name__ == '__main__':
    find_videos(missing_only=True)
