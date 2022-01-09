import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP, CAMERA_IDXS
from wormlab3d.data.model import Trial, Frame, ObjectPoint, Cameras
from wormlab3d.toolkit.util import resolve_targets

dry_run = False
show_plots = False
save_plots = True
img_extension = 'svg'


def smooth(X: np.ndarray, window_len=5):
    assert X.shape[0] > window_len, 'Time dimension needs to be bigger than window size.'
    assert window_len > 2, 'Window size must be > 2.'
    assert window_len % 2 == 1, 'Window size must be odd.'

    X_padded = np.r_[
        X[1:window_len // 2 + 1][::-1],
        X,
        X[-window_len // 2:-1][::-1]
    ]

    # Smoothing window
    w = np.ones(window_len, 'd')
    w /= w.sum()

    # Convolve window with trajectory
    X_s = np.convolve(w, X_padded, mode='valid')

    return X_s


def fix_tracking_trial(trial_id: int, missing_only: bool = True):
    logger.info(f'------ Fixing tracking for trial id={trial_id}.')
    trial = Trial.objects.get(id=trial_id)

    if trial.fps == 0:
        logger.error('Video error (fps=0). Cannot fix.')
        return

    frame_ids = []
    fixed_frames = []
    frame_nums = []
    frame_nums_with_data = []
    frame_nums_missing_data = []
    centres_3d = []
    bad_pts = []
    cameras_ids = []
    cams_models = {}

    # Fetch the 3d centres
    logger.info('Fetching the 3d centres.')
    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
            'centre_3d': 1,
            'centre_3d_fixed': 1
        }},
        {'$sort': {'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)

    logger.info('Collating triangulated points.')
    for i, res in enumerate(cursor):
        n = res['frame_num']

        # Check we don't miss any frames
        if i == 0:
            n0 = n
        assert n == n0 + i

        frame_nums.append(n)
        frame_ids.append(res['_id'])

        # If only fixing missed
        if 'centre_3d_fixed' in res and res['centre_3d_fixed'] is not None:
            fixed_frames.append(res['_id'])

        # Check that the centre is present
        if 'centre_3d' not in res or res['centre_3d'] is None:
            frame_nums_missing_data.append(n)
            bad_pts.append(np.array([np.nan, np.nan, np.nan]))
            cameras_ids.append(cameras_ids[-1])
            continue
        pt = np.array(res['centre_3d']['point_3d'])

        # Get the cameras model
        cams_id = res['centre_3d']['cameras']
        if cams_id not in cams_models:
            cams_models[cams_id] = Cameras.objects.get(id=cams_id).get_camera_model_triplet()
        cameras_ids.append(cams_id)

        # Check the displacement isn't too large -- set max displacement of 0.8mm/second
        if i > 0:
            displacement = np.linalg.norm(pt - centres_3d[-1])
            if displacement > (0.8 / trial.fps) * (n - frame_nums_with_data[-1]):
                frame_nums_missing_data.append(n)
                bad_pts.append(pt)
                continue

        # Check the reprojection error isn't too large -- set max of 50
        if res['centre_3d']['error'] > 50:
            frame_nums_missing_data.append(n)
            bad_pts.append(pt)
            continue

        centres_3d.append(pt)
        frame_nums_with_data.append(n)

    if len(frame_nums) == 0 or len(frame_nums_with_data) == 0:
        logger.warning('No frames found with triangulation results. Cannot fix!')
        return

    if len(frame_nums_missing_data) > len(frame_nums) * 0.2:
        logger.error(
            f'Number of missing/broken centres ({len(frame_nums_missing_data)}) > 20% of total frames ({len(frame_nums)}). '
            f'Cannot fix!'
        )
        return

    # Expect the tail to not have data, just interpolate up to the last frame with data
    tail_size = frame_nums[-1] - frame_nums_with_data[-1]
    n_missing_centres = len(frame_nums_missing_data) - tail_size
    if tail_size > 0:
        frame_nums = frame_nums[:-tail_size]
    logger.info(
        f'Number of missing centres = {n_missing_centres} '
        f'({len(frame_nums_missing_data) / len(frame_nums) * 100:.2f}%).'
    )

    # Check for already-fixed frames
    if len(fixed_frames) == len(frame_nums):
        if missing_only:
            logger.info('All frames have fixed centres already, skipping.')
            return
        else:
            logger.info('All frames have previously fixed centres, overwriting these.')
    elif len(fixed_frames) > 0:
        logger.info(f'{len(fixed_frames)} frames already contain fixed centres, overwriting all anyway.')

    # Convert to arrays
    centres_3d = np.array(centres_3d)
    frame_nums_with_data = np.array(frame_nums_with_data)
    bad_pts = np.array(bad_pts)

    # Do interpolation and smooth the good+interpolated values
    logger.info('Interpolating and smoothing good+interpolated values.')
    interpolations = []
    new_centres = np.zeros((3, len(frame_nums)))
    for i in range(3):
        existing_vals = centres_3d[:, i]
        f = interpolate.interp1d(frame_nums_with_data, existing_vals, fill_value='extrapolate')
        interpolations.append(f)
        new_vals = f(frame_nums)
        window_len = round(trial.fps)
        if window_len % 2 == 0:
            window_len += 1
        new_centres[i] = smooth(new_vals, window_len=window_len)

    # Update database
    if not dry_run:
        logger.info('Updating database.')
        for i, frame_num in enumerate(frame_nums):
            if (i + 1) % 100 == 0:
                logger.debug(f'Updating frame {i + 1}/{len(frame_nums)}')

            # Create the fixed object point
            p = ObjectPoint()
            p.point_3d = list(new_centres[:, i])

            # Re-project the point back to new 2d image points
            p.reprojected_points_2d = [
                list(cams_models[cameras_ids[i]][c].project_to_2d(p.point_3d))
                for c in CAMERA_IDXS
            ]
            Frame.objects(id=frame_ids[i]).update(centre_3d_fixed=p)

    # Make plot
    if save_plots or show_plots:
        logger.info('Plotting.')
        fig, axes = plt.subplots(3, sharex=True, figsize=(16, 16))
        fig.suptitle(f'Trial={trial_id}')
        for i in range(3):
            f = interpolations[i]
            smoothed_vals = new_centres[i]
            ax = axes[i]
            ax.set_title(['x', 'y', 'z'][i])
            ax.plot(smoothed_vals, color='green', alpha=0.5, label='Fixed')
            ax.scatter(frame_nums_with_data, f(frame_nums_with_data), s=1, color='blue', alpha=0.7, label='Included')
            if len(frame_nums_missing_data) > 0:
                ax.scatter(frame_nums_missing_data, bad_pts[:, i], s=1, color='red', alpha=0.7, label='Excluded')
                ax.scatter(frame_nums_missing_data, f(frame_nums_missing_data), s=1, color='orange', alpha=0.7,
                           label='Interpolated')
            if i == 0:
                ax.legend()

        fig.tight_layout()
        if save_plots:
            os.makedirs(LOGS_PATH, exist_ok=True)
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trial={trial_id}.{img_extension}')
        if show_plots:
            plt.show()


def fix_tracking(missing_only: bool = True):
    trials, _ = resolve_targets()
    trial_ids = [trial.id for trial in trials]
    for trial_id in trial_ids:
        fix_tracking_trial(trial_id, missing_only=missing_only)


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    fix_tracking(missing_only=True)
