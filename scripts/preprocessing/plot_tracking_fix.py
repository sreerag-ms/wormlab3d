import os
from argparse import Namespace, ArgumentParser
from typing import Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from scipy import interpolate

from wormlab3d import logger, LOGS_PATH, START_TIMESTAMP, CAMERA_IDXS
from wormlab3d.data.model import Trial, Frame, Cameras
from wormlab3d.preprocessing.contour import CONT_THRESH_RATIO_DEFAULT
from wormlab3d.toolkit.util import print_args
from wormlab3d.trajectories.util import smooth_trajectory

save_plots = True
show_plots = True
img_extension = 'svg'
use_uncompressed_videos = True
contour_threshold_ratio: float = CONT_THRESH_RATIO_DEFAULT


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to plot the tracking fixes.')

    parser.add_argument('--trial', type=int, required=True, help='Trial by id.')
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.')
    parser.add_argument('--end-frame', type=int, help='Frame number to end at.')
    parser.add_argument('--x-label', type=str, default='frame', help='Label x-axis with time or frame number.')
    parser.add_argument('--highlight-frame', type=int, help='Highlight this frame.')
    args = parser.parse_args()

    print_args(args)

    return args


def calculate_fixes(trial: Trial, args: Namespace, centre: bool = False) -> Dict[str, Any]:
    """
    Calculate the tracking fixes.
    """
    logger.info(f'Calculating tracking fixes for trial={trial.id}.')
    if trial.fps == 0:
        logger.error('Video error (fps=0). Cannot fix.')
        return

    frame_ids = []
    frame_nums = []
    frame_nums_with_data = []
    frame_nums_missing_data = []
    centres_3d = []
    bad_pts = []
    cameras_ids = []
    cams_models = {}
    last_good_cams_id = None

    # Fetch the 3d centres
    logger.info('Fetching the 3d centres.')
    start_frame = args.start_frame if args.start_frame is not None else 0
    end_frame = args.end_frame if args.end_frame is not None else trial.n_frames_max + 1
    pipeline = [
        {'$match': {
            'trial': trial.id,
            'frame_num': {'$gte': start_frame, '$lte': end_frame}
        }},
        {'$project': {
            '_id': 1,
            'frame_num': 1,
            'centre_3d': 1,
        }},
        {'$sort': {'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)

    def bad_pt_abort(n_, bad_pt):
        frame_nums_missing_data.append(n_)
        bad_pts.append(bad_pt)
        if last_good_cams_id is not None:
            cameras_ids.append(last_good_cams_id)
        else:
            cameras_ids.append(None)

    logger.info('Collating triangulated points.')
    for i, res in enumerate(cursor):
        n = res['frame_num']

        # Check we don't miss any frames
        if i == 0:
            n0 = n
        assert n == n0 + i

        frame_nums.append(n)
        frame_ids.append(res['_id'])

        # Check that the centre is present
        if 'centre_3d' not in res or res['centre_3d'] is None:
            bad_pt_abort(n, np.array([np.nan, np.nan, np.nan]))
            continue
        pt = np.array(res['centre_3d']['point_3d'])

        # Check the displacement isn't too large -- set max displacement of 0.8mm/second
        if len(centres_3d) > 1:
            displacement = np.linalg.norm(pt - centres_3d[-1])
            if displacement > (0.5 / trial.fps) * (n - frame_nums_with_data[-1]):
                bad_pt_abort(n, pt)
                continue

        # Check the reprojection error isn't too large -- set max of 50
        if res['centre_3d']['error'] > 50:
            bad_pt_abort(n, pt)
            continue

        # Get the cameras model
        cams_id = res['centre_3d']['cameras']
        if cams_id not in cams_models:
            cams_models[cams_id] = Cameras.objects.get(id=cams_id).get_camera_model_triplet()

        last_good_cams_id = cams_id
        cameras_ids.append(cams_id)
        centres_3d.append(pt)
        frame_nums_with_data.append(n)

    if len(frame_nums) == 0 or len(frame_nums_with_data) == 0:
        raise RuntimeError('No frames found with triangulation results. Cannot calculate fixes!')

    # Expect the tail to not have data, just interpolate up to the last frame with data
    tail_size = frame_nums[-1] - frame_nums_with_data[-1]
    n_missing_centres = len(frame_nums_missing_data) - tail_size
    if tail_size > 0:
        frame_nums = frame_nums[:-tail_size]
    logger.info(
        f'Number of missing centres = {n_missing_centres} '
        f'({len(frame_nums_missing_data) / len(frame_nums) * 100:.2f}%).'
    )

    # Convert to arrays
    centres_3d = np.array(centres_3d)
    frame_nums_with_data = np.array(frame_nums_with_data)
    bad_pts = np.array(bad_pts)

    # Centre the points
    if centre:
        all_pts = np.concatenate([centres_3d, bad_pts])
        mean_pt = all_pts.mean(axis=0)
        centres_3d -= mean_pt
        bad_pts -= mean_pt

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
        new_centres[i] = smooth_trajectory(new_vals, window_len=window_len)

    # Fix any missing cameras at the start due to rejected points at the start of the video
    i = 0
    while cameras_ids[i] is None:
        i += 1
    if i > 0:
        for j in range(i):
            cameras_ids[j] = cameras_ids[i]

    return {
        'interpolations': interpolations,
        'new_centres': new_centres,
        'frame_nums_with_data': np.array(frame_nums_with_data),
        'frame_nums_missing_data': np.array(frame_nums_missing_data),
        'good_pts': centres_3d,
        'bad_pts': bad_pts
    }


def plot_tracking_fix():
    """
    Plot the fixed tracking for given trial.
    """
    args = get_args()
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'------ Fixing tracking for trial id={trial.id}.')
    res = calculate_fixes(trial, args, centre=True)
    frame_nums = np.concatenate([res['frame_nums_with_data'], res['frame_nums_missing_data']])
    frame_nums.sort()

    # Make plot
    logger.info('Plotting.')
    plt.rc('axes', labelsize=8)  # fontsize of the Y label
    plt.rc('xtick', labelsize=7)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=7)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=9)  # fontsize of the legend

    fig, axes = plt.subplots(3, figsize=(5.8, 3.5), sharex=True, gridspec_kw={
        'left': 0.1,
        'right': 0.99,
        'top': 0.98,
        'bottom': 0.1,
        'hspace': 0.03,
    })

    scatter_args = dict(
        s=10,
        marker='o',
        facecolors='none',
        alpha=0.5
    )
    highlight_args = dict(
        s=100,
        marker='o',
        facecolors='none',
        alpha=0.8,
        linewidth=3,
        zorder=100
    )

    for i in range(3):
        ax = axes[i]
        ax.set_ylabel(['x', 'y', 'z'][i] + '-position (mm)')
        ax.axhline(y=0, color='lightgrey', zorder=-10)

        # Plot the smoothed values
        smoothed_vals = res['new_centres'][i]
        ax.plot(frame_nums, smoothed_vals, color='green', linewidth=2, alpha=0.8, label='Fixed')

        # Plot the good values
        ax.scatter(res['frame_nums_with_data'], res['good_pts'][:, i],
                   edgecolors='blue', label='Included', **scatter_args)

        # Plot the bad values and their fixed interpolations
        if len(res['frame_nums_missing_data']) > 0:
            f = res['interpolations'][i]
            ax.scatter(res['frame_nums_missing_data'], res['bad_pts'][:, i],
                       edgecolors='red', label='Excluded', **scatter_args)
            ax.scatter(res['frame_nums_missing_data'], f(res['frame_nums_missing_data']),
                       edgecolors='orange', label='Interpolated', **scatter_args)

        # Add highlighted-frames markers
        ylim = ax.get_ylim()
        if args.highlight_frame is not None:
            n = args.highlight_frame
            ax.vlines(x=n, ymin=ylim[0] * 1.2, ymax=ylim[1] * 1.2, linestyle='-', linewidth=highlight_args['linewidth'],
                      alpha=highlight_args['alpha'], color='darkviolet', zorder=highlight_args['alpha'] - 1)
            frame_idx = (res['frame_nums_missing_data'] == n).nonzero()[0][0]
            ax.scatter(x=n, y=res['bad_pts'][frame_idx, i], edgecolors='red', **highlight_args)
            ax.scatter(x=n, y=smoothed_vals[n - frame_nums[0]], edgecolors='green', **highlight_args)
        ax.set_ylim(ylim)

        if i == 0:
            legend = ax.legend(loc='upper left')
            legend.legendHandles[0].set_alpha(1.)
            legend.legendHandles[1].set_sizes([50])
            legend.legendHandles[1].set_alpha(1.)
            legend.legendHandles[2].set_sizes([50])
            legend.legendHandles[2].set_alpha(1.)
            legend.legendHandles[3].set_sizes([50])
            legend.legendHandles[3].set_alpha(1.)
        if i == 2:
            ax.set_xlabel('Frame')

    fig.tight_layout()
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}'
                                f'_trial={trial.id}'
                                f'_f={args.start_frame}-{args.end_frame}'
                                f'.{img_extension}',
                    transparent=True)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_tracking_fix_examples():
    """
    Plot cropped examples of the fixed tracking at specific frames.
    """
    args = get_args()
    trial = Trial.objects.get(id=args.trial)
    logger.info(f'------ Fixing tracking for trial id={trial.id}.')
    res = calculate_fixes(trial, args, centre=False)
    frame_nums = np.concatenate([res['frame_nums_with_data'], res['frame_nums_missing_data']])
    frame_nums.sort()
    cameras = trial.get_cameras()
    cams_model = cameras.get_camera_model_triplet()

    # Set crop sizes
    crop_size_final = trial.crop_size
    crop_size_intermediate = int(crop_size_final * 1.6)

    # Project the good and bad points
    frame_idx = args.highlight_frame - frame_nums[0]
    p2d_good = cams_model.project_to_2d(res['new_centres'][:, frame_idx])[0]
    frame_idx = (res['frame_nums_missing_data'] == args.highlight_frame).nonzero()[0][0]
    p2d_bad = cams_model.project_to_2d(res['bad_pts'][frame_idx, :])[0]
    midpoint = (p2d_good + p2d_bad) / 2

    # Set the frame number, fetch the images from each video and find objects in all 3
    reader = trial.get_video_triplet_reader(use_uncompressed_videos=use_uncompressed_videos)
    reader.set_frame_num(args.highlight_frame)
    images = reader.get_images()
    if len(images) != 3:
        raise RuntimeError('Raw image triplet not available.')
    contours, thresholds = reader.find_contours(cont_threshold_ratios=[contour_threshold_ratio] * 3)

    # Plot the results
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(5.7, 1.9),
        gridspec_kw=dict(
            wspace=0.01,
            top=0.99,
            bottom=0.01,
            left=0.01,
            right=0.99
        ),
    )

    scatter_args = dict(
        s=200,
        alpha=0.8,
        marker='x',
        linewidths=3
    )
    rect_args = dict(
        width=crop_size_final,
        height=crop_size_final,
        linewidth=2,
        facecolor='none',
        linestyle='--',
        alpha=0.7,
    )

    for c in CAMERA_IDXS:
        ax = axes[c]
        ax.axis('off')

        # Draw contours
        img = cv2.cvtColor(images[c], cv2.COLOR_GRAY2RGB)
        if c in contours:
            cv2.drawContours(img, contours[c], -1, (0, 0, 200), 3)

        # Zoom in
        l = max(0, round(midpoint[c][1] - crop_size_intermediate / 2))
        r = max(0, round(midpoint[c][0] - crop_size_intermediate / 2))
        img = img[l:l + crop_size_intermediate + 1, r:r + crop_size_intermediate + 1]
        ax.imshow(img, vmin=0, vmax=255)

        # # Show the bad tracking point
        x_bad = p2d_bad[c][0] - (midpoint[c][0] - crop_size_intermediate / 2)
        y_bad = p2d_bad[c][1] - (midpoint[c][1] - crop_size_intermediate / 2)
        ax.scatter(x=x_bad, y=y_bad, color='red', **scatter_args)
        rect = patches.Rectangle(
            (
                x_bad - crop_size_final / 2,
                y_bad - crop_size_final / 2
            ),
            edgecolor='red', **rect_args
        )
        ax.add_patch(rect)

        # Show the good tracking point
        x_good = p2d_good[c][0] - (midpoint[c][0] - crop_size_intermediate / 2)
        y_good = p2d_good[c][1] - (midpoint[c][1] - crop_size_intermediate / 2)
        ax.scatter(x=x_good, y=y_good, color='chartreuse', **scatter_args)
        rect = patches.Rectangle(
            (
                x_good - crop_size_final / 2,
                y_good - crop_size_final / 2
            ),
            edgecolor='chartreuse', **rect_args
        )
        ax.add_patch(rect)

    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}'
                                f'_trial={trial.id}'
                                f'_f={args.highlight_frame}'
                                f'_images'
                                f'.{img_extension}',
                    transparent=True)
    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    plot_tracking_fix()
    plot_tracking_fix_examples()
