import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from wormlab3d import CAMERA_IDXS, PREPARED_IMAGE_SIZE, LOGS_PATH
from wormlab3d.data.model.trial import Trial
from wormlab3d.preprocessing.contour import CONT_THRESH_RATIO_DEFAULT
from wormlab3d.toolkit.triangulate import triangulate
from wormlab3d.toolkit.util import parse_target_arguments

MAX_RESULTS = 5
save_plot = False
contour_threshold_ratio: float = CONT_THRESH_RATIO_DEFAULT

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica']})


def plot_full_pipeline():
    """
    Plots a triplet of frames in:
        1) Original - with 2d centre points indicated,
        2) Cropped images around centre points.
        3) Inverted and background-subtracted, full size.
        4) Inverted, background-subtracted and cropped.
        5) Prepared images as loaded from database (if available).
    """

    # interactive_plots()

    # Fetch the trial, the video readers and the cameras
    args = parse_target_arguments()
    if args.trial is None:
        raise RuntimeError('This script must be run with the --trial=ID argument defined.')
    if args.frame_num is None:
        raise RuntimeError('This script must be run with the --frame-num=X argument defined.')

    trial = Trial.objects.get(id=args.trial)
    frame = trial.get_frame(args.frame_num)
    if len(frame.images) != 3:
        frame.generate_centre_3d()
        frame.generate_prepared_images()
        frame.save()
    reader = trial.get_video_triplet_reader()
    cameras = trial.get_cameras()

    # Set the frame number, fetch the images from each video and find objects in all 3
    reader.set_frame_num(args.frame_num)
    images = reader.get_images()
    if len(images) != 3:
        raise RuntimeError('Raw image triplet not available.')
    contours, thresholds = reader.find_contours(cont_threshold_ratios=[contour_threshold_ratio] * 3)
    centres, thresholds = reader.find_objects(cont_threshold_ratios=[contour_threshold_ratio] * 3)
    if len(centres) != 3:
        raise RuntimeError('Failed to find centres in all 3 images.')

    # Do the triangulation
    triangulation_res = triangulate(centres, cameras)[0]
    p2d = triangulation_res.reprojected_points_2d

    # Plot the results
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(12, 12),
        gridspec_kw=dict(
            wspace=0.01,
            hspace=0.01,
            # width_ratios=[1] * 3,
            top=0.95,
            bottom=0.01,
            left=0.05,
            right=0.99
        ),
    )

    crop_size = 400

    # Row 1: Originals
    for c in CAMERA_IDXS:
        ax = axes[0, c]
        ax.axis('off')

        img = cv2.cvtColor(images[c], cv2.COLOR_GRAY2RGB)
        ax.imshow(img, vmin=0, vmax=255)

        # Show region which is zoomed into on next row
        rect = patches.Rectangle(
            (
                p2d[c][0] - crop_size / 2,
                p2d[c][1] - crop_size / 2
            ),
            crop_size, crop_size, linewidth=1, edgecolor='darkblue', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

    # Row 2: Object detection and triangulation
    for c in CAMERA_IDXS:
        ax = axes[1, c]
        ax.axis('off')

        img = cv2.cvtColor(images[c], cv2.COLOR_GRAY2RGB)
        if c in contours:
            cv2.drawContours(img, contours[c], -1, (0, 255, 0), 3)

        # Zoom in
        l = max(0, round(p2d[c][1] - crop_size / 2))
        r = max(0, round(p2d[c][0] - crop_size / 2))
        img = img[l:l + crop_size + 1, r:r + crop_size + 1]
        ax.imshow(img, vmin=0, vmax=255)

        # Scatter the 2D image centre points
        centre_pts = np.stack(centres[c])
        ax.scatter(
            x=centre_pts[:, 0] - (p2d[c][0] - crop_size / 2),
            y=centre_pts[:, 1] - (p2d[c][1] - crop_size / 2),
            color='blue', s=200, alpha=0.9, marker='+', linewidths=3
        )

        # Show the 2d reprojections for the triangulated object point
        ax.scatter(
            x=p2d[c][0] - (p2d[c][0] - crop_size / 2),
            y=p2d[c][1] - (p2d[c][1] - crop_size / 2),
            color='red', s=200, alpha=0.7, marker='x', linewidths=3
        )

        # Show final crop region
        rect = patches.Rectangle(
            (
                crop_size / 2 - PREPARED_IMAGE_SIZE[0] / 2,
                crop_size / 2 - PREPARED_IMAGE_SIZE[1] / 2
            ),
            PREPARED_IMAGE_SIZE[0], PREPARED_IMAGE_SIZE[1], linewidth=1, edgecolor='darkblue', facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

    # Row 3: Prepared image
    for c in CAMERA_IDXS:
        ax = axes[2, c]
        ax.axis('off')
        ax.imshow(frame.images[c], vmin=0, vmax=1, cmap='gray')

    # Labels
    for r in range(3):
        ax = axes[r, 0]
        ax.text(-0.05, 0.5, ['a)', 'b)', 'c)'][r], transform=ax.transAxes, fontsize=20, horizontalalignment='right',
                verticalalignment='center')
    for c in range(3):
        ax = axes[0, c]
        ax.text(0.5, 1.02, f'Camera {c}', transform=ax.transAxes, fontsize=20, horizontalalignment='center',
                verticalalignment='bottom')

    if save_plot:
        os.makedirs(LOGS_PATH, exist_ok=True)
        fn = LOGS_PATH + '/' + time.strftime('%Y%m%d_%H%M') + f'_trial={trial.id}_frame={frame.frame_num}.svg'
        plt.savefig(fn)

    plt.show()


if __name__ == '__main__':
    plot_full_pipeline()
