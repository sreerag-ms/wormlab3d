import cv2
import matplotlib.pyplot as plt
import numpy as np

from wormlab3d import CAMERA_IDXS, logger
from wormlab3d.data.model.trial import Trial
from wormlab3d.preprocessing.contour import CONT_THRESH_DEFAULT
from wormlab3d.toolkit.plot_utils import interactive_plots
from wormlab3d.toolkit.triangulate import triangulate

MAX_RESULTS = 5


def plot_triangulations(
        trial_id: int,
        frame_num: int = 1,
        contour_threshold: float = CONT_THRESH_DEFAULT
):
    interactive_plots()

    # Fetch the trial, the video readers and the cameras
    trial = Trial.objects.get(id=trial_id)
    reader = trial.get_video_triplet_reader()
    cameras_trial = trial.get_cameras()

    # Set the frame number, fetch the images from each video and find objects in all 3
    reader.set_frame_num(frame_num)
    images = reader.get_images()
    contours = reader.find_contours(cont_threshold=contour_threshold)
    centres = reader.find_objects(cont_threshold=contour_threshold)

    # Do the triangulation from all experiment cameras
    exp_cameras = trial.experiment.get_cameras(best=False)
    cameras_res = []
    for cameras in exp_cameras:
        logger.debug(
            f'Trying cameras id={cameras.id}. '
            f'Cam error={cameras.reprojection_error}. '
            f'Trial={cameras.trial.id if cameras.trial is not None else "-"}')
        try:
            res_3d = triangulate(centres, cameras)
            cameras_res.extend(res_3d)
        except ValueError:
            logger.warning('Triangulation failed')
    cameras_res.sort(key=lambda r: r.error)
    cameras_res = cameras_res[:MAX_RESULTS]

    # Plot the results
    fig, axes = plt.subplots(1, 3)
    fig.suptitle(
        f'{trial.date:%Y%m%d} Trial #{trial.trial_num}. '
        f'Frame #{frame_num}.'
    )
    for c in CAMERA_IDXS:
        img = cv2.cvtColor(images[c], cv2.COLOR_GRAY2RGB)
        cv2.drawContours(img, contours[c], -1, (0, 255, 0), 3)
        ax = axes[c]
        ax.set_title(trial.videos[c])
        ax.imshow(img, vmin=0, vmax=255)

        # Scatter the 2D image centre points
        centre_pts = np.stack(centres[c])
        ax.scatter(
            x=centre_pts[:, 0],
            y=centre_pts[:, 1],
            color='blue', s=10, alpha=0.8, marker='x',
            label='2D centres'
        )

        # Show the 2d reprojections for the top MAX_RESULTS object points found
        for r in cameras_res:
            is_own = 'OWN ' if cameras_trial.id == r.cameras.id else ''
            ax.scatter(
                x=r.reprojected_points_2d[c][0],
                y=r.reprojected_points_2d[c][1],
                s=10, alpha=0.8,
                label=f'{is_own}error={r.error:.2f}'
            )
            ax.axis('off')
        ax.legend()

    plt.show()


if __name__ == '__main__':
    plot_triangulations(
        # bubble close to worm, detected as second object
        # trial_id=287,
        # frame_num=0,

        # Poor error, spot obscured in one view
        # trial_id=186,
        # frame_num=823,

        # trial_id=4,
        # frame_num=120

        # Lots of 2d points - fails with default threshold, but works with 0.3
        trial_id=301,
        frame_num=79,
        contour_threshold=0.3
    )
