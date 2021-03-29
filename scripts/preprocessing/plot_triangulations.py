import cv2
import matplotlib.pyplot as plt
import numpy as np

from wormlab3d.data.model.cameras import CAMERA_IDXS
from wormlab3d.data.model.trial import Trial
from wormlab3d.toolkit.plot_utils import interactive_plots
from wormlab3d.toolkit.triangulate import triangulate


def plot_triangulations(trial_id, frame_num=1):
    interactive_plots()

    # Fetch the trial, the video readers and the cameras
    trial = Trial.objects.get(id=trial_id)
    reader = trial.get_video_triplet_reader()
    cameras = trial.experiment.get_cameras()

    # Set the frame number, fetch the images from each video and find objects in all 3
    reader.set_frame_num(frame_num)
    images = reader.get_images()
    contours = reader.find_contours()
    centres = reader.find_objects()

    # Do the triangulation
    res_3d = triangulate(centres, cameras)

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
        centre_pts = np.stack(centres[c])
        ax.scatter(x=centre_pts[:, 0], y=centre_pts[:, 1], color='blue', s=10, alpha=0.8)
        for r in res_3d:
            ax.scatter(
                x=r.reprojected_points_2d[c][0],
                y=r.reprojected_points_2d[c][1],
                color='red', s=10, alpha=0.8
            )

    plt.show()


if __name__ == '__main__':
    plot_triangulations(
        trial_id=4,
        frame_num=120
    )
