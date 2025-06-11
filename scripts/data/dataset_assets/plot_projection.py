import json
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

show_plots = True
save_plots = True
plot_frame_idx = 692


class PinholeCamera:
    """
    A pinhole camera model.
    """

    def __init__(
            self,
            pose: np.ndarray,
            matrix: np.ndarray,
            distortion: np.ndarray = None,
            shift: Tuple[float, float] = None
    ):
        assert pose.shape == (4, 4)
        assert matrix.shape == (3, 3)
        if distortion is not None:
            assert distortion.shape == (5,)
        self.pose = pose
        self.matrix = matrix
        self.distortion = distortion
        self.rotation = pose[:3, :3]
        self.translation = pose[:3, 3]
        self.shift = shift

    def project_to_2d(
            self,
            image_point: Union[np.ndarray, list],
            distort: bool = True
    ) -> np.ndarray:
        """
        Project 3D object point down to the 2D image plane.
        """
        if not isinstance(image_point, np.ndarray):
            image_point = np.array(image_point)
        assert image_point.shape == (3,)

        x = np.matmul(self.rotation, image_point) + self.translation
        x, y = x[0] / x[2], x[1] / x[2]
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]

        if self.shift is not None:
            x += self.shift[0] / fx
            y += self.shift[1] / fy

        if distort and self.distortion is not None:
            k1, k2, p1, p2, k3 = self.distortion
            r2 = x * x + y * y
            x, y = (x * (1 + r2 * (k1 + r2 * (k2 + k3 * r2))
                         + 2. * p1 * y) + p2 * (r2 + 2. * x * x),
                    y * (1 + r2 * (k1 + r2 * (k2 + k3 * r2))
                         + 2. * p2 * x) + p1 * (r2 + 2. * y * y))

        cx = self.matrix[0, 2]
        cy = self.matrix[1, 2]

        out = np.array([fx * x + cx, fy * y + cy])

        return out


class CameraModelTriplet:
    """
    Class containing three PinholeCamera models parametrised from a data model instance.
    """

    def __init__(
            self,
            poses: np.ndarray,
            matrices: np.ndarray,
            distortions: np.ndarray = None,
            shifts: np.ndarray = None,
    ):
        self.cameras = [
            PinholeCamera(
                pose=poses[c],
                matrix=matrices[c],
                distortion=distortions[c] if distortions is not None else None,
                shift=shifts[c] if shifts is not None else None,
            )
            for c in [0, 1, 2]
        ]

    def __getitem__(self, i: int) -> PinholeCamera:
        """Return one of the cameras."""
        return self.cameras[i]

    def project_to_2d(self, object_points: np.ndarray, distort: bool = True) -> np.ndarray:
        """
        Takes points in 3d object coordinates and projects them down triplets of 2d image coordinates.
        """
        if object_points.ndim == 1:
            object_points = object_points[np.newaxis, :]
        assert object_points.ndim == 2
        assert object_points.shape[1] == 3

        image_points = np.zeros((3, len(object_points), 2))
        for i, pts in enumerate(object_points):
            image_points[:, i] = np.array([
                self[c].project_to_2d(pts, distort=distort)
                for c in [0, 1, 2]
            ])

        return image_points


# Load the camera parameters
with open('../cameras/trial=%%TRIAL_ID%%_camera_parameters.json', 'r') as f:
    camera_params = json.load(f)
poses = np.array(camera_params['pose'])  # shape: (n_frames, 4, 4)
matrices = np.array(camera_params['matrix'])  # shape: (n_frames, 3, 3)
distortions = camera_params.get('distortion', None)  # shape: (n_frames, 5)
if distortions:
    distortions = np.array(distortions)
shifts = camera_params.get('shift', None)
if shifts:
    shifts = shifts[plot_frame_idx]  # Note that the shifts are per frame

# Load the camera models
cameras = CameraModelTriplet(
    poses=poses,
    matrices=matrices,
    distortions=distortions,
    shifts=shifts
)

# Load the reconstruction
data = np.load('../reconstruction_xyz/trial=%%TRIAL_ID%%_reconstruction=%%RECONSTRUCTION_ID%%.npz')
X = data['X']  # shape: (n_frames, n_body_points, 3)
N = X.shape[1]

# Project the 3D coordinates to 2D using the camera models
Y = cameras.project_to_2d(X[plot_frame_idx])

# Construct colours
colours = np.linspace(0, 1, X.shape[1])
cmap = plt.get_cmap('viridis_r')

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Calculate the maximum range across all cameras
x_ranges = [Y[i][:, 0].max() - Y[i][:, 0].min() for i in range(3)]
y_ranges = [Y[i][:, 1].max() - Y[i][:, 1].min() for i in range(3)]
max_range = max(max(x_ranges), max(y_ranges)) * 1.2  # Use the largest range for both axes

for i, ax in enumerate(axes):
    # Plot the 2D projection for each camera
    ax.scatter(Y[i][:, 0], Y[i][:, 1], c=colours, cmap=cmap, s=10, alpha=0.4)
    ax.set_title(f'Camera {i + 1}')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    x_center = (Y[i][:, 0].max() + Y[i][:, 0].min()) / 2
    y_center = (Y[i][:, 1].max() + Y[i][:, 1].min()) / 2
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
    ax.set_aspect(1, 'box')

fig.tight_layout()

if save_plots:
    plt.savefig('projected_2d.png')
if show_plots:
    plt.show()
