from typing import List

import numpy as np

from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model.cameras import Cameras
from wormlab3d.toolkit.pinhole_camera import PinholeCamera


class CameraModelTriplet:
    """
    Class containing three PinholeCamera models parametrised from a data model instance.
    """

    def __init__(self, camera_models: Cameras, distort: bool = True):
        self.cameras = [
            PinholeCamera(
                pose=camera_models.pose[c],
                matrix=camera_models.matrix[c],
                distortion=camera_models.distortion[c] if distort else None,
                shift=camera_models.get_shift(c),
            )
            for c in CAMERA_IDXS
        ]

    def __getitem__(self, i: int) -> PinholeCamera:
        """Return one of the cameras."""
        return self.cameras[i]

    def project_to_2d(self, object_points: np.ndarray, distort: bool = True) -> List[np.ndarray]:
        """
        Takes points in 3d object coordinates and projects them down triplets of 2d image coordinates.
        """
        if object_points.ndim == 1:
            object_points = object_points[np.newaxis, :]
        assert object_points.ndim == 2
        assert object_points.shape[1] == 3

        image_points = []
        for pts in object_points:
            image_points.append(np.array([
                self[c].project_to_2d(pts, distort=distort)
                for c in CAMERA_IDXS
            ]))

        return image_points
