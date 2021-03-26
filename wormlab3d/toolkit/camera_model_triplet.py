from typing import List

import numpy as np

from wormlab3d.data.model.cameras import Cameras, CAMERA_IDXS
from wormlab3d.toolkit.pinhole_camera import PinholeCamera


class CameraModelTriplet:
    """
    Class containing three PinholeCamera models parametrised from a data model instance.
    """

    def __init__(self, camera_models: Cameras, distort: bool = True, shift_fn=None):
        # todo: shifts
        # self._shift_seek = 0
        # if shift_fn is not None:
        #     shifts = datafile_by_index(shift_fn, ["dx", "dy", "dz"])
        #     self._shifts = iter(shifts)

        self.cameras = [
            PinholeCamera(
                pose=camera_models.pose[c],
                matrix=camera_models.matrix[c],
                distortion=camera_models.distortion[c] if distort else None,
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

        # # todo: shifts
        # if shift is not None:
        #     assert self._shifts is not None
        #     for i in range(shift):
        #         # BUG: assume stride is 10.
        #         shift_stride = 10
        #         if not self._shift_seek % shift_stride:
        #             sh = next(self._shifts)
        #             # BUG: assuming default camera layout
        #             self._shs = (sh[0], 0), (0, -sh[1]), (0, sh[2])
        #         self._shift_seek += 1
        # else:
        #     self._shs = [None] * 3
        # shifts = self._shs

        image_points = []
        for pts in object_points:
            image_points.append([
                self[c].project_to_2d(pts, distort=distort)  # shift=shifts[k])
                for c in CAMERA_IDXS
            ])

        return image_points
