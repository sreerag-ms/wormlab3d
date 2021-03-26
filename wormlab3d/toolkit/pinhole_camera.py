import numpy as np


class PinholeCamera:
    """
    A pinhole camera model.
    """

    def __init__(self, pose: np.ndarray, matrix: np.ndarray, distortion: np.ndarray = None):
        assert pose.shape == (4, 4)
        assert matrix.shape == (3, 3)
        if distortion is not None:
            assert distortion.shape == (5,)
        self.pose = pose
        self.matrix = matrix
        self.distortion = distortion
        self.rotation = pose[:3, :3]
        self.translation = pose[:3, 3]

    def project_to_2d(self, image_point: np.ndarray, distort: bool = True, shift: np.ndarray = None) -> np.ndarray:
        """
        Project 3D object point down to the 2D image plane.
        """
        assert image_point.shape == (3,)
        x = np.matmul(self.rotation, image_point) + self.translation
        x, y = x[0] / x[2], x[1] / x[2]
        fx = self.matrix[0, 0]
        fy = self.matrix[1, 1]

        # if shift is not None:  todo
        #     x += shift[0] / fx
        #     y += shift[1] / fy

        if distort and self.distortion is not None:
            # todo: where does this calculation come from?
            k1, k2, p1, p2, k3 = self.distortion
            r2 = x * x + y * y
            x, y = (x * (1 + r2 * (k1 + r2 * (k2 + k3 * r2))
                         + 2. * p1 * y) + p2 * (r2 + 2. * x * x),
                    y * (1 + r2 * (k1 + r2 * (k2 + k3 * r2))
                         + 2. * p2 * x) + p1 * (r2 + 2. * y * y))

        cx = self.matrix[0, 2]
        cy = self.matrix[1, 2]

        out = np.array((fx * x + cx, fy * y + cy))

        return out
