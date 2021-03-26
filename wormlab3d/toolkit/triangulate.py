import itertools
from typing import List

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

from wormlab3d import logger
from wormlab3d.data.model.cameras import CAMERA_IDXS, Cameras


def triangulate(image_points: List[np.ndarray], cameras: Cameras, x0=None):
    """
    Triangulate a list of 2D image points obtained from each camera view to produce a list of 3D object
    points which correspond up to some tolerance.
    """
    assert len(image_points) == 3
    for pts in image_points:
        assert pts.ndim == 2
        assert pts.shape[1] == 2
    logger.debug(f'Triangulating. Cameras mean reprojection error={cameras.reprojection_error:.4f}')
    cams_model = cameras.get_camera_model_triplet()

    if x0 is None:
        x0 = np.array((1, 1, 1))

    def f(x, pts):
        return np.sqrt(
            sum([
                np.linalg.norm(cams_model[c].project_to_2d(x) - pts[c])
                for c in CAMERA_IDXS
            ])
        )

    def f_explicit(x, pts):
        cam_sum = 0.
        for c in CAMERA_IDXS:
            l2 = np.linalg.norm(cams_model[c].project_to_2d(x) - pts[c])
            cam_sum += l2
        return cam_sum

    point_combinations = list(itertools.product(*image_points))
    for points in point_combinations:
        res = minimize(
            f_explicit,
            x0=x0.copy(),
            args=(points,),
            method='BFGS',
            options={
                'maxiter': 10000,  # todo: tidy this up
                'gtol': 1e-10,
                'disp': True,
            },
            tol=cameras.reprojection_error  # probably will never reach this...
        )
        logger.debug(res)

    # todo: filter the point combinations
    point_3d = res.x
    error = res.fun

    # Re-project the final 3d object point back to the 2d image points
    points_2d = [
        cams_model[c].project_to_2d(point_3d)
        for c in CAMERA_IDXS
    ]

    return point_3d, points_2d
