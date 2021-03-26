import itertools
from typing import List

import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize

from wormlab3d import logger
from wormlab3d.data.model.cameras import CAMERA_IDXS, Cameras


def triangulate(
        image_points: List[np.ndarray],
        cameras: Cameras,
        x0: np.ndarray = None,
        matching_threshold: float = 100,
) -> list:
    """
    Triangulate a list of 2D image points obtained from each camera view to produce a list of 3D object
    points which correspond up to some tolerance.
    """
    assert len(image_points) == 3
    for pts in image_points:
        assert pts.ndim == 2
        assert pts.shape[1] == 2
    logger.debug(f'Triangulating. Cameras mean reprojection error={cameras.reprojection_error:.4f}.')
    cams_model = cameras.get_camera_model_triplet()

    if x0 is None:
        x0 = np.array((1, 1, 1))

    def f(x, pts):
        return np.sqrt(
            sum([
                np.linalg.norm(cams_model[c].project_to_2d(x) - pts[c])**2
                for c in CAMERA_IDXS
            ])
        )

    point_keys = [list(range(len(pts))) for pts in image_points]
    point_key_combinations = list(itertools.product(*point_keys))
    res_3d = []
    for point_key_combination in point_key_combinations:
        points_2d = [
            image_points[0][point_key_combination[0]],
            image_points[1][point_key_combination[1]],
            image_points[2][point_key_combination[2]],
        ]
        res = minimize(
            f,
            x0=x0.copy(),
            args=(points_2d,),
            method='BFGS',
            options={
                'maxiter': 1000,
                'gtol': 1e-4,
                # 'disp': LOG_LEVEL == 'DEBUG',
            },
            tol=cameras.reprojection_error  # probably will never reach this
        )

        res_3d.append({
            'pt': res.x,
            'error': res.fun,
            'source_point_idxs': point_key_combination,
        })

    # Filter out any point combinations with too large of an error
    res_3d = [r for r in res_3d if r['error'] < matching_threshold]

    # Assuming any 3d points to be visible from all 3 views, then maximum number of 3d points
    # is the minimum number of 2d points provided from each views.
    if len(res_3d) > min([len(image_points[c]) for c in CAMERA_IDXS]):
        raise ValueError('Found too many 3D points! Maybe adjust the threshold?')

    # Re-project the final 3d object points back to new 2d image points
    for r in res_3d:
        r['points_2d'] = [
            cams_model[c].project_to_2d(r['pt'])
            for c in CAMERA_IDXS
        ]

    return res_3d
