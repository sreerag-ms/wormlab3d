from typing import List

import numpy as np
from flask import request

from app.views.api import bp_api
from app.views.page.reconstructions import CAM_PARAMETER_KEYS
from wormlab3d.data.model import Reconstruction
from wormlab3d.midlines3d.trial_state import TrialState
from wormlab3d.trajectories.cache import get_trajectory


@bp_api.route('/reconstruction/<string:_id>/trajectory', methods=['GET'])
def get_trajectory_data(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    X, meta = get_trajectory(
        reconstruction_id=_id,
        depth=-1,
        trajectory_point=-1,
    )
    response = {
        'timestamps': _get_timestamps(reconstruction, meta['frame_nums']),
        'X': X.T.tolist(),
    }
    return response


@bp_api.route('/reconstruction/<string:_id>/stats', methods=['GET'])
def get_stats(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    ts = TrialState(reconstruction=reconstruction)
    key = request.args.get('key')
    assert key in ts.stats
    return {
        'timestamps': _get_timestamps(reconstruction),
        'values': ts.stats[key],
    }


@bp_api.route('/reconstruction/<string:_id>/cameras', methods=['GET'])
def get_cameras_parameters(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    ts = TrialState(reconstruction=reconstruction)

    key = request.args.get('key')
    assert key in CAM_PARAMETER_KEYS

    if key == 'Intrinsics':
        p = ts.get('cam_intrinsics')  # (T, 3, 4)
        titles = ['fx', 'fy', 'cx', 'cy']

    elif key == 'Rotations':
        p = ts.get('cam_rotation_preangles')  # (T, 3, 3, 2)
        p = np.arctan2(p[:, :, :, 0], p[:, :, :, 1])
        titles = ['theta', 'phi', 'psi']

    elif key == 'Translations':
        p = ts.get('cam_translations')  # (T, 3, 3)
        titles = ['x', 'y', 'z']

    elif key == 'Distortions':
        p = ts.get('cam_distortions')  # (T, 3, 5)
        titles = ['k1', 'k2', 'p1', 'p2', 'k3']

    elif key == 'Shifts':
        p = ts.get('cam_shifts')  # (T, 3, 1)
        titles = ['ds']

    p = p.transpose(2, 1, 0)  # (4, 3, T)

    return {
        'timestamps': _get_timestamps(reconstruction),
        'data': p.tolist(),
        'shape': tuple(p.shape),
        'titles': titles
    }


def _get_timestamps(reconstruction: Reconstruction, frame_nums: List[int] = None):
    if frame_nums is None:
        frame_nums = np.arange(reconstruction.start_frame, reconstruction.end_frame)
    timestamps = [n / reconstruction.trial.fps for n in frame_nums]
    return timestamps
