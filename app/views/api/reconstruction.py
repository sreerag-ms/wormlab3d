import base64
from io import BytesIO
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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


@bp_api.route('/reconstruction/<string:_id>/point-stats', methods=['GET'])
def get_point_stats(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    D = reconstruction.mf_parameters.depth
    frame_num = request.args.get('frame_num', type=int)
    ts = TrialState(reconstruction=reconstruction)

    vals_flat = {
        k: ts.get(k)[frame_num]
        for k in ['sigmas', 'intensities', 'scores']
    }
    vals = {
        k: {}
        for k in ['xs', 'sigmas', 'intensities', 'scores']
    }

    for d in range(D):
        from_idx = sum([2**d2 for d2 in range(d)])
        to_idx = from_idx + 2**d
        vals['xs'][d] = np.linspace(0, 1, 2**d + 2)[1:-1].tolist()
        for k in ['sigmas', 'intensities', 'scores']:
            vals[k][d] = vals_flat[k][from_idx:to_idx].tolist()

    return {
        'timestamps': _get_timestamps(reconstruction),
        **vals
    }


@bp_api.route('/reconstruction/<string:_id>/posture', methods=['GET'])
def get_posture(_id):
    reconstruction = Reconstruction.objects.get(id=_id)
    D = request.args.get('depth', type=int)
    frame_num = request.args.get('frame_num', type=int)
    ts = TrialState(reconstruction=reconstruction)
    from_idx = sum([2**d2 for d2 in range(D)])
    to_idx = from_idx + 2**D

    # Get 3D posture
    all_points = ts.get('points')[frame_num]
    points = all_points[from_idx:to_idx].T

    # Get 2D projections
    all_projections = ts.get('points_2d')[frame_num]  # (N, 3, 2)
    points_2d = np.round(all_projections[from_idx:to_idx]).astype(np.int32)

    # Get sigmas
    all_sigmas = ts.get('sigmas')[frame_num]
    sigmas = all_sigmas[from_idx:to_idx]

    # Get intensities
    all_intensities = ts.get('intensities')[frame_num]
    intensities = all_intensities[from_idx:to_idx]

    # Colour map
    cmap = plt.get_cmap('jet')
    colours = np.array([cmap(d) for d in np.linspace(0, 1, 2**D)])

    # Add transparency based on intensity
    colours[:, 3] *= intensities / 2

    # Convert to integers
    colours = np.round(colours * 255).astype(np.uint8)

    # Read each numpy array into a PIL Image, write the png image to a buffer and encode as a base64-encoded string
    images = []
    frame = reconstruction.get_frame(frame_num)
    for i, img_array in enumerate(frame.images):
        z = (img_array * 255).astype(np.uint8)
        z = cv2.cvtColor(z, cv2.COLOR_GRAY2RGBA)

        # Overlay 2d projection
        p2d = points_2d[:, i]
        for j, p in enumerate(p2d):
            z = cv2.circle(z, p, int(sigmas[j] * 100), color=colours[j].tolist(), thickness=-1, lineType=cv2.LINE_AA)

        img = Image.fromarray(z, 'RGBA')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        data_uri = base64.b64encode(buffer.read()).decode('utf-8')
        images.append(f'data:image/png;charset=utf-8;base64,{data_uri}')

    response = {
        'images': images,
        'posture': points.tolist()
    }

    return response


@bp_api.route('/reconstruction/<string:_id>/worm-lengths', methods=['GET'])
def get_worm_lengths(_id: str):
    reconstruction = Reconstruction.objects.get(id=_id)
    D = reconstruction.mf_parameters.depth
    ts = TrialState(reconstruction=reconstruction)
    points = ts.get('points')  # (T, N, 3)
    lengths = {}

    for d in range(1, D):
        from_idx = sum([2**d2 for d2 in range(d)])
        to_idx = from_idx + 2**d
        points_d = points[:, from_idx:to_idx]
        segments = points_d[:, 1:] - points_d[:, :-1]
        segment_lengths = np.linalg.norm(segments, axis=-1)
        lengths[d] = np.sum(segment_lengths, axis=-1).tolist()

    return {
        'timestamps': _get_timestamps(reconstruction),
        'lengths': lengths
    }


def _get_timestamps(reconstruction: Reconstruction, frame_nums: List[int] = None):
    if frame_nums is None:
        frame_nums = np.arange(reconstruction.start_frame, reconstruction.end_frame)
    timestamps = [n / reconstruction.trial.fps for n in frame_nums]
    return timestamps
