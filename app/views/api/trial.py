import numpy as np

from app.views.api import bp_api
from flask import request
from wormlab3d.data.model import Trial, Frame


@bp_api.route('/trial/<int:_id>/tracking', methods=['GET'])
def get_tracking_data(_id):
    trial = Trial.objects.get(id=_id)
    fixed = bool(request.args.get('fixed', type=int))
    centres_3d = []
    timestamps = []
    frame_time = 0.

    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$project': {
            '_id': 0,
            'p3d': '$centre_3d' + ('_fixed' if fixed else ''),
        }},
        {'$sort': {'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)

    for res in cursor:
        if 'p3d' in res and res['p3d'] is not None:
            pt = res['p3d']['point_3d']
        else:
            pt = np.array([None, None, None])
        centres_3d.append(pt)
        timestamps.append(frame_time)
        frame_time += 1 / trial.fps
    centres_3d = np.stack(centres_3d).T

    response = {
        'timestamps': timestamps,
        'centres_3d': centres_3d.tolist()
    }

    return response
