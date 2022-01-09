import numpy as np

from app.views.api import bp_api
from wormlab3d.data.model import Trial, Frame


@bp_api.route('/trial/<int:_id>/tracking', methods=['GET'])
def get_tracking_data(_id):
    trial = Trial.objects.get(id=_id)

    centres_3d = []
    timestamps = []
    frame_time = 0.

    pipeline = [
        {'$match': {'trial': trial.id}},
        {'$project': {
            '_id': 0,
            'centre_3d': 1,
            'centre_3d_fixed': 1,
        }},
        {'$sort': {'frame_num': 1}},
    ]
    cursor = Frame.objects().aggregate(pipeline)

    for res in cursor:
        if 'centre_3d_fixed' in res and res['centre_3d_fixed'] is not None:
            pt = res['centre_3d_fixed']['point_3d']
        elif 'centre_3d' in res and res['centre_3d'] is not None:
            pt = res['centre_3d']['point_3d']
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
