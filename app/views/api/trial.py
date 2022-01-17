from app.views.api import bp_api
from flask import request
from wormlab3d.data.model import Trial


@bp_api.route('/trial/<int:_id>/tracking', methods=['GET'])
def get_tracking_data(_id):
    trial = Trial.objects.get(id=_id)
    fixed = bool(request.args.get('fixed', type=int))
    centres_3d, timestamps = trial.get_tracking_data(fixed)

    response = {
        'timestamps': timestamps.tolist(),
        'centres_3d': centres_3d.T.tolist()
    }

    return response
