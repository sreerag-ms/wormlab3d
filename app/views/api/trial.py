from flask import request

from app.views.api import bp_api
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


@bp_api.route('/trial/<int:_id>/set-comments', methods=['POST'])
def set_comments(_id):
    trial = Trial.objects.get(id=_id)
    comments = request.values.get('new_comments', type=str)
    trial.comments = comments
    trial.save()
    response = {'status': 1}

    return response


@bp_api.route('/trial/<int:_id>/adjust-crop-size', methods=['POST'])
def adjust_crop_size(_id):
    trial = Trial.objects.get(id=_id)
    new_size = request.values.get('new_size', type=int)
    assert 150 <= new_size <= 400, 'New size outside of sensible region!'
    trial.crop_size = new_size
    trial.save()
    response = {'status': 1}

    return response


@bp_api.route('/trial/<int:_id>/quality-toggle', methods=['POST'])
def quality_toggle(_id):
    trial = Trial.objects.get(id=_id)
    key = request.values.get('key', type=str)
    assert key in trial.quality_checks, f'Unrecognised key: {key}.'
    new_status = False if trial.quality_checks[key] else True
    trial.quality_checks[key] = new_status
    trial.save()
    response = {'status': new_status}

    return response
