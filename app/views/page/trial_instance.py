import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Trial

# Form blueprint
bp_trial_instance = Blueprint('trial_instance', __name__)


@bp_trial_instance.route('/trials/<int:_id>', methods=['GET'])
def trial_instance(_id):

    active = 'trial_instance'
    os.environ['script_name'] = active

    trial = Trial.objects(id=_id)[0]

    return render_template(
        'trial_instance.html',
        title=f"Trial #{_id}",
        active=active,
        _id=_id,
        video_id=f"{trial.legacy_id:03d}",
        trial=trial
    )
