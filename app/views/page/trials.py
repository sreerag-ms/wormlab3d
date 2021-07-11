import os

from flask import Blueprint, render_template

from wormlab3d.data.model import Trial

bp_trials = Blueprint('trials', __name__)


@bp_trials.route('/trials', methods=['GET'])
def trials():
    active = 'trials'
    os.environ['script_name'] = active
    return render_template(
        'trials.html',
        title='Trials',
        active=active,
        trials=Trial.objects
    )
