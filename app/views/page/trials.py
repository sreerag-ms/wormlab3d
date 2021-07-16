import os

from flask import Blueprint, render_template

from wormlab3d.data.model import Trial

# Form blueprint
bp_trials = Blueprint('trials', __name__)


@bp_trials.route('/trials', methods=['GET'])
def trials():
    active = 'trials'
    os.environ['script_name'] = active
    return render_template(
        'trials.html',
        title='Trials',
        active=active,
        ajax_url="/ajax/trials",
        source_headers=["_id", "experiment", "date", "trial_num",
                        "num_frames", "fps", "temperature", "comments"],
        date_headers=["date"],
        trials=Trial.objects.none()   # Let views.api.trials.ajax_trials do the query
    )
