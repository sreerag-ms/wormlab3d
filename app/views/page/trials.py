import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Trial

# Form blueprint
bp_trials = Blueprint('trials', __name__)


@bp_trials.route('/trials', methods=['GET'])
def trials():

    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

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
