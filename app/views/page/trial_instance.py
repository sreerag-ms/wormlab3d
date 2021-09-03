import os

from flask import Blueprint, render_template

from wormlab3d.data.model import Trial

from app.util.data_model import *

# Form blueprint
bp_trial_instance = Blueprint('trial_instance', __name__)


@bp_trial_instance.route('/trials/<int:_id>', methods=['GET'])
def trial_instance(_id):

    active = 'trial_instance'
    os.environ['script_name'] = active

    attrs = attr_names(Trial,
                       exclude_underscore=True,
                       excludes=["id", "pk", "legacy_id", "legacy_data",         # ID and legacy
                                 "n_frames_min", "n_frames_max", "num_frames",   # Aggregated
                                 "backgrounds", "videos",                        # File paths
                                 "experiment", "date", "comments"                # Needs more formatting
                                 ])

    trial = Trial.objects(id=_id)[0]

    filenames = [f"{trial.legacy_id:03d}_{i}.avi" for i in range(3)]

    return render_template(
        'trial_instance.html',
        title=f"Trial #{_id}",
        active=active,
        _id=_id,
        video_id=f"{trial.legacy_id:03d}",
        filenames=filenames,
        attrs=attrs,
        trial=trial
    )
