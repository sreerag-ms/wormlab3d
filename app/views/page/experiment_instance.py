import os

from flask import Blueprint, render_template

from app.util.data_model import *
from wormlab3d.data.model import Experiment

# Form blueprint
bp_experiment_instance = Blueprint('experiment_instance', __name__)


@bp_experiment_instance.route('/experiments/<int:_id>', methods=['GET'])
def experiment_instance(_id):
    active = 'experiment_instance'
    os.environ['script_name'] = active

    attrs = attr_names(Experiment,
                       exclude_underscore=True,
                       excludes=[])

    exp = Experiment.objects(id=_id)[0]

    return render_template(
        'experiment_instance.html',
        title=f"Experiment #{_id}",
        active=active,
        _id=_id,
        video_id=f"{exp.legacy_id:03d}",
        attrs=attrs,
        trial=exp
    )
