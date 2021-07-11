import os

from flask import Blueprint, render_template

from wormlab3d.data.model import Experiment

bp_experiments = Blueprint('experiments', __name__)


@bp_experiments.route('/experiments', methods=['GET'])
def experiments():
    active = 'experiments'
    os.environ['script_name'] = active
    return render_template(
        'experiments.html',
        title='Experiments',
        active=active,
        experiments=Experiment.objects
    )
