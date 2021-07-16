import os

from wormlab3d.data.model import Experiment

from flask import Blueprint, render_template

# Form blueprint
bp_experiments = Blueprint('experiments', __name__)


@bp_experiments.route('/experiments', methods=['GET'])
def experiments():
    active = 'experiments'
    os.environ['script_name'] = active
    return render_template(
        'experiments.html',
        title='Experiments',
        active=active,
        ajax_url="/ajax/experiments",
        source_headers=["_id", "user", "strain", "sex",
                        "age", "concentration", "worm_length", "legacy_id"],
        date_headers=[],
        experiments=Experiment.objects.none()   # Let views.api.experiments.ajax_experiments do the query
    )
