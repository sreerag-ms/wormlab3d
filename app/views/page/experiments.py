import os

from flask import Blueprint, render_template

from app.model.experiment import ExperimentView
from app.model.trial import TrialView
from wormlab3d.data.model import Experiment

bp_experiments = Blueprint('experiments', __name__, url_prefix='/experiment')


@bp_experiments.route('/', methods=['GET'])
def experiments():
    active = 'experiment'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Experiments',
        active=active,
        doc_view=ExperimentView()
    )


@bp_experiments.route('/<int:_id>', methods=['GET'])
def experiment_instance(_id):
    active = 'experiment'
    os.environ['script_name'] = active
    experiment = Experiment.objects.get(id=_id)

    experiment_view = ExperimentView(
        hide_fields=['_id']
    )

    trial_view = TrialView(
        hide_fields=['experiment*'],
        field_values={'experiment': _id}
    )

    created_trial_row_cb = 'function(row, data, dataIndex) {$(row).addClass(\'quality-\' + String(data[\'quality\']));}'

    return render_template(
        'item/experiment.html',
        title=f'Experiment #{_id}',
        active=active,
        experiment=experiment,
        experiment_view=experiment_view,
        trial_view=trial_view,
        created_trial_row_cb=created_trial_row_cb
    )
