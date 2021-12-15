import os

from flask import Blueprint, render_template

from app.views.api import ExperimentView
from app.model.trial import TrialView
from wormlab3d.data.model import Trial

bp_trials = Blueprint('trials', __name__, url_prefix='/trial')


@bp_trials.route('/', methods=['GET'])
def trials():
    active = 'trial'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Trials',
        active=active,
        doc_view=TrialView(
            hide_fields=['experiment___id', 'experiment__legacy_id', 'experiment__num_trials', 'experiment__num_frames']
        ),
    )


@bp_trials.route('/<int:_id>', methods=['GET'])
def trial_instance(_id):
    active = 'trial'
    os.environ['script_name'] = active
    trial = Trial.objects.get(id=_id)
    trial_view = TrialView(
        hide_fields=['_id', 'experiment*'],
        field_values={'experiment': _id}
    )
    experiment_view = ExperimentView(
        hide_fields=['num_frames', 'legacy_id']
    )

    return render_template(
        'item/trial.html',
        title=f'Trial #{_id}',
        active=active,
        trial=trial,
        trial_view=trial_view,
        experiment_view=experiment_view,
    )
