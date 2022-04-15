import os

from app.model import ReconstructionView, TrialView, FrameView
from app.views.api import ExperimentView
from flask import Blueprint, render_template
from wormlab3d.data.model import Trial

bp_trials = Blueprint('trials', __name__, url_prefix='/trial')


@bp_trials.route('/', methods=['GET'])
def trials():
    active = 'trial'
    os.environ['script_name'] = active
    created_row_cb = 'function(row, data, dataIndex) {$(row).addClass(\'quality-\' + String(data[\'quality\']));}'

    return render_template(
        'list_view.html',
        title='Trials',
        active=active,
        doc_view=TrialView(
            hide_fields=['experiment___id', 'experiment__legacy_id', 'experiment__num_trials', 'experiment__num_frames']
        ),
        created_row_cb=created_row_cb
    )


@bp_trials.route('/<int:_id>', methods=['GET'])
def trial_instance(_id):
    active = 'trial'
    os.environ['script_name'] = active
    trial = Trial.objects.get(id=_id)
    trial_view = TrialView(
        hide_fields=['_id', 'experiment*', 'quality'],
    )
    experiment_view = ExperimentView(
        hide_fields=['num_frames', 'legacy_id']
    )
    frame_view = FrameView(
        hide_fields=['trial*'],
        field_values={'trial': _id}
    )
    reconstruction_view = ReconstructionView(
        hide_fields=[
            'trial*',
            'mf_parameters___id',
            'mf_parameters__created',
            'mf_parameters__use_master',
            'mf_parameters__sigmas_init',
            'mf_parameters__n_steps*',
            'mf_parameters__conv*',
            'mf_parameters__algorithm',
            'mf_parameters__lr*'
        ],
        field_values={'trial': _id}
    )
    quality_check_fields = {
        'fps': 'FPS',
        'durations': 'Durations',
        'brightnesses': 'Brightnesses',
        'triangulations': 'Triangulations',
        'triangulations_fixed': 'Triangulations fixed',
        'tracking_video': 'Tracking video exists',
        'syncing': 'Syncing',
        'crop_size': 'Crop size',
        'verified': 'Manually verified',
    }
    quality_check_toggleable_fields = ['syncing', 'crop_size', 'verified']

    return render_template(
        'item/trial.html',
        title=f'Trial #{_id}',
        active=active,
        trial=trial,
        trial_view=trial_view,
        experiment_view=experiment_view,
        frame_view=frame_view,
        reconstruction_view=reconstruction_view,
        quality_check_fields=quality_check_fields,
        quality_check_toggleable_fields=quality_check_toggleable_fields
    )
