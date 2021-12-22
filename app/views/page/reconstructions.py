import os

from flask import Blueprint, render_template

from app.model import ExperimentView, TrialView, MFParametersView
from app.model.reconstruction import ReconstructionView
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF

bp_reconstructions = Blueprint('reconstructions', __name__, url_prefix='/reconstruction')


@bp_reconstructions.route('/', methods=['GET'])
def reconstructions():
    active = 'reconstruction'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Reconstructions',
        active=active,
        doc_view=ReconstructionView(
            hide_fields=[
                'trial___id',
                'trial__legacy_id',
                'trial__trial_num',
                'trial__date',
                'trial__num_frames',
                'trial__duration',
                'trial__comments',
                'trial__num_reconstructions',
                'trial__num_reconstructions',
                'trial__experiment___id',
                'trial__experiment__legacy_id',
                'trial__experiment__worm_length',
                'trial__experiment__num_trials',
                'trial__experiment__num_frames',
            ]
        ),
    )


@bp_reconstructions.route('/<string:_id>', methods=['GET'])
def reconstruction_instance(_id):
    active = 'reconstruction'
    os.environ['script_name'] = active
    reconstruction = Reconstruction.objects.get(id=_id)

    hide_reconstruction_fields =['_id', 'trial*', 'mf_parameters*']
    if reconstruction.source == M3D_SOURCE_MF:
        hide_reconstruction_fields.append('source_file')

    reconstruction_view = ReconstructionView(
        hide_fields=hide_reconstruction_fields,
    )
    experiment_view = ExperimentView(
        hide_fields=['num_frames', 'legacy_id']
    )
    trial_view = TrialView(
        hide_fields=['num_frames', 'legacy_id', 'num_reconstructions', 'experiment*']
    )
    parameters_view = MFParametersView()

    return render_template(
        'item/reconstruction.html',
        title=f'Reconstruction #{_id}',
        active=active,
        reconstruction=reconstruction,
        reconstruction_view=reconstruction_view,
        trial_view=trial_view,
        experiment_view=experiment_view,
        parameters_view=parameters_view
    )
