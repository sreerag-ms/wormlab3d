import os

from flask import Blueprint, render_template

from app.model.reconstruction import ReconstructionView

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
                # 'experiment*',
                # 'experiment___id',
                # 'experiment__legacy_id',
                # 'experiment__worm_length',
                # 'experiment__num_trials',
                # 'experiment__num_frames',
                'trial___id',
                'trial__legacy_id',
                'trial__trial_num',
                'trial__date',
                'trial__num_frames',
                'trial__comments',
                'trial__experiment___id',
                'trial__experiment__legacy_id',
                'trial__experiment__worm_length',
                'trial__experiment__num_trials',
                'trial__experiment__num_frames',
            ]
        ),
    )
