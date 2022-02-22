import os

from flask import Blueprint, render_template

from app.model import DatasetView, ReconstructionView
from wormlab3d.data.model import Dataset

bp_datasets = Blueprint('datasets', __name__, url_prefix='/dataset')


@bp_datasets.route('/', methods=['GET'])
def datasets():
    active = 'dataset'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Datasets',
        active=active,
        doc_view=DatasetView(
            hide_fields=[
                'include_experiments',
                'exclude_experiments',
                'include_trials',
                'exclude_trials',
            ]
        ),
        table_order=[2, 'desc']
    )


@bp_datasets.route('/<string:_id>', methods=['GET'])
def dataset_instance(_id: str):
    active = 'dataset'
    os.environ['script_name'] = active
    dataset = Dataset.objects.get(id=_id)

    dataset_view = DatasetView(
        hide_fields=['_id']
    )

    reconstruction_view = ReconstructionView(
        hide_fields=[
            'trial___id',
            'trial__experiment___id',
            'trial__experiment__legacy_id',
            'trial__experiment__num_trials',
            'trial__experiment__num_frames',
            'trial__trial_num',
            'trial__n_frames_min',
            'trial__duration',
            'trial__fps',
            'trial__temperature',
            'trial__legacy_id',
            'trial__comments',
            'trial__num_reconstructions',
            'trial__quality',
            'mf_parameters___id',
            'mf_parameters__created',
            'mf_parameters__use_master',
            'mf_parameters__sigmas_init',
            'mf_parameters__n_steps*',
            'mf_parameters__conv*',
            'mf_parameters__algorithm',
            'mf_parameters__lr*'
        ],
        field_values={'_id': '|'.join(str(r.id) for r in dataset.reconstructions)}
    )

    return render_template(
        'item/dataset.html',
        title=f'Dataset #{_id}',
        active=active,
        dataset=dataset,
        dataset_view=dataset_view,
        reconstruction_view=reconstruction_view,
    )
