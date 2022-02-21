import os

from flask import Blueprint, render_template

from app.model.dataset import DatasetView

# Form blueprint
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

    return render_template(
        'item/dataset.html',
        title=f'Dataset #{_id}',
        active=active,
        dataset=dataset,
        dataset_view=dataset_view,
    )
