import os

from flask import Blueprint, render_template

from app.model.dataset import DatasetView

# Form blueprint
bp_datasets = Blueprint('datasets', __name__, url_prefix='/dataset')


@bp_datasets.route('/', methods=['GET'])
def datasets():
    active = 'dataset'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Datasets',
        active=active,
        doc_view=DatasetView(),
        table_order=[2, 'desc']
    )
