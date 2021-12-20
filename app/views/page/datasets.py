import os

from flask import Blueprint, render_template, flash

from app.model.dataset import DatasetsView

# Form blueprint
bp_datasets = Blueprint('datasets', __name__)


@bp_datasets.route('/datasets', methods=['GET'])
def datasets():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'datasets'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Datasets',
        active=active,
        doc_view=DatasetsView(),
        ajax_url="/ajax/datasets",
    )
