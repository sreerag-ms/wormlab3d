import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Dataset

# Form blueprint
bp_datasets = Blueprint('datasets', __name__)


@bp_datasets.route('/datasets', methods=['GET'])
def datasets():

    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'datasets'
    os.environ['script_name'] = active
    return render_template(
        'datasets.html',
        title='Datasets',
        active=active,
        ajax_url="/ajax/datasets",
        source_headers=[
            "_id",
            "dataset_type",
            "created",
            "train_test_split_target",
            "train_test_split_actual",
            "size_all",
            "size_train",
            "size_test",
            "restrict_tags",
            "restrict_concs",
            "centre_3d_max_error",
            "exclude_experiments",
            "include_experiments",
            "exclude_trials",
            "include_trials",
        ],
        date_headers=["created"],
        datasets=Dataset.objects.none()   # Let views.api.datasets.ajax_datasets do the query
    )


