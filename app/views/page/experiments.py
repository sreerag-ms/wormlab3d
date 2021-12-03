import os

from flask import Blueprint, render_template, flash

from app.views.api.experiment import ExperimentsView

# Form blueprint
bp_experiments = Blueprint('experiments', __name__)


@bp_experiments.route('/experiments', methods=['GET'])
def experiments():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'experiments'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Experiments',
        active=active,
        doc_view=ExperimentsView(),
        ajax_url="/ajax/experiments",
    )
