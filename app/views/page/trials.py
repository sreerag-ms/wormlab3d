import os

from flask import Blueprint, render_template, flash

from app.views.api.trial import TrialsView

# Form blueprint
bp_trials = Blueprint('trials', __name__)


@bp_trials.route('/trials', methods=['GET'])
def trials():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'trials'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Trials',
        active=active,
        doc_view=TrialsView(),
        ajax_url='/ajax/trials',
    )
