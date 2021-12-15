import os

from flask import Blueprint, render_template, flash

from app.model.reconstruction import ReconstructionsView

# Form blueprint
bp_reconstructions = Blueprint('reconstructions', __name__)


@bp_reconstructions.route('/reconstructions', methods=['GET'])
def reconstructions():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'reconstructions'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Reconstructions',
        active=active,
        doc_view=ReconstructionsView(),
        ajax_url="/ajax/reconstructions",
    )
