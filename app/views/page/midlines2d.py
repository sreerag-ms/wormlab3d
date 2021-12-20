import os

from flask import Blueprint, render_template, flash

from app.model.midline2d import Midlines2dView

# Form blueprint
bp_midlines2d = Blueprint('midlines2d', __name__)


@bp_midlines2d.route('/midlines2D', methods=['GET'])
def midlines2d():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'midlines2d'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Midlines 2D',
        active=active,
        doc_view=Midlines2dView(),
        ajax_url="/ajax/midlines2d",
    )
