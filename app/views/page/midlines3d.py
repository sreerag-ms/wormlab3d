import os

from flask import Blueprint, render_template, flash

from app.views.api.midline3d import Midlines3dView

# Form blueprint
bp_midlines3d = Blueprint('midlines3d', __name__)


@bp_midlines3d.route('/midlines3D', methods=['GET'])
def midlines3d():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'midlines3d'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Midlines 3D',
        active=active,
        doc_view=Midlines3dView(),
        ajax_url="/ajax/midlines3d",
    )
