import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Midline3D

# Form blueprint
bp_midlines3d = Blueprint('midlines3d', __name__)


@bp_midlines3d.route('/midlines3D', methods=['GET'])
def midlines3d():

    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'midlines3D'
    os.environ['script_name'] = active
    return render_template(
        'midlines3d.html',
        title='Midlines 3D',
        active=active,
        ajax_url="/ajax/midlines3d",
        source_headers=["_id", "frame", "source", "source_file", "model", "error"],
        date_headers=[],
        midline3ds=Midline3D.objects.none()   # Let views.api.midline3ds.ajax_midline3ds do the query
    )
