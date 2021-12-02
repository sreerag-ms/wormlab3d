import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Midline2D

# Form blueprint
bp_midlines2d = Blueprint('midlines2d', __name__)


@bp_midlines2d.route('/midlines2D', methods=['GET'])
def midlines2d():

    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'midlines2D'
    os.environ['script_name'] = active
    return render_template(
        'midlines2d.html',
        title='Midlines 2D',
        active=active,
        ajax_url="/ajax/midlines2d",
        source_headers=["_id", "frame", "camera", "user", "model"],
        date_headers=[],
        midline2ds=Midline2D.objects.none()   # Let views.api.midline2ds.ajax_midline2ds do the query
    )
