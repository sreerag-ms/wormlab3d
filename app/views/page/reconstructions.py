import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Reconstruction

# Form blueprint
bp_reconstructions = Blueprint('reconstructions', __name__)


@bp_reconstructions.route('/reconstructions', methods=['GET'])
def reconstructions():

    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'reconstructions'
    os.environ['script_name'] = active
    return render_template(
        'reconstructions.html',
        title='Reconstructions',
        active=active,
        ajax_url="/ajax/reconstructions",
        source_headers=["_id", "trial", "trial.experiment", "start_frame", "end_frame",
                        "source", "source_file", "model"],
        date_headers=[],
        reconstructions=Reconstruction.objects.none()   # Let views.api.reconstructions.ajax_reconstructions do the query
    )
