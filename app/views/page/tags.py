import os

from flask import Blueprint, render_template, flash

from wormlab3d.data.model import Tag

# Form blueprint
bp_tags = Blueprint('tags', __name__)


@bp_tags.route('/tags', methods=['GET'])
def tags():

    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'tags'
    os.environ['script_name'] = active
    return render_template(
        'tags.html',
        title='Tags',
        active=active,
        ajax_url="/ajax/tags",
        source_headers=["_id", "name", "short_name", "symbol","description"],
        date_headers=[],
        tags=Tag.objects.none()   # Let views.api.tags.ajax_tags do the query
    )
