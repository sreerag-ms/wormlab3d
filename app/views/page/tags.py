import os

from flask import Blueprint, render_template, flash

from app.model.tag import TagsView

# Form blueprint
bp_tags = Blueprint('tags', __name__)


@bp_tags.route('/tags', methods=['GET'])
def tags():
    # Display a tip on how to do multi-column sorting
    flash("To sort by multiple columns, hold SHIFT then left click.", category="info")

    active = 'tags'
    os.environ['script_name'] = active
    return render_template(
        'list_view.html',
        title='Tags',
        active=active,
        doc_view=TagsView(),
        ajax_url="/ajax/tags",
    )
