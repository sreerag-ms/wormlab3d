from flask import Blueprint, request

from app.util.datatables import *
from app.views.document_view import DocumentView
from wormlab3d.data.model import Tag

# Form blueprint
bp_api_tag = Blueprint('api_tag', __name__)


class TagsView(DocumentView):
    @property
    def fields(self):
        return [
            {
                'key': '_id',
                'title': 'ID',
                'type': 'integer',
            },
            {
                'key': 'name',
                'title': 'Name',
                'type': 'string',
            },
            {
                'key': 'short_name',
                'title': 'Short name',
                'type': 'string',
            },
            {
                'key': 'symbol',
                'title': 'Symbol',
                'type': 'string',
            },
            {
                'key': 'description',
                'title': 'Description',
                'type': 'string',
            },
        ]


@bp_api_tag.route('/ajax/tags', methods=['GET'])
def ajax_tags():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Tag)
