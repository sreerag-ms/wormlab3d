from flask import Blueprint, request

from app.util.datatables import *
from app.views.document_view import DocumentView
from wormlab3d.data.model import Midline2D

# Form blueprint
bp_api_midline2d = Blueprint('api_midline2d', __name__)


class Midlines2dView(DocumentView):
    @property
    def fields(self):
        return [
            {
                'key': '_id',
                'title': 'ID',
                'type': 'id',
            },
            {
                'key': 'frame',
                'title': 'Frame',
                'type': 'id',
            },
            {
                'key': 'camera',
                'title': 'Camera',
                'type': 'integer',
            },
            {
                'key': 'user',
                'title': 'User',
                'type': 'string',
            },
            {
                'key': 'model',
                'title': 'Model',
                'type': 'id',
            }
        ]


@bp_api_midline2d.route('/ajax/midlines2d', methods=['GET'])
def ajax_midlines2d():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Midline2D)
