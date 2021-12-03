from flask import Blueprint, request

from app.util.datatables import *
from app.views.document_view import DocumentView
from wormlab3d.data.model import Midline3D

# Form blueprint
bp_api_midline3d = Blueprint('api_midline3d', __name__)


class Midlines3dView(DocumentView):
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
                'key': 'source',
                'title': 'Source',
                'type': 'string',
            },
            {
                'key': 'source_file',
                'title': 'Source file',
                'type': 'string',
            },
            {
                'key': 'model',
                'title': 'Model',
                'type': 'id',
            },
            {
                'key': 'error',
                'title': 'Error',
                'type': 'float',
                'precision': 5,
            }
        ]


@bp_api_midline3d.route('/ajax/midlines3d', methods=['GET'])
def ajax_midlines3d():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Midline3D)
