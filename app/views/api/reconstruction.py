from flask import Blueprint, request

from app.util.datatables import dt_query
from app.views.document_view import DocumentView
from wormlab3d.data.model import Reconstruction

# Form blueprint
bp_api_reconstruction = Blueprint('api_reconstruction', __name__)


class ReconstructionsView(DocumentView):
    @property
    def fields(self):
        return [
            {
                'key': '_id',
                'title': 'ID',
                'type': 'id',
            },
            {
                'key': 'experiment',
                'title': 'Experiment',
                'type': 'integer',
            },
            {
                'key': 'trial',
                'title': 'Trial',
                'type': 'integer',
            },
            {
                'key': 'start_frame',
                'title': 'Start frame',
                'type': 'integer',
            },
            {
                'key': 'end_frame',
                'title': 'End frame',
                'type': 'integer',
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
            }
        ]


@bp_api_reconstruction.route('/ajax/reconstructions', methods=['GET'])
def ajax_reconstructions():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Reconstruction)
