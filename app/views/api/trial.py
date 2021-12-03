from flask import Blueprint, request

from app.util.datatables import *
from app.views.document_view import DocumentView
from wormlab3d.data.model import Trial

# Form blueprint
bp_api_trial = Blueprint('api_trial', __name__)


class TrialsView(DocumentView):
    has_item_view = True

    @property
    def fields(self):
        return [
            {
                'key': '_id',
                'title': 'ID',
                'type': 'integer',
            },
            {
                'key': 'experiment',
                'title': 'Experiment',
                'type': 'integer',
            },
            {
                'key': 'date',
                'title': 'Date',
                'type': 'date',
            },
            {
                'key': 'trial_num',
                'title': 'Trial num.',
                'type': 'integer',
            },
            {
                'key': 'num_frames',
                'title': 'Num. frames',
                'type': 'integer',
            },
            {
                'key': 'fps',
                'title': 'FPS',
                'type': 'float',
                'precision': 2,
            },
            {
                'key': 'temperature',
                'title': 'Temperature',
                'type': 'float',
                'precision': 2,
            },
            {
                'key': 'comments',
                'title': 'Comments',
                'type': 'string',
            },
        ]


@bp_api_trial.route('/ajax/trials', methods=['GET'])
def ajax_trials():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Trial)
