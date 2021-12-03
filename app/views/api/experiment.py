from flask import Blueprint, request

from app.util.datatables import *
from app.views.document_view import DocumentView
from wormlab3d.data.model import Experiment

# Form blueprint
bp_api_experiment = Blueprint('api_experiment', __name__)


class ExperimentsView(DocumentView):
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
                'key': 'user',
                'title': 'User',
                'type': 'string',
            },
            {
                'key': 'strain',
                'title': 'Strain',
                'type': 'string',
            },
            {
                'key': 'sex',
                'title': 'Sex',
                'type': 'string',
            },
            {
                'key': 'age',
                'title': 'Age',
                'type': 'string',
            },
            {
                'key': 'concentration',
                'title': 'Conc.',
                'type': 'float',
                'precision': 2
            },
            {
                'key': 'worm_length',
                'title': 'Worm length',
                'type': 'float',
                'precision': 2
            },
            {
                'key': 'legacy_id',
                'title': 'Legacy id',
                'type': 'integer',
            }
        ]


@bp_api_experiment.route('/ajax/experiments', methods=['GET'])
def ajax_experiments():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Experiment)
