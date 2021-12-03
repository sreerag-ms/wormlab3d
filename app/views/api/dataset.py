from flask import Blueprint, request

from app.util.datatables import dt_query
from app.views.document_view import DocumentView
from wormlab3d.data.model import Dataset

# Form blueprint
bp_api_dataset = Blueprint('api_dataset', __name__)


class DatasetsView(DocumentView):
    @property
    def fields(self):
        return [
            {
                'key': '_id',
                'title': 'ID',
                'type': 'id',
            },
            {
                'key': 'dataset_type',
                'title': 'Type',
                'type': 'string',
            },
            {
                'key': 'created',
                'title': 'Created',
                'type': 'date',
            },
            {
                'key': 'train_test_split_target',
                'title': 'Split Target',
                'type': 'float',
                'precision': 2,
            },
            {
                'key': 'train_test_split_actual',
                'title': 'Split Actual',
                'type': 'float',
                'precision': 2,
            },
            {
                'key': 'size_all',
                'title': 'Size All',
                'type': 'integer',
            },
            {
                'key': 'size_train',
                'title': 'Size Train',
                'type': 'integer',
            },
            {
                'key': 'size_test',
                'title': 'Size Test',
                'type': 'integer',
            },
            {
                'key': 'restrict_tags',
                'title': 'Restrict Tags',
                'type': 'string',
            },
            {
                'key': 'restrict_concs',
                'title': 'Restrict Concs.',
                'type': 'string',
            },
            {
                'key': 'centre_3d_max_error',
                'title': 'Max Centre Error',
                'type': 'string',
            },
            {
                'key': 'exclude_experiments',
                'title': 'Exclude Experiments',
                'type': 'string',
            },
            {
                'key': 'include_experiments',
                'title': 'Include Experiments',
                'type': 'string',
            },
            {
                'key': 'exclude_trials',
                'title': 'Exclude Trials',
                'type': 'string',
            },
            {
                'key': 'include_trials',
                'title': 'Include Trials',
                'type': 'string',
            },
        ]


@bp_api_dataset.route('/ajax/datasets', methods=['GET'])
def ajax_datasets():
    '''
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    '''

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Dataset)
