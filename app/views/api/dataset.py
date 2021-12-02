from wormlab3d.data.model import Dataset

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_dataset = Blueprint('api_dataset', __name__)


@bp_api_dataset.route('/ajax/datasets', methods=['GET'])
def ajax_datasets():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Dataset)
