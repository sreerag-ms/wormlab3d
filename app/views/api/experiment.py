from wormlab3d.data.model import Experiment

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_experiment = Blueprint('api_experiment', __name__)


@bp_api_experiment.route('/ajax/experiments', methods=['GET'])
def ajax_experiments():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Experiment)
