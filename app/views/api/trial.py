from wormlab3d.data.model import Trial

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_trial = Blueprint('api_trial', __name__)


@bp_api_trial.route('/ajax/trials', methods=['GET'])
def ajax_trials():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Trial)
