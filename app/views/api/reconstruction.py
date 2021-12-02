from wormlab3d.data.model import Reconstruction

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_reconstruction = Blueprint('api_reconstruction', __name__)


@bp_api_reconstruction.route('/ajax/reconstructions', methods=['GET'])
def ajax_reconstructions():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Reconstruction)
