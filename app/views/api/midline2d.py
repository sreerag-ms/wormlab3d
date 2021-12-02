from wormlab3d.data.model import Midline2D

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_midline2d = Blueprint('api_midline2d', __name__)


@bp_api_midline2d.route('/ajax/midlines2d', methods=['GET'])
def ajax_midlines2d():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Midline2D)
