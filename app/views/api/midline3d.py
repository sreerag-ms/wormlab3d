from wormlab3d.data.model import Midline3D

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_midline3d = Blueprint('api_midline3d', __name__)


@bp_api_midline3d.route('/ajax/midlines3d', methods=['GET'])
def ajax_midlines3d():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Midline3D)
