from wormlab3d.data.model import Tag

from app.util.datatables import *

from flask import Blueprint, request

# Form blueprint
bp_api_tag = Blueprint('api_tag', __name__)


@bp_api_tag.route('/ajax/tags', methods=['GET'])
def ajax_tags():
    """
    :return: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Parameters are sent via the URL from DataTables,
    # which can be obtained with request.args.
    return dt_query(request.args, Tag)
