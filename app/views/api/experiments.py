import json

from wormlab3d.data.model import Experiment

from app.util.datatables import *

from flask import Blueprint, request

# Register blueprint
bp_api_experiments = Blueprint('api_experiments', __name__)


@bp_api_experiments.route('/ajax/experiments', methods=['GET'])
def ajax_experiments():
    """
    :return: queried data as a json formatted string
    """

    # Parameters are sent via the URL from DataTables. Query accordingly
    q_params = mongo_query_params(request.args)

    # Form the query set
    ordered_queryset = Experiment.objects.order_by(*q_params["sort_list"])
    # filtered_queryset = ordered_queryset.filter()   # TODO: Implement
    num_matching_records = ordered_queryset.count()
    queryset = ordered_queryset[q_params["start_id"]:q_params["end_id"]]

    # Set draw, recordsTotal, recordsFilered, data in the response json
    # Optionally set error as well if there is an error.
    response = {"data": json.loads(queryset.to_json()),
                "draw": int(request.args.get("draw")),   # Cast to int to avoid XSS
                "recordsTotal": Experiment.objects.count(),
                "recordsFilered": num_matching_records
                }

    return json.dumps(response)
