import re
import json

from wormlab3d.data.model import Experiment

from flask import Blueprint, request
from pprint import pprint

bp_api_experiments = Blueprint('api_experiments', __name__)


@bp_api_experiments.route('/ajax/experiments', methods=['GET'])
def ajax_experiments():
    """

    Sample parameters dict (not actual data)
    ----------------------------------------
    {'_': '1374248696',
     'columns[0][data]': '_id',
     'columns[0][name]': '',
     'columns[0][orderable]': 'true',
     'columns[0][search][regex]': 'false',
     'columns[0][search][value]': '',
     'columns[0][searchable]': 'true',
     'columns[1][data]': 'user',
     'columns[1][name]': '',
     'columns[1][orderable]': 'true',
     'columns[1][search][regex]': 'false',
     'columns[1][search][value]': '',
     'columns[1][searchable]': 'true',
     'draw': '2',
     'length': '100',
     'order[0][column]': '0',
     'order[0][dir]': 'asc',
     'search[regex]': 'false',
     'search[value]': '',
     'start': '0'}

    :return: queried data as a json formatted string
    """

    # Parameters are sent via the URL from DataTables. Query accordingly

    # Id range of records to retrieve
    start_id = int(request.args.get("start"))
    end_id = start_id + int(request.args.get("length"))

    # Compile regex for sorting
    order_column = re.compile("order(.+?)column")
    order_dir = re.compile("order(.+?)dir")

    # Find the value parameter keys that match strings like "order[0][column]" and "order[0][dir]"
    sort_cols = filter(lambda s: order_column.match(s), request.args)
    sort_dirs = filter(lambda s: order_dir.match(s), request.args)

    sort_list = []
    sort_col = next(sort_cols, None)
    sort_dir = next(sort_dirs, None)
    while sort_col is not None:
        sort_val = request.args.get(sort_col)
        sort_dir = request.args.get(sort_dir)  # asc, desc, None

        # Find the name of the column to be requested from the parameters (the key looks like "columns[0][data]")
        sort_col_name = request.args.get(f"columns[{sort_val}][data]")

        sort_symbol = "-" if sort_dir == "desc" else "+"

        # Form the column sorting string and append it to the list
        sort_list.append(f"{sort_symbol}{sort_col_name}")

        # Go to next sort column
        sort_col = next(sort_cols, None)
        sort_dir = next(sort_dirs, None)

    # Form the query set
    queryset = Experiment.objects.order_by(*sort_list)   # TODO: Change this to do filter
    num_matching_records = queryset.count()
    queryset = Experiment.objects.order_by(*sort_list)[start_id: end_id]

    # Set draw, recordsTotal, recordsFilered, data in the response json
    # Optionally set error as well if there is an error.
    response = {"data": json.loads(queryset.to_json()),
                "draw": int(request.args.get("draw")),   # Cast to int to avoid XSS
                "recordsTotal": Experiment.objects.count(),
                "recordsFilered": num_matching_records
                }

    return json.dumps(response)
