"""
datatables.py

Module to handle conversation between DataTables and MongoDB.
"""

import re
import json

from mongoengine import Document


def dt_query(dt_params, document: Document):   # TODO: Inheritance type hinting?
    """

    :param dt_params: ImmutableMultiDict
        Direct DataTables to a flask route, then pass in request.args from flask.

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
    :param document: Document
        A MongoEngine Document. For example, the Trial table.

    :return output: str
        A json string containing the queried result and parameters required by DataTables.
    """

    # Id range of records to retrieve
    start_id = int(dt_params.get("start"))
    end_id = start_id + int(dt_params.get("length"))

    # Compile regex for sorting
    order_column = re.compile("order(.+?)column")
    order_dir = re.compile("order(.+?)dir")

    # Find the value parameter keys that match strings like "order[0][column]" and "order[0][dir]"
    sort_cols = filter(lambda s: order_column.match(s), dt_params)
    sort_dirs = filter(lambda s: order_dir.match(s), dt_params)

    sort_list = []
    sort_col = next(sort_cols, None)
    sort_dir = next(sort_dirs, None)
    while sort_col is not None:
        sort_val = dt_params.get(sort_col)
        sort_dir = dt_params.get(sort_dir)  # asc, desc, None

        # Find the name of the column to be requested from the parameters (the key looks like "columns[0][data]")
        sort_col_name = dt_params.get(f"columns[{sort_val}][data]")

        sort_symbol = "-" if sort_dir == "desc" else "+"

        # Form the column sorting string and append it to the list
        sort_list.append(f"{sort_symbol}{sort_col_name}")

        # Go to next sort column
        sort_col = next(sort_cols, None)
        sort_dir = next(sort_dirs, None)

    # Form the query set
    ordered_queryset = document.objects.order_by(*sort_list)
    # filtered_queryset = ordered_queryset.filter()   # TODO: Implement
    num_matching_records = ordered_queryset.count()
    queryset = ordered_queryset[start_id:end_id]

    # Set draw, recordsTotal, recordsFilered, data in the response json
    # Optionally set error as well if there is an error.
    response = {"data": json.loads(queryset.to_json()),
                "draw": int(dt_params.get("draw")),  # Cast to int to avoid XSS
                "recordsTotal": document.objects.count(),
                "recordsFiltered": num_matching_records
                }

    return json.dumps(response)
