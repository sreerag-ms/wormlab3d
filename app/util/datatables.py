"""
datatables.py

Module to handle conversation between DataTables and MongoDB.
"""

import json
import re

from app.util.encoder import JSONEncoder
from app.model import DocumentView
from wormlab3d import logger


def dt_query(request, doc_view: DocumentView):  # TODO: Inheritance type hinting?
    """

    :param request: ImmutableMultiDict
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
    :param doc_view: DocumentView
        A DocumentView. For example, TrialView.

    :return output: str
        A json string containing the queried result and parameters required by DataTables.
    """
    collection_name = doc_view.collection_name

    # Build aggregation pipeline
    pipeline = []
    lookups = {}

    # Projects
    projects = {}
    for key, field_spec in doc_view.fields.items():
        key_as = field_spec['as'] if 'as' in field_spec is not None else key

        if '__' in key:
            rel_keys = key.split('__')
            doc_links = []
            while len(rel_keys) > 1:
                rel_collection = rel_keys[0]
                if rel_collection not in lookups:
                    lookup_key = '.'.join(doc_links + [rel_collection,])
                    lookups[lookup_key] = [
                        {'$lookup': {
                            'from': rel_collection,
                            'localField': '.'.join(doc_links + [rel_collection,]),
                            'foreignField': '_id',
                            'as': f'{lookup_key}_doc'
                        }},
                        {'$unwind': {'path': f'${lookup_key}_doc'}}
                    ]
                doc_links.append(f'{rel_collection}_doc')
                rel_keys = rel_keys[1:]
            projects[key] = f'${".".join(doc_links)}.{rel_keys[0]}'

        elif 'query' in field_spec:
            q = field_spec['query']
            if 'lookup' in q:
                lookup_key = q['lookup']

                if lookup_key not in lookups:
                    lookups[lookup_key] = [
                        {'$lookup': {
                            'from': lookup_key,
                            'localField': '_id',
                            'foreignField': collection_name,
                            'as': lookup_key
                        }}
                    ]

                if q['aggregation'] == 'count':
                    projects[key_as] = {'$size': f'${lookup_key}'}
                elif q['aggregation'] == 'sum':
                    projects[key_as] = {'$sum': f'${lookup_key}.{q["field"]}'}
                else:
                    raise ValueError(f'Unrecognised aggregation value: {q["aggregation"]}.')

            else:
                raise ValueError(f'Unrecognised query: {q}.')

        else:
            projects[key_as] = f'${key}'

    # Filters
    filters = {}
    for i, (key, field_spec) in enumerate(doc_view.fields.items()):
        filter_value = request.get(f'columns[{i}][search][value]')
        if filter_value != '':
            if field_spec['type'] == 'integer':
                filter_value = int(filter_value)
            elif field_spec['type'] == 'float':
                filter_value = float(filter_value)
            elif field_spec['type'] == 'relation':
                filter_value = int(filter_value)
            # elif field_spec['type'] == 'float':
            #     filter_value = float(filter_value)
            filters[key] = filter_value
        # todo: check more lookups
    # print('filters', filters)

    # Add lookups, project then filter
    print('lookups', lookups)
    print('projects', projects)
    for lookup_spec in lookups.values():
        pipeline.extend(lookup_spec)
    pipeline.append({'$project': projects})
    if len(filters):
        pipeline.append({'$match': filters})

    # Sorts
    order_column = re.compile('order(.+?)column')
    order_dir = re.compile('order(.+?)dir')

    # Find the value parameter keys that match strings like 'order[0][column]' and 'order[0][dir]'
    sort_req_idxs = filter(lambda s: order_column.match(s), request)
    sort_req_dirs = filter(lambda s: order_dir.match(s), request)
    sort_keys = [request.get(f'columns[{request.get(k)}][data]') for k in sort_req_idxs]
    sort_dirs = [-1 if request.get(d) == 'desc' else 1 for d in sort_req_dirs]
    assert len(sort_keys) == len(sort_dirs), 'Different numbers of sort keys to sort directions received!'

    sorts = {}
    for k, d in zip(sort_keys, sort_dirs):
        sorts[k] = d
    if len(sorts) > 0:
        pipeline.append({'$sort': sorts})

    # Range of records to retrieve
    start_idx = int(request.get('start'))
    end_idx = start_idx + int(request.get('length'))

    # Return a count of all filtered results plus a slice of full results
    pipeline += [
        {'$group': {
            '_id': None,
            'count': {'$sum': 1},
            'results': {'$push': '$$ROOT'}
        }},
        {'$project': {
            'count': 1,
            'rows': {'$slice': ['$results', start_idx, end_idx]}
        }}
    ]
    logger.debug(pipeline)
    cursor = doc_view.document_class.objects.aggregate(pipeline)
    try:
        results = list(cursor)[0]
    except IndexError:
        results = {
            'rows': [],
            'count': 0
        }


    # Set draw, recordsTotal, recordsFiltered, data in the response json
    # Optionally set error as well if there is an error.
    response = {
        'data': results['rows'],
        'draw': int(request.get('draw')),  # Cast to int to avoid XSS
        'recordsTotal': doc_view.document_class.objects.count(),
        'recordsFiltered': results['count']
    }

    return json.dumps(response, cls=JSONEncoder)
