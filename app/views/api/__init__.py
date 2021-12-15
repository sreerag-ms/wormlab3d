from flask import Blueprint, request

from app.util.datatables import *
from app.views.api.experiment import ExperimentView
from app.views.api.trial import TrialView

# API blueprint
bp_api = Blueprint('api', __name__, url_prefix='/api')

collections = {
    'experiment': ExperimentView,
    'trial': TrialView,
}


@bp_api.route('/<string:collection_name>', methods=['GET'])
def get_table_data(collection_name):
    assert collection_name in collections
    return dt_query(request.args, collections[collection_name]())
