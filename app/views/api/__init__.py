from flask import Blueprint, request

from app.model import ExperimentView, FrameView, MFParametersView, TrialView, ReconstructionView
from app.util.datatables import dt_query

# API blueprint
bp_api = Blueprint('api', __name__, url_prefix='/api')

collections = {
    'experiment': ExperimentView,
    'frame': FrameView,
    'mf_parameters': MFParametersView,
    'reconstruction': ReconstructionView,
    'trial': TrialView
}


@bp_api.route('/<string:collection_name>', methods=['GET'])
def get_table_data(collection_name):
    assert collection_name in collections
    return dt_query(request.args, collections[collection_name]())


import app.views.api.eigenworms
import app.views.api.frame
import app.views.api.reconstruction
import app.views.api.trial
