from wormlab3d.data.model import Experiment

from flask import Blueprint, request

bp_api_experiments = Blueprint('api_experiments', __name__)


@bp_api_experiments.route('/ajax/experiments', methods=['GET'])
def ajax_experiments():
    # Parameters are sent via the URL from DataTables. Query accordingly

    # Draw counter (the number of Ajax request). Cast to int to avoid XSS
    draw = int(request.args.get("draw"))
    print(draw)

    # Return data expects these parameters to be set: draw, recordsTotal, recordsFilered, data, error


    # Serialize the QuerySet into JSON, let DataTables present it
    return Experiment.objects().to_json()
