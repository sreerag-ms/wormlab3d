import json

from wormlab3d.data.model import Frame

from flask import Blueprint

# Form blueprint
bp_api_frame = Blueprint('api_frame', __name__)


@bp_api_frame.route('/ajax/frame', methods=['GET'])
def ajax_frames(trial=None, frame_num=0):

    trial_id = trial if trial else 0

    return Frame.objects(trial=trial_id, frame_num=frame_num).to_json()
