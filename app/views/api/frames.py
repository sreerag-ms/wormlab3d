import json

from wormlab3d.data.model import Frame

from flask import Blueprint

# Form blueprint
bp_api_frames = Blueprint('api_frames', __name__)


@bp_api_frames.route('/ajax/frames', methods=['GET'])
def ajax_frames(trial=None, frame_num=0):

    trial_num = trial if trial else 0

    return Frame.objects(trial=trial_num, frame_num=frame_num).to_json()
