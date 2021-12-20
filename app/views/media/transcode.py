import os
import re
import subprocess

from flask import Response, jsonify, request, abort
from mongoengine import DoesNotExist

import app.util.config as config
from app.views.media import bp_media
from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model import Trial
from wormlab3d.data.util import fix_path


def _get_video_path(_id: int, cam_idx: int) -> str:
    try:
        trial = Trial.objects.get(id=_id)
    except DoesNotExist:
        abort(404)
    if cam_idx not in CAMERA_IDXS:
        abort(404)
    path = fix_path(trial.videos[cam_idx])
    if not os.path.isfile(path):
        abort(404)
    return path


@bp_media.route('/stream/<int:_id>/<int:cam_idx>')
def media_content(_id: int, cam_idx: int):
    """
    Retrieve a media file at <path> and transcode it to mp4.

    Code taken from https://github.com/derolf/transcoder
    """
    path = _get_video_path(_id, cam_idx)
    start = request.args.get('start') or 0

    def generate():
        args = config.ffmpeg_transcode_args['*']
        common_options = config.ffmpeg_common_options['*']

        cmdline = config.ffmpeg + args.format(str(start), path, 'mp4', 'copy') + common_options
        FNULL = open(os.devnull, 'w')
        proc = subprocess.Popen(cmdline.split(), stdout=subprocess.PIPE, stderr=FNULL)
        try:
            f = proc.stdout
            byte = f.read(512)
            while byte:
                yield byte
                byte = f.read(512)
        finally:
            proc.kill()

    return Response(response=generate(), status=200, mimetype='video/mp4',
                    headers={'Access-Control-Allow-Origin': '*',
                             'Content-Type': 'video/mp4', 'Content-Disposition': 'inline',
                             'Content-Transfer-Enconding': 'binary'})


@bp_media.route('/duration/<int:_id>/<int:cam_idx>')
def get_duration(_id: int, cam_idx: int):
    """
    Figure out duration of a media file using FFmpeg.

    Code taken from https://github.com/derolf/transcoder
    """
    path = _get_video_path(_id, cam_idx)
    cmdline = f'{config.ffmpeg} -i {path}'
    duration = -1
    FNULL = open(os.devnull, 'w')
    proc = subprocess.Popen(cmdline.split(), stderr=subprocess.PIPE, stdout=FNULL)
    try:
        for line in proc.stderr:
            line = str(line)
            line = line.rstrip()
            # Duration: 00:00:45.13, start: 0.000000, bitrate: 302 kb/s
            m = re.search('Duration: (..):(..):(..)\...', line)
            if m is not None:
                duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + 1
    finally:
        proc.kill()

    return jsonify(duration=duration)
