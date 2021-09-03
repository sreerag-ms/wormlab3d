import os
import re
import subprocess

from flask import Blueprint, Response, jsonify, request, abort

import app.util.config as config

# Form blueprint
bp_transcode = Blueprint('transcode', __name__)


@bp_transcode.route('/media/<path:path>')
def media_content(path):
    """
    Retrieve a media file at <path> and transcode it to mp4.

    Code taken from https://github.com/derolf/transcoder
    """

    d = os.path.abspath(os.path.join(config.media_folder, path))
    if not os.path.isfile(d):
        abort(404)
    start = request.args.get("start") or 0

    def generate():
        args = config.ffmpeg_transcode_args["*"]
        common_options = config.ffmpeg_common_options["*"]

        cmdline = config.ffmpeg + args.format(str(start), d, "mp4", "copy") + common_options
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
                             "Content-Type": "video/mp4", "Content-Disposition": "inline",
                             "Content-Transfer-Enconding": "binary"})


@bp_transcode.route('/media/duration/<path:path>')
def get_duration(path):
    """
    Figure out duration of a media file using FFmpeg.

    Code taken from https://github.com/derolf/transcoder
    """
    d = os.path.abspath(os.path.join(config.media_folder, path))
    if not os.path.isfile(d):
        abort(404)

    cmdline = f"{config.ffmpeg} -i {d}"

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
