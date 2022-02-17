import os
import re
import subprocess

from flask import Response, jsonify, request, abort
from mongoengine import DoesNotExist

import app.util.config as config
from app.views.media import bp_media
from wormlab3d import CAMERA_IDXS, TRACKING_VIDEOS_PATH
from wormlab3d.data.model import Trial, Reconstruction
from wormlab3d.data.util import fix_path


@bp_media.route('/stream/<int:_id>/<int:cam_idx>')
def stream_trial_video(_id: int, cam_idx: int):
    """
    Stream a trial video.
    """
    path = _get_video_path(_id, cam_idx)
    start = request.args.get('start') or 0
    return _generate_stream_response(path, start)


@bp_media.route('/duration/<int:_id>/<int:cam_idx>')
def get_trial_video_duration(_id: int, cam_idx: int):
    """
    Get the duration of a trial video.
    """
    path = _get_video_path(_id, cam_idx)
    return _get_media_duration(path)


@bp_media.route('/tracking-videos/<int:_id>')
def stream_tracking_videos(_id: int):
    """
    Stream the tracking video-triplet.
    """
    trial = Trial.objects.get(id=_id)
    start = request.args.get('start') or 0
    path = str(TRACKING_VIDEOS_PATH / f'{trial.id:03d}.mp4')
    return _generate_stream_response(path, start)


@bp_media.route('/tracking-videos-duration/<int:_id>')
def get_tracking_videos_duration(_id: int):
    """
    Get the duration of the tracking video-triplet.
    """
    trial = Trial.objects.get(id=_id)
    path = str(TRACKING_VIDEOS_PATH / f'{trial.id:03d}.mp4')
    return _get_media_duration(path)


@bp_media.route('/reconstruction-video/<string:_id>')
def stream_reconstruction_video(_id: str):
    """
    Stream the reconstruction video.
    """
    reconstruction = Reconstruction.objects.get(id=_id)
    start = request.args.get('start') or 0
    path = str(reconstruction.video_filename)
    return _generate_stream_response(path, start)


@bp_media.route('/reconstruction-video-duration/<string:_id>')
def get_reconstruction_video_duration(_id: str):
    """
    Get the duration of the reconstruction video.
    """
    reconstruction = Reconstruction.objects.get(id=_id)
    path = str(reconstruction.video_filename)
    return _get_media_duration(path)


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


def _transcode_media(path: str, start: float):
    """
    Retrieve a media file at <path> and transcode it to mp4.
    Code taken from https://github.com/derolf/transcoder
    """
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


def _generate_stream_response(path: str, start: float):
    """
    Generate a streamed media response.
    """
    return Response(response=_transcode_media(path, start), status=200, mimetype='video/mp4',
                    headers={'Access-Control-Allow-Origin': '*',
                             'Content-Type': 'video/mp4', 'Content-Disposition': 'inline',
                             'Content-Transfer-Enconding': 'binary'})


def _get_media_duration(path: str):
    """
    Figure out duration of a media file using FFmpeg.
    Code taken from https://github.com/derolf/transcoder
    """
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
