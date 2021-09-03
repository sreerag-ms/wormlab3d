import os
import subprocess

from flask import Blueprint, Response, jsonify, request, abort, send_file

import app.util.config as config

from wormlab3d.data.model import Frame

# Form blueprint
bp_extract_img = Blueprint('extract_img', __name__)


@bp_extract_img.route('/media/extract_img/<path:path>')
def screenshot(path):
    """
    Screenshot a single frame from the supplied video path.

    ffmpeg -ss 1 -i 100_0.avi -frames:v 1 screenshot.jpg
    """

    # Get absolute path to video
    d = os.path.abspath(os.path.join(config.media_folder, path))
    if not os.path.isfile(d):
        abort(404)

    # Input params
    trial_num = int(request.args.get("trial_num") or -1)
    frame_num = int(request.args.get("frame_num") or 0)
    fps = request.args.get("fps") or 25   # TODO: confirm unspecified FPS=25 (seems to be that way w/ 100 and 120)
    # Get cam_num from the video path name if cam_num isn't supplied
    cam_num = int(request.args.get("cam_num") or path[4])

    # Output params
    filename = f"{path.split('.')[0]}-{frame_num}.jpg"
    output = os.path.abspath(os.path.join(config.media_folder, filename))   # TODO: require a path to a local cache folder instead of media_folder (because videos are stored on git annex, probably can't write to that)

    # Raise error if trial_num isn't specified. TODO: test this actually works
    if trial_num == -1:
        return Response(response="trial_num must be specified!", status=404, mimetype='text',
                        headers={'Access-Control-Allow-Origin': '*',
                                 "Content-Type": "video/mp4", "Content-Disposition": "inline",
                                 "Content-Transfer-Enconding": "binary"})

    # If file exists (perhaps created previously), fetch from the cache folder
    if os.path.isfile(output):
        return send_file(output, mimetype="image/jpeg")

    # Convert frame number to seconds - this is dependent on the FPS of the trial video
    start = frame_num * 1/fps

    # Get frame object from database
    frame = Frame.objects(trial=trial_num, frame_num=frame_num).first()
    if frame is None:
        raise Exception(f"Trial {trial_num} has no data for Frame {frame_num}")

    # Crop video at 2d coordinates (not reprojections). The output is a 200x200 square jpg image
    coords = frame.centres_2d[cam_num][0]   # TODO: some data have more than one bubble. centre_3d.source_point_idxs?
    crop_option = f"crop=200:200:{coords[0]-100}:{coords[1]-100}"

    # Command to extract the image w/ ffmpeg
    args = config.ffmpeg_screenshot["*"]

    cmdline = config.ffmpeg + args.format(str(start), d, crop_option, output)
    proc = subprocess.run(cmdline.split())

    return send_file(output, mimetype="image/jpeg")
