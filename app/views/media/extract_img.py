import os
import subprocess
import time

from flask import Blueprint, Response, jsonify, request, abort, send_file

import app.util.config as config

from wormlab3d.data.model import Frame

# Form blueprint
bp_extract_img = Blueprint('extract_img', __name__)


class ScreenshotArgs:
    """
    Cast (potentially) user-supplied arguments and form parameters.
    Default values can be changed here.
    """

    def __init__(self, path):
        # Get absolute path to video
        self.dir = os.path.abspath(os.path.join(config.media_folder, path))

        # Input params
        self.trial_num = int(request.args.get("trial_num") or -1)
        self.frame_num = int(request.args.get("frame_num") or 0)
        self.fps = int(request.args.get("fps") or 25)  # TODO: confirm unspecified FPS=25 (seems to be that way w/ 100 and 120)
        # Get cam_num from the video path name if cam_num isn't supplied
        self.cam_num = int(request.args.get("cam_num") or (path[4] if len(path) >= 5 else 0))

        # Extra input params for checking whether the frame images exist
        self.timeout_sec = int(request.args.get("timeout_ms") or 5000) / 1000
        self.refresh_sec = int(request.args.get("refresh_ms") or 50) / 1000   # Periodically refresh the check every [refresh_sec]

        # Convert frame number to seconds - this is dependent on the FPS of the trial video
        self.start = self.frame_num * 1/self.fps

        # Output params
        self.filename = f"{path.split('.')[0]}-{self.frame_num}.jpg"
        self.output = os.path.abspath(os.path.join(config.media_folder, self.filename))  # TODO: require a path to a local cache folder instead of media_folder (because videos are stored on git annex, probably can't write to that)


@bp_extract_img.route('/media/extract_img/<path:path>')
def screenshot(path):
    """
    Screenshot a single frame from the supplied video path.

    ffmpeg -ss 1 -i 100_0.avi -frames:v 1 screenshot.jpg
    """
    args = ScreenshotArgs(path)
    if not os.path.isfile(args.dir):
        abort(404)

    # Raise error if trial_num isn't specified.
    if args.trial_num == -1:
        return Response(response="trial_num must be specified!",
                        status=404, mimetype='text',
                        headers={'Access-Control-Allow-Origin': '*',
                                 "Content-Type": "video/mp4",
                                 "Content-Disposition": "inline",
                                 "Content-Transfer-Enconding": "binary"})

    # If file exists (perhaps created previously), fetch from the cache folder
    if os.path.isfile(args.output):
        return send_file(args.output, mimetype="image/jpeg")

    # Get frame object from database
    frame = Frame.objects(trial=args.trial_num, frame_num=args.frame_num).first()
    if frame is None:
        raise Exception(f"Trial {args.trial_num} has no data for Frame {args.frame_num}")

    # Crop video at 2d coordinates (not reprojections). The output is a 200x200 square jpg image
    coords = frame.centres_2d[args.cam_num][0]   # TODO: some data have more than one bubble. centre_3d.source_point_idxs?
    crop_option = f"crop=200:200:{coords[0]-100}:{coords[1]-100}"

    # Command to extract the image w/ ffmpeg
    ffmpeg_args = config.ffmpeg_screenshot["*"]

    cmdline = config.ffmpeg + ffmpeg_args.format(str(args.start), args.dir, crop_option, args.output)
    proc = subprocess.run(cmdline.split())

    return send_file(args.output, mimetype="image/jpeg")


@bp_extract_img.route('/media/extract_img/completion/<int:video_id>')
def completion(video_id):
    """
    Check if all 3 camera videos at [frame_num] have been screenshotted.

    :return:
        True if the screenshots exist before the timeout period,
        False otherwise.
    """
    args = ScreenshotArgs("")

    # Keep checking whether the frame images exist every [refresh_ms]
    # Terminate when [timeout_ms} has passed
    time_elapsed = 0
    while time_elapsed < args.timeout_sec:
        filenames = [f"{video_id}_{i}-{args.frame_num}.jpg" for i in range(3)]
        outputs = [os.path.abspath(os.path.join(config.media_folder, filename)) for filename in filenames]
        if not all([os.path.isfile(output) for output in outputs]):
            time.sleep(args.refresh_sec)
            time_elapsed += args.refresh_sec
        else:
            break

    return str(time_elapsed < args.timeout_sec)
