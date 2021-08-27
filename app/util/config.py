"""Config file."""
from wormlab3d import ANNEX_PATH

port = 8124  # what is this port for?
media_folder = ANNEX_PATH + '/video'

# transcoder config
ffmpeg = "ffmpeg"

types = {
    "mp3": "audio",
    "jpg": "image",
    "mp4": "video"}


transcode_mime = {
    "*": "video/mp4",
    "mp3": "audio/mp3",
    "jpg": "image/jpg"}


ffmpeg_transcode_args_2 = {
    "*": ["-f", "mp4", "-strict", "experimental", "-preset", "ultrafast", "-movflags", "frag_keyframe+empty_moov+faststart", "pipe:1"],
    "mp3": ["-f", "mp3", "-codec", "copy", "pipe:1"]}

ffmpeg_transcode_args = {
    "*": " -ss {} -i {} -f {} -vcodec {} -strict experimental -preset ultrafast -movflags frag_keyframe+empty_moov+faststart pipe:1",
    "mp3": ["-f", "mp3", "-codec", "copy", "pipe:1"]}

ffmpeg_poster_args = ["-f", "mjpeg", "-vf", "scale=512x512", "pipe:1"]
# "-noaccurate_seek"
