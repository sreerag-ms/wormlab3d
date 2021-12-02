"""Config file."""
from wormlab3d import ANNEX_PATH

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


# Transcoding
ffmpeg_transcode_args = {
    "*": " -ss {} -i {} -f {} -vcodec {} ",
    "mp3": " -f mp3 -codex copy "
}

ffmpeg_screenshot = {
    "*": " -ss {} -i {} -filter:v {} -frames:v 1 {} "   # ffmpeg -ss 1 -i 100_0.avi -frames:v 1 screenshot.jpg
}

ffmpeg_common_options = {
    "*": "-strict experimental -preset ultrafast -movflags frag_keyframe+empty_moov+faststart pipe:1",
    "mp3": "pipe:1"
}

ffmpeg_poster_args = ["-f", "mjpeg", "-vf", "scale=512x512", "pipe:1"]
# "-noaccurate_seek"
