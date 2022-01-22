from flask import Blueprint

bp_media = Blueprint('media', __name__, url_prefix='/media')

import app.views.media.transcode
