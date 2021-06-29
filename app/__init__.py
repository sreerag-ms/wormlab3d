import os

from flask import Flask

from wormlab3d import APP_SECRET

WTF_CSRF_ENABLED = False
app = Flask(__name__)

app.config.from_mapping({
    'WTF_CSRF_ENABLED': True,
    'SECRET_KEY': APP_SECRET,
})

os.environ['script_name'] = 'app'

from app import controllers
