import os

from flask import Flask

# Import api (ajax) blueprints
from .views.api import bp_api

# Import page blueprints
from .views.page.datasets import bp_datasets
from .views.page.experiments import bp_experiments
from .views.page.index import bp_index
from .views.page.midlines2d import bp_midlines2d
from .views.page.midlines3d import bp_midlines3d
from .views.page.reconstructions import bp_reconstructions
from .views.page.tags import bp_tags
from .views.page.trials import bp_trials

# Import media helper blueprints
from .views.media import bp_media

from wormlab3d import APP_SECRET

WTF_CSRF_ENABLED = False
app = Flask(__name__)

app.config.from_mapping({
    'WTF_CSRF_ENABLED': True,
    'SECRET_KEY': APP_SECRET,
})

os.environ['script_name'] = 'app'


# Register blueprints on the app (can specify subdomain)

# API (ajax) routes
app.register_blueprint(bp_api)

# Page routes
app.register_blueprint(bp_datasets)
app.register_blueprint(bp_experiments)
app.register_blueprint(bp_index)
app.register_blueprint(bp_midlines2d)
app.register_blueprint(bp_midlines3d)
app.register_blueprint(bp_reconstructions)
app.register_blueprint(bp_tags)
app.register_blueprint(bp_trials)

# Media routes
app.register_blueprint(bp_media)
