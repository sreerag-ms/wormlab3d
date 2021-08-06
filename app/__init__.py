import os

from flask import Flask

# Import blueprints
from .views.api.experiments import bp_api_experiments
from .views.api.trials import bp_api_trials
from .views.page.index import bp_index
from .views.page.experiments import bp_experiments
from .views.page.experiment_instance import bp_experiment_instance
from .views.page.trials import bp_trials
from .views.page.trial_instance import bp_trial_instance
from .views.media.transcode import bp_transcode

from wormlab3d import APP_SECRET

WTF_CSRF_ENABLED = False
app = Flask(__name__)

app.config.from_mapping({
    'WTF_CSRF_ENABLED': True,
    'SECRET_KEY': APP_SECRET,
})

os.environ['script_name'] = 'app'


# Register blueprints on the app (can specify subdomain)
app.register_blueprint(bp_api_experiments)
app.register_blueprint(bp_api_trials)

app.register_blueprint(bp_index)
app.register_blueprint(bp_experiments)
app.register_blueprint(bp_experiment_instance)
app.register_blueprint(bp_trials)
app.register_blueprint(bp_trial_instance)

app.register_blueprint(bp_transcode)
