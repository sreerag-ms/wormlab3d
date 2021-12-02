import os

from flask import Flask

# Import api (ajax) blueprints
from .views.api.dataset import bp_api_dataset
from .views.api.experiment import bp_api_experiment
from .views.api.frame import bp_api_frame
from .views.api.midline2d import bp_api_midline2d
from .views.api.midline3d import bp_api_midline3d
from .views.api.reconstruction import bp_api_reconstruction
from .views.api.tag import bp_api_tag
from .views.api.trial import bp_api_trial

# Import page blueprints
from .views.page.datasets import bp_datasets
from .views.page.experiments import bp_experiments
from .views.page.experiment_instance import bp_experiment_instance
from .views.page.index import bp_index
from .views.page.midlines2d import bp_midlines2d
from .views.page.midlines3d import bp_midlines3d
from .views.page.reconstructions import bp_reconstructions
from .views.page.tags import bp_tags
from .views.page.trials import bp_trials
from .views.page.trial_instance import bp_trial_instance

# Import media helper blueprints
from .views.media.transcode import bp_transcode
from .views.media.extract_img import bp_extract_img

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
app.register_blueprint(bp_api_dataset)
app.register_blueprint(bp_api_experiment)
app.register_blueprint(bp_api_frame)
app.register_blueprint(bp_api_midline2d)
app.register_blueprint(bp_api_midline3d)
app.register_blueprint(bp_api_reconstruction)
app.register_blueprint(bp_api_tag)
app.register_blueprint(bp_api_trial)

# Page routes
app.register_blueprint(bp_datasets)
app.register_blueprint(bp_experiments)
app.register_blueprint(bp_experiment_instance)
app.register_blueprint(bp_index)
app.register_blueprint(bp_midlines2d)
app.register_blueprint(bp_midlines3d)
app.register_blueprint(bp_reconstructions)
app.register_blueprint(bp_tags)
app.register_blueprint(bp_trials)
app.register_blueprint(bp_trial_instance)

# Media routes
app.register_blueprint(bp_transcode)
app.register_blueprint(bp_extract_img)
