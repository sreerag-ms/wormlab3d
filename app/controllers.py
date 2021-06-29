import os

from flask import render_template, flash

from app import app
from wormlab3d import ENV, ROOT_PATH
from wormlab3d.data.model import Experiment, Trial


@app.route('/', methods=['GET'])
def index():
    active = 'index'
    os.environ['script_name'] = active
    code = 200

    # Check database connection
    try:
        from mongoengine import get_connection
        from pymongo import MongoClient
        conn = get_connection()
        assert isinstance(conn, MongoClient)
        flash('Database Connected!', category='success')
    except Exception as e:
        flash('No database connection!\n' + str(e), category='error')

    # Check if .env file exists
    if not os.path.exists(ROOT_PATH + '/.env'):
        flash('.env file does not exist!', category='error')

    # Get logger level
    try:
        from wormlab3d import logger
        import logging
        log_level = logging._levelToName.get(logger.getEffectiveLevel())
        flash('Logger level: ' + log_level + '(' + str(logger.getEffectiveLevel()) + ')', category='success')
    except Exception as e:
        flash('Couldn\'t load logger:\n' + str(e), category='error')

    # Display env vars in local env only
    if ENV == 'local':
        env_vars = os.environ.items()
    else:
        env_vars = []

    return render_template(
        'index.html',
        title='Index',
        active=active,
        env_vars=env_vars,
    ), code


@app.route('/experiments', methods=['GET'])
def experiments():
    active = 'experiments'
    os.environ['script_name'] = active
    return render_template(
        'experiments.html',
        title='Experiments',
        active=active,
        experiments=Experiment.objects
    )


@app.route('/trials', methods=['GET'])
def trials():
    active = 'trials'
    os.environ['script_name'] = active
    return render_template(
        'trials.html',
        title='Trials',
        active=active,
        trials=Trial.objects
    )
