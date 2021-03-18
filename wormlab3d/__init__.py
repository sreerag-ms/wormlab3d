import logging
import os
import sys

LOGS_DIR = 'logs'
LOG_LEVEL = 'debug'

# Set formatting
logging.basicConfig(
    format='[%(levelname)s %(asctime)s]: %(message)s',
)

# Create a logger with the name corresponding to the script being executed
script_name = os.path.basename(sys.argv[0])[:-3]
logger = logging.getLogger(script_name)
logger.setLevel(LOG_LEVEL)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Setup handlers
print('log path', f'{LOGS_DIR}/{script_name}.log')
os.makedirs(LOGS_DIR, exist_ok=True)
logger.addHandler(logging.FileHandler(f'{LOGS_DIR}/{script_name}.log', mode='a'))

# Don't propagate logs to the root logger as this causes duplicate entries
logger.propagate = False


# Handle uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical(
        'Uncaught exception',
        exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception
