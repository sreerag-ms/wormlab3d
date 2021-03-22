import logging
import os
import sys
import time
from pathlib import Path

# todo: config from env vars
ROOT_PATH = str(Path(__file__).parent.parent)
SCRIPT_PATH = os.path.dirname(sys.argv[0])
LOGS_PATH = ROOT_PATH + '/logs' + SCRIPT_PATH[len(ROOT_PATH):]
LOG_LEVEL = 'DEBUG'
WRITE_LOG_FILES = False

# Data paths
DATA_PATH = ROOT_PATH + '/data'
ANNEX_PATH = str(Path(__file__).parent.parent.parent) + '/worm_data'
WT3D_PATH = str(Path(__file__).parent.parent.parent) + '/3DWT_Data'

# Set formatting
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

# Create a logger with the name corresponding to the script being executed
script_name = os.path.basename(sys.argv[0])[:-3]
logger = logging.getLogger(script_name)
logger.setLevel(LOG_LEVEL)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Setup handlers
if WRITE_LOG_FILES:
    LOG_FILENAME = f'{script_name}_{time.strftime("%Y-%m-%d_%H%M%S")}.log'
    print(f'Writing logs to: {LOGS_PATH}/{LOG_FILENAME}')
    os.makedirs(LOGS_PATH, exist_ok=True)
    file_handler = logging.FileHandler(f'{LOGS_PATH}/{LOG_FILENAME}', mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Don't propagate logs to the root logger as this causes duplicate entries
logger.propagate = False

# Handle uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    logger.critical(
        'Uncaught exception',
        exc_info=(exc_type, exc_value, exc_traceback)
    )


sys.excepthook = handle_exception
