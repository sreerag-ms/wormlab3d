import logging
import os
import sys
import time
from pathlib import Path

import dotenv

# Get running environment
ENV = os.getenv('ENV', 'local')

# Set base path to point to the repository root
ROOT_PATH = str(Path(__file__).parent.parent)

# Load environment variables from .env file
dotenv.load_dotenv(ROOT_PATH + '/.env')

# Data paths
DATA_PATH = ROOT_PATH + '/data'
ANNEX_PATH = os.getenv('ANNEX_PATH', str(Path(__file__).parent.parent.parent) + '/worm_data')
WT3D_PATH = os.getenv('WT3D_PATH', str(Path(__file__).parent.parent.parent) + '/3DWT_Data')
DATASET_CACHE_PATH = DATA_PATH + '/ds_cache'

# When fetching annexed files on demand, ensure that this much space is always kept free
MIN_FREE_DISK_SPACE = os.getenv('MIN_FREE_DISK_SPACE', '100G')

# Size of prepared images, changing this will break lots of things :)
PREPARED_IMAGE_SIZE = (200, 200)

# Camera indices, for the avoidance of doubt
CAMERA_IDXS = [0, 1, 2]

# Worm length in terms of number of coordinates/sections
# N_WORM_POINTS = 128
N_WORM_POINTS = 50
# N_WORM_POINTS = 3


# || ------------------------------ DATABASE ------------------------------- ||

DB_NAME = os.getenv('DB_NAME', 'wormlab3d')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = int(os.getenv('DB_PORT', 27017))
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# || -------------------------------- LOGS --------------------------------- ||

cwd = os.getcwd()
dir_name = os.path.dirname(sys.argv[0]).replace(cwd, '').lstrip('/')
SCRIPT_PATH = (cwd + '/' + dir_name).rstrip('/')
LOGS_PATH = ROOT_PATH + '/logs' + SCRIPT_PATH[len(ROOT_PATH):] + '/' + os.path.basename(sys.argv[0]).rstrip('.py')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
WRITE_LOG_FILES = os.getenv('WRITE_LOG_FILES', False)

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

# || ---------------------------------------------------------------------- ||
