import logging
import os
import sys
import time
from pathlib import Path

import dotenv

# Get running environment
ENV = os.getenv('ENV', 'local')

# Set base path to point to the repository root
ROOT_PATH = Path(__file__).parent.parent

# Load environment variables from .env file
dotenv.load_dotenv(ROOT_PATH / '.env')


def _load_env_path(k: str, default: Path):
    ep = os.getenv(k)
    if ep is None:
        ep = default
    if ep is not None:
        ep = Path(ep)
    return ep


# Data paths
DATA_PATH = ROOT_PATH / 'data'
ANNEX_PATH = _load_env_path('ANNEX_PATH', ROOT_PATH.parent / 'worm_data')
WT3D_PATH = _load_env_path('WT3D_PATH', ROOT_PATH.parent / '3DWT_Data')
DATASETS_PATH = _load_env_path('DATASETS_PATH', DATA_PATH / 'datasets')
DATASETS_MIDLINES3D_PATH = _load_env_path('DATASETS_MIDLINES3D_PATH', DATASETS_PATH / 'midlines3d')
DATASETS_SEG_MASKS_PATH = _load_env_path('DATASET_CACHE_PATH', DATASETS_PATH / 'seg_masks')
DATASETS_EIGENTRACES_PATH = _load_env_path('DATASET_CACHE_PATH', DATASETS_PATH / 'eigentraces')
MF_DATA_PATH = _load_env_path('MF_DATA_PATH', DATA_PATH / 'MF_outputs')
EIGENWORMS_PATH = _load_env_path('EIGENWORMS_PATH', DATA_PATH / 'eigenworms')
PREPARED_IMAGES_PATH = _load_env_path('PREPARED_IMAGES_PATH', DATA_PATH / 'prepared_images')
TRACKING_VIDEOS_PATH = _load_env_path('TRACKING_VIDEOS_PATH', DATA_PATH / 'tracking_videos')
TRAJECTORY_CACHE_PATH = _load_env_path('TRAJECTORY_CACHE_PATH', DATA_PATH / 'trajectory_cache')
POSTURE_DISTANCES_CACHE_PATH = _load_env_path('POSTURE_DISTANCES_CACHE_PATH', DATA_PATH / 'posture_distances_cache')
POSTURE_CLUSTERS_CACHE_PATH = _load_env_path('POSTURE_CLUSTERS_CACHE_PATH', DATA_PATH / 'posture_clusters_cache')
RECONSTRUCTION_VIDEOS_PATH = _load_env_path('RECONSTRUCTION_VIDEOS_PATH', DATA_PATH / 'reconstruction_videos')
PCA_CACHE_PATH = _load_env_path('PCA_CACHE_PATH', DATA_PATH / 'pca_cache')

# When fetching annexed files on demand, ensure that this much space is always kept free
MIN_FREE_DISK_SPACE = os.getenv('MIN_FREE_DISK_SPACE', '100G')

# Default size for cropped, prepared images
PREPARED_IMAGE_SIZE_DEFAULT = 200

# Camera indices, for the avoidance of doubt
CAMERA_IDXS = [0, 1, 2]

# Worm length in terms of number of coordinates/sections. todo: remove
# N_WORM_POINTS = 128
N_WORM_POINTS = 50
# N_WORM_POINTS = 3

# Timestamp for when the script was started, can be used for log names
START_TIMESTAMP = time.strftime('%Y%m%d_%H%M')

# Number of parallel workers to use for tasks
N_WORKERS = int(os.getenv('N_WORKERS', 8))

# PyTorch to use JIT where possible
PYTORCH_JIT = os.getenv('PYTORCH_JIT', 1)

# || ------------------------------ DATABASE ------------------------------- ||

DB_NAME = os.getenv('DB_NAME', 'wormlab3d')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = int(os.getenv('DB_PORT', 27017))
DB_USERNAME = os.getenv('DB_USERNAME')
DB_PASSWORD = os.getenv('DB_PASSWORD')

# || -------------------------------- APP --------------------------------- ||

APP_SECRET = os.getenv('APP_SECRET')
APP_PORT = os.getenv('APP_PORT')

# || -------------------------------- LOGS --------------------------------- ||

cwd = Path.cwd()
dir_name = os.path.dirname(sys.argv[0]).replace(str(cwd), '').lstrip('/')
SCRIPT_PATH = (cwd / dir_name).resolve()
LOGS_PATH = ROOT_PATH / 'logs' / SCRIPT_PATH.relative_to(ROOT_PATH) / Path(sys.argv[0]).stem
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
