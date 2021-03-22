from wormlab3d import WORM_DATA_PATH, DATA_PATH

DATA_PATH_PLACEHOLDER = '$DATA_PATH$'
WORM_DATA_PATH_PLACEHOLDER = '$WORM_DATA$'


def fix_path(path: str) -> str:
    """
    Replace any placeholders in paths with actual filesystem locations.
    """
    if path is None:
        return
    if DATA_PATH_PLACEHOLDER in path:
        path = path.replace(DATA_PATH_PLACEHOLDER, DATA_PATH)
    if WORM_DATA_PATH_PLACEHOLDER in path:
        path = path.replace(WORM_DATA_PATH_PLACEHOLDER, WORM_DATA_PATH)

    return path
