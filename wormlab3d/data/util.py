from wormlab3d import ANNEX_PATH, DATA_PATH

DATA_PATH_PLACEHOLDER = '$DATA_PATH$'
ANNEX_PATH_PLACEHOLDER = '$ANNEX$'


def fix_path(path: str) -> str:
    """
    Replace any placeholders in paths with actual filesystem locations.
    """
    if path is None:
        return
    if DATA_PATH_PLACEHOLDER in path:
        path = path.replace(DATA_PATH_PLACEHOLDER, DATA_PATH)
    if ANNEX_PATH_PLACEHOLDER in path:
        path = path.replace(ANNEX_PATH_PLACEHOLDER, ANNEX_PATH)

    return path
