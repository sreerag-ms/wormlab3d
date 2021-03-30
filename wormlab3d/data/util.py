import re

from wormlab3d import ANNEX_PATH, DATA_PATH

DATA_PATH_PLACEHOLDER = '$DATA_PATH$'
ANNEX_PATH_PLACEHOLDER = '$ANNEX$'


def fix_path(path: str) -> str:
    """
    Replace any placeholders in paths with actual filesystem locations.
    """
    if path is None or path == '':
        return
    if DATA_PATH_PLACEHOLDER in path:
        path = path.replace(DATA_PATH_PLACEHOLDER, DATA_PATH)
    if ANNEX_PATH_PLACEHOLDER in path:
        path = path.replace(ANNEX_PATH_PLACEHOLDER, ANNEX_PATH)

    return path


def parse_size(size: str) -> int:
    """
    Parse a size string like "20k" or "13.5 megabytes" into bytes.
    based on https://stackoverflow.com/a/42865957/2002471
    """
    units = {'B': 1, 'KB': 2**10, 'MB': 2**20, 'MB': 2**20, 'GB': 2**30, 'TB': 2**40}
    key_map = {'BYTES': 'B', 'KILOBYTES': 'KB', 'K': 'KB', 'MEGABYTES': 'MB', 'M': 'MB', 'GIGABYTES': 'GB', 'G': 'GB',
               'TERABYTES': 'TB', 'T': 'TB'}
    size = size.upper()
    pattern = re.compile('|'.join(key_map.keys()))
    size = pattern.sub(lambda m: key_map[re.escape(m.group(0))], size)
    if not re.match(r' ', size):
        size = re.sub(r'([KMGT]?B)', r' \1', size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])
