import os
import shutil
import subprocess

from wormlab3d import ANNEX_PATH, logger, MIN_FREE_DISK_SPACE
from wormlab3d.data.util import parse_size, ANNEX_PATH_PLACEHOLDER


def is_annexed_file(path: str) -> bool:
    """
    Checks if the path points at a git-annexed file.
    """
    return (ANNEX_PATH_PLACEHOLDER in path or str(ANNEX_PATH) in path) and os.path.islink(path)


def _execute_annex_cmd(cmd: str, path: str, check: bool = True) -> subprocess.CompletedProcess:
    """
    Executes a git annex command via a subprocess.
    Allowed cmds are "info" and "get". A path must be provided.
    """
    assert cmd in ['info', 'get']
    cmd = ['git', 'annex', cmd, path]
    logger.debug(f'Executing: "{" ".join(cmd)}"')
    proc = subprocess.run(
        cmd,
        cwd=ANNEX_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check  # raises exception for non-zero return codes
    )
    return proc


def fetch_from_annex(path: str, quiet: bool = False):
    """
    Fetch a git-annexed file. 
    If not already locally available it tries to fetch it using the command line git-annex functionality. 
    """
    proc = _execute_annex_cmd('info', path)
    if not quiet:
        logger.debug(proc.stdout)

    # Parse output
    info = proc.stdout.decode().splitlines()
    present_line = info[-1]
    assert present_line[:9] == 'present: '
    assert present_line[9:] in ['true', 'false']
    present = present_line[9:] == 'true'

    # If the file is available, continue as usual
    if present:
        if not quiet:
            logger.debug('Annexed file is available locally.')
        return
    # Otherwise we need to fetch it from the annex

    # First, check there is disk space available
    filesize_line = info[1]
    assert filesize_line[:6] == 'size: '
    filesize_str = filesize_line[6:]
    filesize = parse_size(filesize_str)
    free_space = shutil.disk_usage(ANNEX_PATH).free
    min_required = parse_size(MIN_FREE_DISK_SPACE)
    if free_space - filesize < min_required:
        raise RuntimeError(f'Cannot fetch annexed file {path} ({filesize_str}) '
                           f'as it would leave less than {MIN_FREE_DISK_SPACE} free disk space.')

    # Fetch the file from the annex
    _execute_annex_cmd('get', path)
    if not quiet:
        logger.debug('File fetched from annex.')
