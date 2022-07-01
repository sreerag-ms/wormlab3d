from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.data.model.midline3d import M3D_SOURCE_MF
from wormlab3d.midlines3d.trial_state import TrialState

SYMBOLS = {
    'customary': ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'),
    'customary_ext': ('byte', 'kilo', 'mega', 'giga', 'tera', 'peta', 'exa',
                      'zetta', 'iotta'),
    'iec': ('Bi', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi', 'Yi'),
    'iec_ext': ('byte', 'kibi', 'mebi', 'gibi', 'tebi', 'pebi', 'exbi',
                'zebi', 'yobi'),
}


def bytes2human(n, format='%(value).1f %(symbol)s', symbols='customary'):
    """
    https://stackoverflow.com/questions/13343700/bytes-to-human-readable-and-back-without-data-loss
    Convert n bytes into a human readable string based on format.
    symbols can be either "customary", "customary_ext", "iec" or "iec_ext",
    see: http://goo.gl/kTQMs
    """
    n = int(n)
    if n < 0:
        raise ValueError("n < 0")
    symbols = SYMBOLS[symbols]
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i + 1) * 10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)


def delete_mask_renders(dry_run: bool = True):
    """
    Delete masks renders.
    """
    logger.info('Deleting mask renders.')
    if dry_run:
        logger.info('DRY RUN - no files will be removed.')

    r_ids = []
    for reconstruction in Reconstruction.objects(source=M3D_SOURCE_MF):
        r_ids.append(reconstruction.id)

    files_removed = 0
    cum_size = 0

    for i, r_id in enumerate(r_ids):
        reconstruction = Reconstruction.objects.get(id=r_id)
        logger.info(f'Checking reconstruction={reconstruction.id} ({i + 1}/{len(r_ids)}).')
        try:
            ts = TrialState(reconstruction=reconstruction)
        except Exception:
            logger.warning('Could not load trial state.')
            continue

        for name in ['curve', 'target', 'target_residuals']:
            fn = ts.path / f'masks_{name}.npz'
            if fn.exists():
                size = fn.stat().st_size
                logger.info(f'Removing {fn} ({bytes2human(size)}).')
                if not dry_run:
                    fn.unlink()
                files_removed += 1
                cum_size += size

    logger.info(f'Files removed = {files_removed}.')
    logger.info(f'Disk space freed = {bytes2human(cum_size)}.')


if __name__ == '__main__':
    delete_mask_renders()
