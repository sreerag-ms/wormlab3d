from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import scipy.io as sio

from wormlab3d import logger
from wormlab3d.data.model import Reconstruction, Trial, Midline3D
from wormlab3d.data.model.midline3d import M3D_SOURCE_WT3D
from wormlab3d.toolkit.util import print_args, str2bool


def get_args() -> Namespace:
    """
    Parse command line arguments.
    """
    parser = ArgumentParser(description='Wormlab3D script to import WT3D results.')
    parser.add_argument('--file', type=str, help='.mat file to import.', required=True)
    parser.add_argument('--trial', type=str, help='Trial by id.', required=True)
    parser.add_argument('--start-frame', type=int, help='Frame number to start from.', required=True)
    parser.add_argument('--dry-run', type=str2bool, help='Don\'t make any database changes.', default=True)
    args = parser.parse_args()

    print_args(args)

    return args


def import_mat():
    args = get_args()

    # Load the mat file
    path = Path(args.file)
    assert path.exists(), f'File {args.file} not found!'
    assert path.suffix == '.mat', 'File is not .mat!'
    filename = path.stem
    mat = sio.loadmat(path)
    X = mat['XYZ'].astype(np.float64).transpose(0, 2, 1)
    assert X.ndim == 3, 'Data has wrong number of dimensions!'
    assert X.shape[-1] == 3, 'Data dimensions unrecognised!'

    # Load the trial
    trial = Trial.objects.get(id=args.trial)

    # Generate midlines
    midlines = []
    frame_nums = range(args.start_frame, args.start_frame + len(X))
    for i, frame_num in enumerate(frame_nums):
        if (i + 1) % 10 == 0:
            logger.info(f'Generating midline {i + 1}/{len(X)}')

        # Get frame
        frame = trial.get_frame(frame_num)

        # Check for duplicates
        existing = frame.get_midlines3d(filters={'source': M3D_SOURCE_WT3D, 'source_file': filename})
        if existing.count() > 0:
            raise RuntimeError('Duplicate midline found with same filename, consider renaming import file.')

        # Create midline
        midline = Midline3D()
        midline.frame = frame
        midline.X = X[i]
        midline.source = M3D_SOURCE_WT3D
        midline.source_file = filename
        midline.validate()
        midlines.append(midline)

    assert len(midlines) == len(X)
    if not args.dry_run:
        logger.info(f'Importing {len(midlines)} midlines.')
        Midline3D.objects.insert(midlines)
    else:
        logger.info(f'DRY RUN: NOT Importing {len(midlines)} midlines.')

    rec = Reconstruction(
        trial=trial.id,
        start_frame=frame_nums[0],
        end_frame=frame_nums[-1],
        midlines=[m.id for m in midlines],
        source=M3D_SOURCE_WT3D,
        source_file=filename,
    )
    if not args.dry_run:
        rec.save()
        logger.info(f'Generated reconstruction id={rec.id}.')
    else:
        logger.info(f'DRY RUN: NOT generating reconstruction.')


if __name__ == '__main__':
    import_mat()
