import numpy as np
from wormlab3d import logger
from wormlab3d.data.model import Reconstruction
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import calculate_speeds


def main():
    reconstruction_ids = []
    for trial in Reconstruction.objects:
        reconstruction_ids.append(trial.id)

    for rid in reconstruction_ids:
        logger.info(f'Checking reconstruction {rid}.')

        try:
            X, meta = get_trajectory(
                reconstruction_id=rid,
                smoothing_window=11
            )
        except Exception:
            logger.warning('Could not load, skipping.')
            continue
        speeds = calculate_speeds(X, signed=True)

        N = len(X) - (speeds == 0).sum() - np.isnan(speeds).sum()
        fwd = (speeds > 0).sum() / N
        bwd = (speeds < 0).sum() / N

        logger.info(f'Forwards: {fwd * 100:.2f}%. Backwards: {bwd * 100:.2f}%. Total frames: {N}.')

        if bwd < fwd:
            logger.info('Backwards < forwards, OK.')
        elif N < 25 * 4:
            logger.info('Less than 4 seconds, ignoring.')
        else:
            logger.warning('Backwards > forwards, flipped?')


if __name__ == '__main__':
    main()

