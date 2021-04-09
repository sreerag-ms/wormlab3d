from typing import List

import torch
from torchvision.transforms.functional import to_tensor

from wormlab3d import logger, PREPARED_IMAGE_SIZE, CAMERA_IDXS
from wormlab3d.data.model import Checkpoint, SegmentationMasks
from wormlab3d.midlines2d.args import DatasetArgs, NetworkArgs, OptimiserArgs, RuntimeArgs
from wormlab3d.midlines2d.manager import Manager
from wormlab3d.toolkit.util import resolve_targets, to_numpy


def generate_2d_segmentation_masks(
        checkpoint_id: str,
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None,
        frame_batch_size: int = 16,
        missing_only: bool = True
):
    """
    Generate triplets of 2D segmentation masks for matching frames by processing the prepared
    images for each camera view with a pre-trained neural network, which must be specified.
    """
    assert checkpoint_id is not None
    checkpoint = Checkpoint.objects.get(id=checkpoint_id)
    logger.info(f'Loaded checkpoint id={checkpoint.id}')
    nn_batch_size = frame_batch_size * 3

    # Build arguments - don't use the args stored in the checkpoint as these
    # refer to how it was trained, not how we intend to use it now
    dataset_args = DatasetArgs(ds_id=checkpoint.dataset.id, load=True)
    net_args = NetworkArgs(net_id=checkpoint.network_params.id, load=True)
    optimiser_args = OptimiserArgs(**checkpoint.optimiser_args)  # not used
    runtime_args = RuntimeArgs(
        resume=True,
        resume_from=checkpoint.id,
        gpu_only=False,
        batch_size=nn_batch_size
    )

    # Construct manager
    manager = Manager(
        dataset_args=dataset_args,
        net_args=net_args,
        optimiser_args=optimiser_args,
        runtime_args=runtime_args
    )

    # Resolve which trials want processing
    trials, cam_idxs = resolve_targets(experiment_id, trial_id, camera_idx, frame_num)
    logger.info(f'Generating 2D segmentation maps for {len(trials)} trials.')
    batch: List[SegmentationMasks] = []

    def process_batch():
        """Process a batch of frames/masks through the network."""
        nonlocal batch
        logger.info('Processing batch...')

        # Stack all views from all frames
        X = torch.zeros((nn_batch_size, 1) + PREPARED_IMAGE_SIZE)
        X[:len(batch) * 3] = torch.stack([
            to_tensor(m.frame.images[c])
            for m in batch for c in CAMERA_IDXS
        ])

        # Run through the network. Shape=(nn_batch_size, 1, 200, 200)
        Y_pred = to_numpy(manager.predict(X))

        # Unpack and split up into updates and inserts
        batch_updates = []
        batch_inserts = []
        for i, m in enumerate(batch):
            m.X = Y_pred[i * 3:(i + 1) * 3].squeeze()  # (3, 200, 200)
            m.validate()
            if m.id is None:
                batch_inserts.append(m)
            else:
                batch_updates.append(m)
        if len(batch_inserts):
            SegmentationMasks.objects.insert(batch_inserts)
        if len(batch_updates):
            SegmentationMasks.objects.update(batch_updates)
        batch = []

    # Iterate over matching trials
    for trial in trials:
        logger.info(f'Processing trial id={trial.id}')

        # Find any trials with existing masks for this checkpoint
        existing = SegmentationMasks.objects(
            trial=trial,
            checkpoint=checkpoint
        )
        masks_existing = {m.frame.id: m for m in existing}

        # Iterate over the frames
        if frame_num is not None:
            frames = [trial.get_frame(frame_num)]
        else:
            if missing_only:
                filters = {'id__nin': masks_existing.keys()}
            else:
                filters = {}
            frames = trial.get_frames(filters)
        for frame in frames:
            log_prefix = f'Frame #{frame.frame_num}/{trial.num_frames} (id={frame.id}). '
            if not frame.is_ready():
                logger.warning(log_prefix + 'Not ready, skipping.')
                continue

            if frame.id in masks_existing.keys():
                if missing_only:
                    logger.info(log_prefix + 'Already exists, skipping.')
                    continue
                else:
                    logger.info(log_prefix + 'Already exists, replacing.')
                masks = existing[0]
            else:
                logger.info(log_prefix + 'Adding to batch.')
                masks = SegmentationMasks(
                    trial=trial,
                    frame=frame,
                    checkpoint=checkpoint
                )

            batch.append(masks)
            if len(batch) == frame_batch_size:
                process_batch()

    # Process any leftovers
    process_batch()


if __name__ == '__main__':
    generate_2d_segmentation_masks(
        checkpoint_id='606f270a27951dfe6533e26c',
        frame_batch_size=6  # this is multiplied by 3 for the neural-network batch size
    )
