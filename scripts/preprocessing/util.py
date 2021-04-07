from typing import Tuple, List

from wormlab3d import CAMERA_IDXS
from wormlab3d.data.model.trial import Trial


def process_args(
        experiment_id: int = None,
        trial_id: int = None,
        camera_idx: int = None,
        frame_num: int = None,
) -> Tuple[List[Trial], list]:
    """
    Resolves any combination of passed experiments, trials and cameras.
    """

    # Get trials
    if trial_id is not None:
        trial = Trial.objects.get(id=trial_id)
        if experiment_id is not None:
            assert trial.experiment.id == experiment_id, 'Trial is not part of target experiment!'
        trials = [trial]
    else:
        if experiment_id is not None:
            trials = Trial.objects(experiment_id=experiment_id)
        else:
            trials = Trial.objects

    # Camera indices
    cam_idxs = CAMERA_IDXS
    if camera_idx is not None:
        assert trial_id is not None
        assert camera_idx in cam_idxs
        cam_idxs = [camera_idx]

    # Frame number - only makes sense in combination with a trial_id
    if frame_num is not None:
        assert trial_id is not None
        assert 0 <= frame_num < trial.num_frames

    return trials, cam_idxs
