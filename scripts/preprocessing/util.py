from wormlab3d.data.model.trial import Trial, CAMERA_IDXS


def process_args(
        experiment_id=None,
        trial_id=None,
        camera_idx=None,
):
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

    return trials, cam_idxs
