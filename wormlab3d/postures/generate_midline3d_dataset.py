from typing import List

import numpy as np
from scipy import interpolate

from wormlab3d import logger
from wormlab3d.data.model import Trial, Reconstruction
from wormlab3d.data.model.dataset import DatasetMidline3D, Dataset
from wormlab3d.postures.args import DatasetMidline3DArgs
from wormlab3d.trajectories.cache import get_trajectory


def fetch_matching_trial_ids(args: DatasetMidline3DArgs) -> List[int]:
    """
    Build a pipeline to fetch trials.
    """
    pipeline = [
        {'$lookup': {
            'from': 'experiment',
            'localField': 'experiment',
            'foreignField': '_id',
            'as': 'experiment'
        }},
        {'$unwind': {'path': '$experiment'}},
    ]

    # Include/exclude experiments/trials
    matches = []
    if len(args.include_experiments) > 0:
        matches.append({'experiment': {'$in': args.include_experiments}})
    if len(args.exclude_experiments) > 0:
        matches.append({'experiment': {'$nin': args.exclude_experiments}})
    if len(args.include_trials) > 0:
        matches.append({'_id': {'$in': args.include_trials}})
    if len(args.exclude_trials) > 0:
        matches.append({'_id': {'$nin': args.exclude_trials}})

    # Trial quality
    if args.min_trial_quality > 0:
        matches.append({'quality': {'$gte': args.min_trial_quality}})

    # Restrict to users
    if len(args.restrict_users) > 0:
        matches.append({'experiment.user': {'$in': args.restrict_concs}})

    # Restrict to strains
    if len(args.restrict_strains) > 0:
        matches.append({'experiment.strain': {'$in': args.restrict_strains}})

    # Restrict to sexes
    if len(args.restrict_sexes) > 0:
        matches.append({'experiment.sex': {'$in': args.restrict_sexes}})

    # Restrict to ages
    if len(args.restrict_ages) > 0:
        matches.append({'experiment.age': {'$in': args.restrict_ages}})

    # Restrict to concentrations
    if len(args.restrict_concs) > 0:
        matches.append({'experiment.concentration': {'$in': args.restrict_concs}})

    # Add matches to pipeline
    if len(matches):
        pipeline.append({'$match': {'$and': matches}})

    # Project and sort by trial id
    pipeline.extend([
        {'$project': {'_id': 1}},
        {'$sort': {'_id': 1}},
    ])

    # Collate results
    cursor = Trial.objects().aggregate(pipeline)
    trial_ids = []
    for res in cursor:
        trial_ids.append(res['_id'])

    logger.info(f'Fetched {len(trial_ids)} matching trials.')

    return trial_ids


def fetch_matching_reconstruction_ids(trial_ids: List[int], args: DatasetMidline3DArgs) -> List[str]:
    """
    Build a pipeline to fetch reconstructions.
    """
    pipeline = []

    # Match trials
    matches = {'trial': {'$in': trial_ids}}

    # Match midline sources
    if len(args.restrict_sources) > 0:
        matches['source'] = {'$in': args.restrict_sources}

    # Add matches to pipeline
    pipeline.append({'$match': matches})

    # Project necessary reconstruction fields
    pipeline.append({'$project': {'_id': 1, 'trial': 1, 'source': 1, 'source_file': 1,
                                  'n_frames': {'$subtract': ['$end_frame', '$start_frame']}}})

    # Match minimum reconstruction duration
    if args.min_reconstruction_frames is not None:
        pipeline.append({'$match': {'n_frames': {'$gte': args.min_reconstruction_frames}}})

    # Group results by trial
    pipeline.extend([
        {'$group': {
            '_id': '$trial',
            'reconstructions': {'$push': {
                '_id': '$_id',
                'source': '$source',
                'source_file': '$source_file',
            }}
        }},
        {'$sort': {'_id': 1}},
    ])

    # Collate results, ensuring just one reconstruction per trial
    cursor = Reconstruction.objects().aggregate(pipeline)
    reconstruction_ids = []
    for res in cursor:
        reconstructions = res['reconstructions']
        if len(reconstructions) == 1:
            reconstruction_ids.append(reconstructions[0]['_id'])
        else:
            r_objs = []
            for r in reconstructions:
                r_objs.append(Reconstruction.objects.get(id=r['_id']))

            # Prefer the longest
            r_objs.sort(key=lambda r: r.n_frames, reverse=True)
            r_objs_longest = [r_objs[0]]
            for r_obj in r_objs[1:]:
                if r_obj.n_frames == r_objs[0].n_frames:
                    r_objs_longest.append(r_obj)
                else:
                    break
            if len(r_objs_longest) == 1:
                reconstruction_ids.append(r_objs_longest[0].id)
                continue
            r_objs = r_objs_longest

            # If multiple are the longest then prefer MF > reconst > WT3D
            r_objs_by_source = {}
            for r_obj in r_objs:
                if r_obj.source not in r_objs_by_source:
                    r_objs_by_source[r_obj.source] = []
                r_objs_by_source[r_obj.source].append(r_obj)
            r_pref = []
            for source in ['MF', 'reconst', 'WT3D']:
                if source not in r_objs_by_source:
                    continue
                r_src = r_objs_by_source[source]
                if len(r_src) == 0:
                    continue
                else:
                    r_pref = r_src
                    break
            if len(r_pref) == 1:
                reconstruction_ids.append(r_pref[0].id)
                continue
            r_objs = r_pref

            # If multiple MF, pick most recent
            if r_objs[0].source == 'MF':
                r_objs.sort(key=lambda r: r.updated, reverse=True)
                reconstruction_ids.append(r_objs[0].id)
                continue

            # Finally, if multiple reconst or WT3D, sort by filename and hopefully get the most recent
            r_objs.sort(key=lambda r: r.source_file, reverse=True)
            reconstruction_ids.append(r_objs[0].id)

    logger.info(f'Fetched {len(reconstruction_ids)} matching reconstructions.')

    return reconstruction_ids


def generate_midline3d_dataset(args: DatasetMidline3DArgs) -> DatasetMidline3D:
    """
    Generate a 3D midline dataset.
    """
    logger.info('Generating 3D midline dataset.')

    # Fetch all matching trials
    trial_ids = fetch_matching_trial_ids(args)
    if len(trial_ids) == 0:
        raise RuntimeError('No matching trials, cannot build a dataset!')

    # Fetch all matching reconstructions
    reconstruction_ids = fetch_matching_reconstruction_ids(trial_ids, args)
    if len(reconstruction_ids) == 0:
        raise RuntimeError('No matching reconstructions, cannot build a dataset!')

    # Fetch the full trajectories for all the reconstructions
    logger.info('Fetching trajectories.')
    Xs = []
    metas = {
        'reconstruction': {},
        'user': {},
        'concentration': {},
        'source': {},
    }
    idx = 0
    for rid in reconstruction_ids:
        reconstruction = Reconstruction.objects.get(id=rid)
        trial = reconstruction.trial
        experiment = trial.experiment
        logger.info(f'------- Loading midlines for trial={trial.id} / reconstruction={rid}.')
        try:
            X, meta = get_trajectory(reconstruction_id=rid, depth=args.mf_depth)
        except Exception as e:
            logger.warning(f'Could not get trajectory. {e}')
            continue

        # Discard any invalid midlines
        if np.isnan(X).any():
            is_nan = np.any(np.isnan(X), axis=(1, 2))
            logger.warning(f'Discarding {len(is_nan.nonzero()[0])}/{len(X)} bad midlines.')
            X = X[~is_nan]

        # todo: check worm lengths and discard any out of range
        # todo: rescale worms to all have length 1?

        # Resample if required
        if X.shape[1] != args.n_worm_points:
            X_new = np.zeros((X.shape[0], args.n_worm_points, 3))
            sl = np.linalg.norm(X[:, :-1] - X[:, 1:], axis=-1)
            u = np.c_[np.zeros((X.shape[0], 1)), sl.cumsum(axis=-1)]
            u = u / u[:, -1][:, None]
            u_new = np.linspace(0, 1, args.n_worm_points)

            for i, Xi in enumerate(X):
                for j in range(3):
                    tck = interpolate.splrep(u[i], Xi[:, j], s=1e-4, k=3)
                    X_new[i, :, j] = interpolate.splev(u_new, tck)

            X = X_new

        idxs = list(range(idx, idx + len(X)))
        idx += len(X)

        # Set meta data
        metas['reconstruction'][str(rid)] = idxs

        if experiment.user not in metas['user']:
            metas['user'][experiment.user] = []
        metas['user'][experiment.user].extend(idxs)

        if experiment.concentration not in metas['concentration']:
            metas['concentration'][experiment.concentration] = []
        metas['concentration'][experiment.concentration].extend(idxs)

        if reconstruction.source not in metas['source']:
            metas['source'][reconstruction.source] = []
        metas['source'][reconstruction.source].extend(idxs)

        Xs.append(X)

    # Combine and centre
    Xs = np.concatenate(Xs)
    Xs = Xs - Xs.mean(axis=1, keepdims=True)

    # # Shuffle todo: need to fix the metas
    # rng = default_rng()
    # rng.shuffle(Xs)

    # Train/test split
    N = len(Xs)
    if args.train_test_split < 1:
        # todo: splitting breaks metas
        raise NotImplementedError()
        N_train = int(np.floor(args.train_test_split * N))
        X_train = Xs[:N_train]
        X_test = Xs[N_train:]
    else:
        X_train = Xs
        X_test = None

    # Build the dataset
    logger.info('Building dataset.')
    DSs = Dataset.find_from_args(args)
    if len(DSs):
        DS = DSs[0]
    else:
        DS: DatasetMidline3D = Dataset.from_args(args)
    DS.set_data(train=X_train, test=X_test, metas=metas)

    # Save dataset
    logger.info('Saving dataset.')
    DS.save()
    logger.info(f'Dataset id={DS.id}.')

    return DS
