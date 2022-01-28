from argparse import ArgumentParser, Namespace
from typing import List

from wormlab3d import logger
from wormlab3d.data.model import Reconstruction, Trial, Eigenworms
from wormlab3d.postures.eigenworms import generate_or_load_eigenworms
from wormlab3d.toolkit.util import str2bool, print_args


def parse_args() -> Namespace:
    parser = ArgumentParser(description='Wormlab3D script to generate an eigenworm basis.')
    parser.add_argument('--dataset', type=str,
                        help='Dataset by id.')
    parser.add_argument('--experiment', type=int,
                        help='Experiment by id.')
    parser.add_argument('--trial', type=int,
                        help='Trial by id.')
    parser.add_argument('--reconstruction', type=str,
                        help='Reconstruction by id.')
    parser.add_argument('--n-components', type=int, default=10,
                        help='Number of eigenworms to generate (basis dimension).')
    parser.add_argument('--regenerate', type=str2bool, default=False,
                        help='Regenerate existing bases.')
    args = parser.parse_args()

    if args.dataset is not None:
        assert args.experiment is None and args.trial is None and args.reconstruction is None, \
            'Dataset cannot be specified along with any of experiment/trial/reconstruction.'

    print_args(args)

    return args


def _get_reconstruction_ids(args: Namespace) -> List[str]:
    """
    Get reconstructions ids requiring eigenworm generation.
    """
    if args.reconstruction is not None:
        logger.info(f'Fetching reconstructions id={args.reconstruction}.')
        reconstructions = [Reconstruction.objects.get(id=args.reconstruction)]
    elif args.trial is not None:
        logger.info(f'Fetching reconstructions for trial={args.trial}.')
        reconstructions = Reconstruction.objects(trial=args.trial)
    elif args.experiment is not None:
        logger.info(f'Fetching reconstructions for experiment={args.trial}.')
        trials = Trial.objects(experiment=args.experiment)
        reconstructions = []
        for trial in trials:
            trial_reconstructions = Reconstruction.objects(trial=trial)
            reconstructions.extend(trial_reconstructions)
    else:
        logger.info(f'Fetching ALL reconstructions.')
        reconstructions = Reconstruction.objects

    if len(reconstructions) == 0:
        raise RuntimeError('No reconstructions found!')

    rids = []
    for r in reconstructions:
        rids.append(r.id)

    return rids


def _check_cpca(
        eigenworms: Eigenworms,
        regenerate: bool,
        n_components: int,
        dataset_id: str = None,
        reconstruction_id: str = None
):
    # Check the CPCA works
    try:
        eigenworms.cpca
    except RuntimeError:
        if not regenerate:
            logger.warning('Could not restore CPCA from eigenworms. Attempting to regenerate.')
            args = {
                'n_components': n_components,
                'regenerate': True
            }
            if dataset_id is not None:
                args['dataset_id'] = dataset_id
            else:
                assert reconstruction_id is not None
                args['reconstruction_id'] = reconstruction_id
            eigenworms = generate_or_load_eigenworms(**args)
            eigenworms.cpca
        raise


def generate_eigenworms():
    args = parse_args()

    if args.dataset is not None:
        logger.info(f'------- Generating eigenworms for dataset id={args.dataset}.')
        eigenworms = generate_or_load_eigenworms(
            dataset_id=args.dataset,
            n_components=args.n_components,
            regenerate=args.regenerate
        )
        _check_cpca(eigenworms, args.regenerate, args.n_components, dataset_id=args.dataset)
    else:
        reconstruction_ids = _get_reconstruction_ids(args)
        for reconstruction_id in reconstruction_ids:
            logger.info(f'------- Generating eigenworms for reconstruction id={reconstruction_id}.')
            eigenworms = generate_or_load_eigenworms(
                reconstruction_id=reconstruction_id,
                n_components=args.n_components,
                regenerate=args.regenerate
            )
            _check_cpca(eigenworms, args.regenerate, args.n_components, reconstruction_id=reconstruction_id)


if __name__ == '__main__':
    generate_eigenworms()
