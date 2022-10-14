from multiprocessing import Pool
from typing import Union, List

import numpy as np

from wormlab3d import logger, N_WORKERS
from wormlab3d.data.model import Eigenworms, Dataset
from wormlab3d.postures.cpca import CPCA
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.trajectories.cache import get_trajectory
from wormlab3d.trajectories.util import fetch_reconstruction


def calculate_nf_mc(X: np.ndarray) -> np.ndarray:
    """
    Calculate the natural frame curvature.
    """
    NF = NaturalFrame(X)
    return NF.mc


def _calculate_nf_mc_parallel(X: np.ndarray) -> np.ndarray:
    """
    Calculate natural frame curvatures in parallel.
    """
    with Pool(processes=N_WORKERS) as pool:
        res = pool.map(
            calculate_nf_mc,
            [Xi for Xi in X]
        )
    return np.array(res)


def calculate_natural_frame_curvatures(X: np.ndarray, parallel: bool = True) -> np.ndarray:
    """
    Calculate the natural frame curvatures for all the postures.
    """
    assert X.ndim == 3, 'Full postures required!'
    if parallel and N_WORKERS > 1:
        Z = _calculate_nf_mc_parallel(X)
    else:
        Z = np.zeros(len(X))
        for i, Xi in enumerate(X):
            Z[i] = calculate_nf_mc(Xi)
    return Z


def calculate_cpca(X: np.ndarray, n_components: int = None) -> CPCA:
    """
    Calculate the Complex-PCA decomposition of the shapes as represented in the Bishop frame.
    """
    M = X.shape[0]
    N = X.shape[1]
    if N == 1:
        raise RuntimeError('Trajectory returned only a single point. Eigenworms requires full postures!')
    logger.info(f'Calculating eigenworms with {M} midlines of length {N}.')

    # Calculate the natural frame representations for all midlines
    logger.info('Calculating natural frame representations.')
    Z = calculate_natural_frame_curvatures(X)

    # Z = []
    # bad_idxs = []
    # for i, Xi in enumerate(X):
    #     if (i + 1) % 100 == 0:
    #         logger.info(f'Calculating for midline {i + 1}/{M}.')
    #     try:
    #         nf = NaturalFrame(Xi)
    #         Z.append(nf.mc)
    #     except Exception:
    #         bad_idxs.append(i)
    # if len(bad_idxs):
    #     logger.warning(f'Failed to calculate {len(bad_idxs)} midline idxs: {bad_idxs}.')
    # Z = np.array(Z)

    # Calculate CPCA components
    logger.info('Calculating basis.')
    cpca = CPCA(n_components=n_components, whiten=False)
    cpca.fit(Z)

    return cpca


def generate_or_load_eigenworms(
        eigenworms_id: str = None,
        dataset_id: str = None,
        reconstruction_id: str = None,
        n_components: int = None,
        restrict_concs: List[float] = None,
        depth: int = -1,
        regenerate: bool = False
) -> Eigenworms:
    """
    Try to load an existing eigenworm basis or generate it otherwise.
    """
    eigenworms = fetch_eigenworms(eigenworms_id, dataset_id, reconstruction_id, n_components, restrict_concs)
    if eigenworms is not None:
        # Check that the components are there
        if eigenworms.components is None or eigenworms.cpca is None:
            logger.warning('Eigenworms cpca data not found, regenerating.')
            regenerate = True

        if not regenerate:
            return eigenworms

        # Use the dataset or reconstruction as defined in the eigenworm instance
        if eigenworms.dataset is not None:
            dataset_id = eigenworms.dataset.id
        else:
            reconstruction_id = eigenworms.reconstruction.id
        logger.info(f'Rebuilding matching eigenworms id={eigenworms.id}.')
    else:
        logger.info('No matching eigenworms found, generating.')
        eigenworms = Eigenworms()

    assert dataset_id is not None or reconstruction_id is not None, \
        'Must define a dataset or reconstruction to generate eigenworms basis from.'
    assert dataset_id is None or reconstruction_id is None, \
        'Cannot define a dataset and a reconstruction to generate eigenworms basis from.'
    assert reconstruction_id is None or restrict_concs is None, \
        'Cannot restrict_concs with a reconstruction defined.'

    X = None

    # Generate eigenworms for a dataset
    if dataset_id is not None:
        logger.info(f'Building eigenworms basis for dataset id={dataset_id}.')
        dataset = Dataset.objects.get(id=dataset_id)
        eigenworms.dataset = dataset
        X = dataset.X_all
        if restrict_concs is not None:
            eigenworms.restrict_concs = restrict_concs
            include_idxs = []
            for conc in restrict_concs:
                assert str(conc) in dataset.metas['concentration'], \
                    f'No data with concentration {conc} present in dataset!'
                include_idxs.extend(dataset.metas['concentration'][str(conc)])
            X = X[include_idxs]

    # Generate eigenworms for reconstruction
    if reconstruction_id is not None:
        logger.info(f'Building eigenworms basis for reconstruction id={reconstruction_id}.')
        reconstruction = fetch_reconstruction(reconstruction_id)
        eigenworms.reconstruction = reconstruction
        if reconstruction is None:
            raise RuntimeError('No matching reconstruction found, cannot compute eigenworms.')

        # Get the midlines
        X, meta = get_trajectory(reconstruction_id=reconstruction.id, depth=depth)

    # Calculate the CPCA
    if X is None or len(X) == 0:
        raise RuntimeError('Failed to find any midlines.')
    cpca = calculate_cpca(X, n_components=n_components)

    # Create or update the eigenworms instance
    if eigenworms.id is None:
        eigenworms.n_components = cpca.n_components_
        eigenworms.n_samples = cpca.n_samples_
        eigenworms.n_features = cpca.n_features_
    else:
        assert eigenworms.n_components == cpca.n_components_, 'n_components has changed!'
        assert eigenworms.n_samples == cpca.n_samples_, 'n_samples has changed!'
        assert eigenworms.n_features == cpca.n_features_, 'n_features has changed!'
    eigenworms.explained_variance = cpca.explained_variance_.tolist()
    eigenworms.explained_variance_ratio = cpca.explained_variance_ratio_.tolist()
    eigenworms.singular_values = cpca.singular_values_.tolist()
    eigenworms.mean = cpca.mean_
    eigenworms.components = cpca.components_

    eigenworms.save()
    logger.info(f'Eigenworms saved to database and components to {eigenworms.components_path}.')

    return eigenworms


def fetch_eigenworms(
        eigenworms_id: str = None,
        dataset_id: str = None,
        reconstruction_id: str = None,
        n_components: int = None,
        restrict_concs: List[float] = None
) -> Union[Eigenworms, None]:
    """
    Try to find a eigenworms instance satisfying arguments.
    """
    eigenworms = None
    if eigenworms_id is not None:
        eigenworms = Eigenworms.objects.get(id=eigenworms_id)
    else:
        # Try to find a suitable set of eigenworms
        filters = {}
        if dataset_id is not None:
            filters['dataset'] = dataset_id
        if reconstruction_id is not None:
            filters['reconstruction'] = reconstruction_id
        if n_components is not None:
            filters['n_components'] = n_components
        if restrict_concs is not None:
            filters['restrict_concs'] = restrict_concs
        else:
            filters['restrict_concs'] = None

        matching_eigenworms = Eigenworms.objects(**filters).order_by('-n_components')
        if matching_eigenworms.count() == 0:
            logger.warning(f'Found no eigenworms for parameters {filters}.')
        else:
            eigenworms = matching_eigenworms[0]
            logger.info(
                f'Found {matching_eigenworms.count()} matching eigenworms. '
                f'Using: {eigenworms}'
            )

    return eigenworms
