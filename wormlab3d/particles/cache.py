from argparse import Namespace
from typing import Union

from wormlab3d import logger
from wormlab3d.data.model import PEParameters
from wormlab3d.particles.args.parameter_args import ParameterArgs
from wormlab3d.particles.simulation_state import SimulationState


def _init_parameters(args: ParameterArgs) -> PEParameters:
    """
    Create or load particle explorer parameters.
    """
    db_params = args.get_db_params()
    parameters = None

    # If we have a model id then load this from the database
    if args.params_id is not None:
        parameters = PEParameters.objects.get(id=args.params_id)
    else:
        # Otherwise, try to find one matching the same parameters
        params_matching = PEParameters.objects(**db_params)
        if params_matching.count() > 0:
            parameters = params_matching[0]
            logger.info(
                f'Found {len(params_matching)} suitable parameter records in database, using most recent.')
        else:
            logger.info(f'No suitable parameter records found in database.')
    if parameters is not None:
        logger.info(f'Loaded parameters (id={parameters.id}, created={parameters.created}).')

    # Not loaded model, so create one
    if parameters is None:
        parameters = PEParameters(**db_params)
        parameters.save()
        logger.info(f'Saved parameters to database (id={parameters.id})')

    return parameters


def get_sim_state_from_args(args: Union[ParameterArgs, Namespace]) -> SimulationState:
    """
    Generate or load the trajectories from parameters set in an argument namespace.
    """
    if type(args) == Namespace:
        args = ParameterArgs.from_args(args)
    params = _init_parameters(args)
    SS = SimulationState(params, read_only=False, regenerate=args.regenerate)
    return SS
