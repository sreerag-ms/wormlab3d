from argparse import ArgumentParser, _ArgumentGroup
from typing import Tuple, Optional

from wormlab3d.data.model import PEParameters
from wormlab3d.data.model.pe_parameters import PE_PAUSE_TYPES, PE_ANGLE_DIST_TYPES
from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class ParameterArgs(BaseArgs):
    def __init__(
            self,
            load: bool,
            params_id: Optional[str],
            regenerate: bool,

            sim_duration: float,
            sim_dt: float,
            batch_size: int,

            rate_01: float,
            rate_10: float,
            rate_02: float,
            rate_20: float,
            speeds_0_mu: float,
            speeds_0_sig: float,
            speeds_1_mu: float,
            speeds_1_sig: float,

            theta_dist_type: str,
            theta_dist_params: Tuple[float],
            phi_dist_type: str,
            phi_dist_params: Tuple[float],

            nonp_pause_type: str,
            nonp_pause_max: float,

            **kwargs
    ):
        self.load = load
        self.params_id = params_id
        self.regenerate = regenerate

        self.duration = sim_duration
        self.dt = sim_dt
        self.batch_size = batch_size

        self.rate_01 = rate_01
        self.rate_10 = rate_10
        self.rate_02 = rate_02
        self.rate_20 = rate_20

        self.speeds_0_mu = speeds_0_mu
        self.speeds_0_sig = speeds_0_sig
        self.speeds_1_mu = speeds_1_mu
        self.speeds_1_sig = speeds_1_sig

        self.theta_dist_type = theta_dist_type
        self.theta_dist_params = theta_dist_params
        self.phi_dist_type = phi_dist_type
        self.phi_dist_params = phi_dist_params

        self.delta_type = nonp_pause_type
        self.delta_max = nonp_pause_max

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Particle Explorer Parameters')

        group.add_argument('--load', type=str2bool, default=True,
                           help='Try to load an existing parameters database object if available matching the given parameters.')
        group.add_argument('--params-id', type=str,
                           help='Load existing parameters by its database id.')
        group.add_argument('--regenerate', type=str2bool, default=False,
                           help='Regenerate the simulation data.')

        parser.add_argument('--batch-size', type=int,
                            help='Batch size.')
        parser.add_argument('--sim-duration', type=float,
                            help='Simulation time.')
        parser.add_argument('--sim-dt', type=float,
                            help='Simulation timestep.')

        parser.add_argument('--rate-01', type=float,
                            help='Transition rate from slow speed to fast speed.')
        parser.add_argument('--rate-10', type=float,
                            help='Transition rate from fast speed to slow speed.')
        parser.add_argument('--rate-02', type=float,
                            help='Transition rate from slow speed to turn.')
        parser.add_argument('--rate-20', type=float,
                            help='Transition rate from turn to slow speed.')

        parser.add_argument('--speeds-0-mu', type=float,
                            help='Slow speed average.')
        parser.add_argument('--speeds-0-sig', type=float,
                            help='Slow speed standard deviation.')
        parser.add_argument('--speeds-1-mu', type=float,
                            help='Fast speed average.')
        parser.add_argument('--speeds-1-sig', type=float,
                            help='Fast speed standard deviation.')

        parser.add_argument('--theta-dist-type', type=str, choices=PE_ANGLE_DIST_TYPES,
                            help='Planar angle distribution type.')
        parser.add_argument('--theta-dist-params', type=lambda s: [float(item) for item in s.split(',')],
                            help='Planar angle distribution parameters.')
        parser.add_argument('--phi-dist-type', type=str, choices=PE_ANGLE_DIST_TYPES,
                            help='Non-planar angle distribution type.')
        parser.add_argument('--phi-dist-params', type=lambda s: [float(item) for item in s.split(',')],
                            help='Non-planar angle distribution parameters.')

        parser.add_argument('--nonp-pause-type', type=str, choices=PE_PAUSE_TYPES,
                            help='Non-planar turn pause penalty type.')
        parser.add_argument('--nonp-pause-max', type=float,
                            help='Maximum non-planar turn pause penalty.')

        return group

    def get_db_params(self) -> dict:
        p = {}
        for k in PEParameters._fields.keys():
            if hasattr(self, k):
                p[k] = getattr(self, k)
        return p
