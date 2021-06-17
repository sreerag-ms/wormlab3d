from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.nn.args.base_args import BaseArgs


class SimulationArgs(BaseArgs):
    def __init__(
            self,
            sim_id: str = None,
            worm_length: int = 10,
            duration: float = None,
            dt: float = None,
            **kwargs
    ):
        if sim_id is None:
            assert all(v is not None for v in [worm_length, duration, dt]), \
                'Simulation parameters not well defined.'
        else:
            assert all(v is None for v in [worm_length, duration, dt]), \
                'A simulation id will override any command-line simulation arguments.'
        self.sim_id = sim_id
        self.worm_length = worm_length
        self.duration = duration
        self.dt = dt

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Simulation Args')
        group.add_argument('--sim-id', type=str,
                           help='Load a simulation configuration by its database id.')
        group.add_argument('--worm-length', type=int, default=10,
                           help='Number of worm points along the body (default=10).')
        group.add_argument('--duration', type=float,
                           help='Time (in seconds) to run the simulation for.')
        group.add_argument('--dt', type=float,
                           help='Simulation timestep.')

        return group
