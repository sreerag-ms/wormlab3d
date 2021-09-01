from argparse import ArgumentParser, _ArgumentGroup

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            resume: bool = True,
            resume_from: str = 'latest',
            n_steps: int = 300,
            checkpoint_every_n_steps: int = 1,
            plot_every_n_steps: int = -1,
            plot_n_examples: int = 1,
            save_plots: bool = False,
            videos_every_n_steps: int = -1,
            parallel_solvers: int = 0,
            **kwargs
    ):
        self.resume = resume
        self.resume_from = resume_from
        self.n_steps = n_steps
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.plot_every_n_steps = plot_every_n_steps
        self.plot_n_examples = plot_n_examples
        self.save_plots = save_plots
        self.videos_every_n_steps = videos_every_n_steps
        self.parallel_solvers = parallel_solvers

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> _ArgumentGroup:
        """
        Add arguments to a command parser.
        """
        group = parser.add_argument_group('Runtime Args')
        resume_parser = group.add_mutually_exclusive_group(required=False)
        resume_parser.add_argument('--resume', action='store_true',
                                   help='Resume from a previous checkpoint.')
        resume_parser.add_argument('--no-resume', action='store_false', dest='resume',
                                   help='Do not resume from a previous checkpoint.')
        resume_parser.set_defaults(resume=False)
        group.add_argument('--resume-from', type=str, default='latest',
                           help='Resume from a specific checkpoint id, or "latest" or "best". Default="latest".')
        group.add_argument('--n-steps', type=int, default=300,
                           help='Number of steps to run for.')
        group.add_argument('--checkpoint-every-n-steps', type=int, default=1,
                           help='Save a checkpoint every n steps, -1 turns this off.')
        group.add_argument('--plot-every-n-steps', type=int, default=-1,
                           help='Plot example inputs and outputs every n steps, -1 turns this off.')
        group.add_argument('--plot-n-examples', type=int, default=1,
                           help='Plot this number of examples from the batch at each iteration.')
        group.add_argument('--save-plots', type=str2bool, default=False,
                           help='Save plot images to disk. Default = False.')
        group.add_argument('--videos-every-n-steps', type=int, default=-1,
                           help='Generate videos every n steps, -1 turns this off.')
        group.add_argument('--parallel-solvers', type=int, default=0,
                           help='Number of parallel solvers to use for batch processing. If 0 (default) then serial mode used.')

        return group
