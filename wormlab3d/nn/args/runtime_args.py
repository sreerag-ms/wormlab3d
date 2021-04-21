from argparse import ArgumentParser, _ArgumentGroup
from typing import List

from wormlab3d.nn.args.base_args import BaseArgs


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            resume: bool = True,
            resume_from: str = 'latest',
            gpu_only: bool = False,
            cpu_only: bool = False,
            batch_size: int = 32,
            n_epochs: int = 300,
            checkpoint_every_n_epochs: int = 1,
            checkpoint_every_n_batches: int = -1,
            plot_every_n_batches: int = -1,
            plot_n_examples: int = 4,
            track_metrics: List[str] = [],
            **kwargs
    ):
        assert not (cpu_only and gpu_only), 'Invalid combination: gpu_only AND cpu_only!'
        self.resume = resume
        self.resume_from = resume_from
        self.gpu_only = gpu_only
        self.cpu_only = cpu_only
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_batches = checkpoint_every_n_batches
        self.plot_every_n_batches = plot_every_n_batches
        self.plot_n_examples = plot_n_examples
        self.track_metrics = track_metrics

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
        group.add_argument('--gpu-only', action='store_true',
                           help='Abort if no gpus are detected.')
        group.add_argument('--cpu-only', action='store_true',
                           help='Only run on CPU. Otherwise will use GPU if available.')
        group.add_argument('--batch-size', type=int, default=32,
                           help='Batch size to use for training and testing')
        group.add_argument('--n-epochs', type=int, default=300,
                           help='Number of epochs to run for.')
        group.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                           help='Save a checkpoint every n epochs, -1 turns this off.')
        group.add_argument('--checkpoint-every-n-batches', type=int, default=-1,
                           help='Save a checkpoint every n batches, -1 turns this off.')
        group.add_argument('--plot-every-n-batches', type=int, default=-1,
                           help='Plot example inputs and outputs every n batches, -1 turns this off.')
        group.add_argument('--plot-n-examples', type=int, default=4,
                           help='Show this many random examples in a single plot.')
        group.add_argument('--track-metrics', type=lambda s: [str(item) for item in s.split(',')], default=[],
                            help='Comma delimited list of metrics to track.')

        return group
