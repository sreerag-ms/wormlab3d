from argparse import ArgumentParser, Namespace


class RuntimeArgs:
    def __init__(
            self,
            resume: bool = True,
            resume_from: str = 'latest',
            gpu_only: bool = False,
            batch_size: int = 32,
            n_epochs: int = 300,
            checkpoint_every_n_epochs: int = 1,
            checkpoint_every_n_batches: int = -1,
            plot_every_n_batches: int = -1,
            plot_n_examples: int = 4,
    ):
        self.resume = resume
        self.resume_from = resume_from
        self.gpu_only = gpu_only
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.checkpoint_every_n_batches = checkpoint_every_n_batches
        self.plot_every_n_batches = plot_every_n_batches
        self.plot_n_examples = plot_n_examples

    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Add arguments to a command parser.
        """
        resume_parser = parser.add_mutually_exclusive_group(required=False)
        resume_parser.add_argument('--resume', action='store_true',
                                   help='Resume from a previous checkpoint.')
        resume_parser.add_argument('--no-resume', action='store_false', dest='resume',
                                   help='Do not resume from a previous checkpoint.')
        resume_parser.set_defaults(resume=False)
        parser.add_argument('--resume-from', type=str,
                            help='Resume from a specific checkpoint id.')
        parser.add_argument('--gpu-only', action='store_true',
                            help='Abort if no gpus are detected.')
        parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size to use for training and testing')
        parser.add_argument('--n-epochs', type=int, default=300,
                            help='Number of epochs to run for.')
        parser.add_argument('--checkpoint-every-n-epochs', type=int, default=1,
                            help='Save a checkpoint every n epochs, -1 turns this off.')
        parser.add_argument('--checkpoint-every-n-batches', type=int, default=-1,
                            help='Save a checkpoint every n batches, -1 turns this off.')
        parser.add_argument('--plot-every-n-batches', type=int, default=-1,
                            help='Plot example inputs and outputs every n batches, -1 turns this off.')
        parser.add_argument('--plot-n-examples', type=int, default=4,
                            help='Show this many random examples in a single plot.')

    @staticmethod
    def from_args(args: Namespace) -> 'RuntimeArgs':
        """
        Create a RuntimeParameters instance from command-line arguments.
        """
        return RuntimeArgs(
            resume=args.resume,
            resume_from=args.resume_from,
            gpu_only=args.gpu_only,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
            checkpoint_every_n_batches=args.checkpoint_every_n_batches,
            plot_every_n_batches=args.plot_every_n_batches,
            plot_n_examples=args.plot_n_examples
        )
