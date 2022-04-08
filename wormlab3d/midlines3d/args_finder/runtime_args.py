from argparse import ArgumentParser

from wormlab3d.nn.args.base_args import BaseArgs
from wormlab3d.toolkit.util import str2bool


class RuntimeArgs(BaseArgs):
    def __init__(
            self,
            resume: bool = True,
            resume_from: str = 'latest',
            copy_state: str = None,
            fix_mode: bool = False,
            fix_decay_rate: float = 10.,
            gpu_only: bool = False,
            cpu_only: bool = False,
            gpu_id: int = 0,
            log_level: int = 0,
            plot_every_n_steps: int = -1,
            plot_every_n_init_steps: int = -1,
            plot_every_n_frames: int = -1,
            plot_n_examples: int = 1,
            plot_sigmas: bool = True,
            plot_exponents: bool = True,
            plot_intensities: bool = True,
            plot_scores: bool = True,
            save_plots: bool = True,
            seed: int = None,
            **kwargs
    ):
        self.resume = resume
        self.resume_from = resume_from
        self.copy_state = copy_state
        self.fix_mode = fix_mode
        self.fix_decay_rate = fix_decay_rate
        self.gpu_only = gpu_only
        self.cpu_only = cpu_only
        self.gpu_id = gpu_id
        self.log_level = log_level
        self.plot_every_n_steps = plot_every_n_steps
        self.plot_every_n_init_steps = plot_every_n_init_steps
        self.plot_every_n_frames = plot_every_n_frames
        self.plot_n_examples = plot_n_examples
        self.plot_sigmas = plot_sigmas
        self.plot_exponents = plot_exponents
        self.plot_intensities = plot_intensities
        self.plot_scores = plot_scores
        self.save_plots = save_plots

        # Generate a random seed if one was not provided.
        if seed is None:
            import os
            rand = os.urandom(4)
            seed = int.from_bytes(rand, byteorder='big')
        self.seed = seed

    @classmethod
    def add_args(cls, parser: ArgumentParser):
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
                           help='Resume from a specific checkpoint id, "latest" or "best". Default="latest".')
        group.add_argument('--copy-state', type=str, default=None,
                           help='Copy trial state from an existing reconstruction id.')
        group.add_argument('--fix-mode', type=str2bool, default=False,
                           help='Run in fix-mode. Requires --resume and --end-frame to be set.')
        group.add_argument('--fix-decay-rate', type=float, default=10.,
                           help='Target frame loss ratio decay rate. Used only in fix-mode.')
        group.add_argument('--gpu-only', action='store_true',
                           help='Abort if no gpus are detected.')
        group.add_argument('--cpu-only', action='store_true',
                           help='Only run on CPU. Otherwise will use GPU if available.')
        group.add_argument('--gpu-id', type=int, default=0,
                           help='GPU id to use if using GPUs.')
        group.add_argument('--plot-every-n-steps', type=int, default=-1,
                           help='Plot example inputs and outputs every n steps, -1 turns this off.')
        group.add_argument('--plot-every-n-init-steps', type=int, default=-1,
                           help='Plot example inputs and outputs every n init steps, -1 turns this off.')
        group.add_argument('--log-level', type=int, default=2, choices=[0, 1, 2],
                           help='Tensorboard logging level. 0=Totals only. 1=Depth losses. 2=All losses and parameter stats.')
        group.add_argument('--plot-every-n-frames', type=int, default=-1,
                           help='Plot example inputs and outputs every n frames, -1 turns this off.')
        group.add_argument('--plot-n-examples', type=int, default=1,
                           help='Plot this number of examples from the batch at each iteration.')
        group.add_argument('--plot-sigmas', type=str2bool, default=True,
                           help='Plot sigmas.')
        group.add_argument('--plot-exponents', type=str2bool, default=True,
                           help='Plot exponents.')
        group.add_argument('--plot-intensities', type=str2bool, default=True,
                           help='Plot intensities.')
        group.add_argument('--plot-scores', type=str2bool, default=True,
                           help='Plot scores.')
        group.add_argument('--save-plots', type=str2bool, default=True,
                           help='Save plot images to disk. Default = True.')
        group.add_argument('--seed', type=int,
                           help='Set a random seed.')
