from wormlab3d.midlines3d.args_finder.parse import parse_arguments
from wormlab3d.midlines3d.midline3d_finder import Midline3DFinder


def train():
    """
    Reconstruct a 3D midline.
    """
    runtime_args, source_args, model_args, optimiser_args = parse_arguments()

    # Construct finder
    manager = Midline3DFinder(
        runtime_args=runtime_args,
        source_args=source_args,
        model_args=model_args,
        optimiser_args=optimiser_args
    )

    # Do some training
    manager.train(
        n_steps_cc=runtime_args.n_steps_cc,
        n_steps_curve=runtime_args.n_steps_curve,
    )


if __name__ == '__main__':
    train()
