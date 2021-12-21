from wormlab3d.midlines3d.args_finder.parse import parse_arguments
from wormlab3d.midlines3d.midline3d_finder import Midline3DFinder


def train():
    """
    Reconstruct 3D midlines for a trial.
    """
    runtime_args, source_args, parameter_args = parse_arguments()

    # Construct finder
    manager = Midline3DFinder(
        runtime_args=runtime_args,
        source_args=source_args,
        parameter_args=parameter_args
    )

    # Process the trial
    manager.process_trial()


if __name__ == '__main__':
    train()
