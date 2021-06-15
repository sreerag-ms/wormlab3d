from wormlab3d.simple_worm.args.parse import parse_arguments
from wormlab3d.simple_worm.manager import Manager


def optimise():
    """
    Run the inverse optimisation.
    """
    runtime_args, frame_sequence_args, simulation_args, optimiser_args, regularisation_args = parse_arguments()

    # Construct manager
    manager = Manager(
        runtime_args=runtime_args,
        frame_sequence_args=frame_sequence_args,
        simulation_args=simulation_args,
        optimiser_args=optimiser_args,
        regularisation_args=regularisation_args
    )

    # Do some training
    manager.train(
        n_steps=runtime_args.n_steps
    )


if __name__ == '__main__':
    optimise()
