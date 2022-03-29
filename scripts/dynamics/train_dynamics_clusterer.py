from wormlab3d.dynamics.args.parse import parse_arguments
from wormlab3d.dynamics.manager import Manager


def train():
    """
    Trains a network to model and cluster dynamics.
    """
    runtime_args, dataset_args, net_args, optimiser_args = parse_arguments()

    # Construct manager
    manager = Manager(
        runtime_args=runtime_args,
        dataset_args=dataset_args,
        net_args=net_args,
        optimiser_args=optimiser_args,
    )

    # Generate the neural network computation graph (view in tensorboard)
    # manager.log_graph()

    # Do some training
    manager.train(
        n_epochs=runtime_args.n_epochs
    )


if __name__ == '__main__':
    train()
