from wormlab3d.midlines3d.args.parse import parse_arguments
from wormlab3d.midlines3d.manager_rotae import ManagerRotAE


def train():
    """
    Trains a network to generate 3D midlines.
    """
    dataset_args, net_args, runtime_args, optimiser_args = parse_arguments(rotae=True)

    # Construct manager
    manager = ManagerRotAE(
        dataset_args=dataset_args,
        net_args=net_args,
        optimiser_args=optimiser_args,
        runtime_args=runtime_args
    )

    # Generate the neural network computation graph (view in tensorboard)
    # manager.log_graph()

    # Do some training
    manager.train(
        n_epochs=runtime_args.n_epochs
    )


if __name__ == '__main__':
    train()
