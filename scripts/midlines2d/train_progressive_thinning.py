import numpy as np

from wormlab3d import logger
from wormlab3d.midlines2d.args.parse import parse_arguments
from wormlab3d.midlines2d.manager import Manager


def train_progressive_thinning(
        n_sigmas: int = 5,
        sigmas_min: float = 0,
        sigmas_max: float = 20,
):
    """
    Trains a network in a number of stages starting with a high initial blur_sigma (ie, fatter targets)
    and dropping down to a low sigma (ie, thin midlines) over time.
    """
    dataset_args, net_args, runtime_args, optimiser_args = parse_arguments()
    blur_sigmas = np.linspace(sigmas_max, sigmas_min, num=n_sigmas)
    n_epochs_per_sigma = runtime_args.n_epochs // n_sigmas
    logger.info(
        f'Train with progressive target thinning. '
        f'Blur sigmas = {blur_sigmas}. '
        f'Num epochs per sigma = {n_epochs_per_sigma}.'
    )
    dataset_args.blur_sigma = blur_sigmas[0]

    # Construct manager
    manager = Manager(
        dataset_args=dataset_args,
        net_args=net_args,
        optimiser_args=optimiser_args,
        runtime_args=runtime_args
    )

    # Generate the neural network computation graph (view in tensorboard)
    # manager.log_graph()

    # Reduce the blur sigma every few epochs
    for i in range(n_sigmas):
        logger.info(f'Setting blur_sigma={blur_sigmas[i]:.1f}')
        manager.dataset_args.blur_sigma = blur_sigmas[i]
        manager.checkpoint.dataset_args['blur_sigma'] = blur_sigmas[i]
        manager.train_loader, manager.test_loader = manager._init_data_loaders()
        manager.train(n_epochs=n_epochs_per_sigma)


if __name__ == '__main__':
    train_progressive_thinning(
        n_sigmas=4,
        sigmas_min=0,
        sigmas_max=20,
    )
