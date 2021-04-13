import matplotlib.pyplot as plt
from mongoengine import DoesNotExist

from wormlab3d.midlines2d.args import DatasetMidline2DArgs
from wormlab3d.midlines2d.data_loader import get_data_loader
from wormlab3d.nn.data_loader import load_dataset


def plot_augmentations(n_examples: int = 5):
    """
    Plot a random selection of images along with their randomly augmented versions.
    """

    # interactive_plots()
    dataset_args = DatasetMidline2DArgs(
        train_test_split=0.8,
        centre_3d_max_error=0,
        exclude_trials=[258, 183, 88, 115, 86, 29, 84, 85, 218],
        blur_sigma=5,
        augment=True
    )
    try:
        ds = load_dataset(dataset_args)
    except DoesNotExist as e:
        raise RuntimeError('No suitable datasets found in database.') from e

    loader = get_data_loader(
        ds=ds,
        ds_args=dataset_args,
        train_or_test='train',
        batch_size=n_examples
    )
    iterator = loader._get_iterator()
    batch = next(iterator)

    n_rows = int((n_examples * 2)**0.5 + 0.5)
    n_cols = n_examples * 2 // n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_rows * 2, n_cols * 3),
        squeeze=False, sharex=True, sharey=True,
        gridspec_kw=dict(
            wspace=0.01, hspace=0.01,
            width_ratios=[1] * n_cols,
            top=0.95, bottom=0.05, left=0.05, right=0.95
        ),
    )
    fig.suptitle('Augmented Images')

    eg_idx = 0
    for row_idx in range(n_rows // 2):
        for col_idx in range(n_cols):
            midline = batch[2][eg_idx]

            # Show original
            ax = axes[row_idx * 2, col_idx]
            if col_idx == 0:
                ax.text(-0.1, 0.4, 'Original', transform=ax.transAxes, rotation='vertical')
            ax.imshow(midline.get_prepared_image(), vmin=0, vmax=1, cmap='gray', aspect='equal')
            ax.imshow(midline.get_segmentation_mask(blur_sigma=dataset_args.blur_sigma), vmin=0, vmax=1, cmap='jet',
                      alpha=0.2, aspect='equal')
            X = midline.get_prepared_coordinates()
            ax.scatter(x=X[:, 0], y=X[:, 1], color='red', s=10, alpha=0.8)
            ax.axis('off')

            # Show augmented
            ax = axes[row_idx * 2 + 1, col_idx]
            if col_idx == 0:
                ax.text(-0.1, 0.3, 'Augmented', transform=ax.transAxes, rotation='vertical')
            ax.imshow(batch[0][eg_idx].squeeze(), vmin=0, vmax=1, cmap='gray', aspect='equal')
            ax.imshow(batch[1][eg_idx].squeeze(), vmin=0, vmax=1, cmap='jet', alpha=0.3, aspect='equal')
            ax.axis('off')

            eg_idx += 1

    plt.show()


if __name__ == '__main__':
    plot_augmentations(n_examples=16)
