import matplotlib.pyplot as plt

from wormlab3d.data.model.midline2d import Midline2D
from wormlab3d.toolkit.plot_utils import interactive_plots


def plot_2d_midline_annotation(midline2d_id: str = None):
    """
    Plot a 2d midline annotation
    """
    interactive_plots()
    midline = Midline2D.objects.get(id=midline2d_id)
    trial = midline.frame.trial
    image = midline.get_image()

    fig, ax = plt.subplots(1)
    ax.set_title(
        f'{trial.date:%Y%m%d} #{trial.trial_num}. '
        f'Video: {trial.videos[midline.camera]}. '
        f'Frame: {midline.frame.frame_num}'
    )
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.scatter(x=midline.X[:, 0], y=midline.X[:, 1], color='red', s=10, alpha=0.8)
    plt.show()


if __name__ == '__main__':
    plot_2d_midline_annotation(midline2d_id='605dbb1ff1f34c19b7ab7422')
