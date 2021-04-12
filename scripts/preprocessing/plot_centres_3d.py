import matplotlib.pyplot as plt
import numpy as np

from wormlab3d.data.model.trial import Trial
from wormlab3d.toolkit.plot_utils import interactive_plots


def plot_centres_3d(trial_id):
    """
    Draw a plot showing how the 3D centre point changes over a trial.
    """
    interactive_plots()

    # Fetch the trial, and collate the 3d centre points over time
    trial = Trial.objects.get(id=trial_id)
    centres_3d = []
    timestamps = []
    frame_time = 0.
    frames = trial.get_frames()
    for frame in frames:
        if frame.centre_3d is not None:
            pt = frame.centre_3d.point_3d
        else:
            pt = (0, np.nan, np.nan)
        centres_3d.append(pt)
        timestamps.append(frame_time)
        frame_time += 1 / trial.fps
    centres_3d = np.stack(centres_3d)
    assert centres_3d.shape == (trial.n_frames_max, 3)

    # Plot the results
    fig, axes = plt.subplots(3, sharex=True)
    fig.suptitle(
        f'{trial.date:%Y%m%d} Trial #{trial.trial_num}. '
    )
    axes[0].set_title('x')
    axes[0].plot(timestamps, centres_3d[:, 0], label='x')
    axes[1].set_title('y')
    axes[1].plot(timestamps, centres_3d[:, 1], label='y')
    axes[2].set_title('z')
    axes[2].plot(timestamps, centres_3d[:, 2], label='z')
    axes[2].set_xlabel('Time (s)')
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    plot_centres_3d(
        trial_id=4
    )
