import os

import matplotlib.pyplot as plt
import torch

from wormlab3d import LOGS_PATH, START_TIMESTAMP
from wormlab3d.particles.sdbn_explorer import SDBNExplorer
from wormlab3d.particles.util import plot_3d_trajectories, plot_states, plot_2d_trajectory, plot_3d_trajectory

plot_n_examples = 1
show_plots = True
save_plots = True
img_extension = 'png'


def simulate():
    pe = SDBNExplorer(
        depth=3,
        batch_size=20,
        transition_rates=[
            torch.tensor([0.01, 0.01]),
            torch.tensor([[0.02, 0.1], [0.1, 0.02]]),
            torch.tensor([[[0.1, 0.9], [0.1, 0.2]], [[0.5, 0.1], [0.2, 0.1]]]),
        ],
        state_parameters={
            '000': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.01, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 1.0, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 1.0, ],
                },
            },
            '001': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.05, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.1, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.1, ],
                },
            },
            '010': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.02, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.2, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.2, ],
                },
            },
            '011': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.1, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.05, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.02, ],
                },
            },
            '100': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.02, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.1, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.01, ],
                },
            },
            '101': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.05, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.1, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.01, ],
                },
            },
            '110': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.02, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.2, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.01, ],
                },
            },
            '111': {
                'speeds': {
                    'dist': 'lognorm',
                    'params': [0, 0.2, ],
                },
                'planar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.01, ],
                },
                'nonplanar_angles': {
                    'dist': 'cauchy',
                    'params': [0, 0.001, ],
                },
            },
        }
    )
    pe = torch.jit.script(pe)

    T = 10000
    dt = 0.1
    ts, Xs, states, speeds, planar_angles, nonplanar_angles = pe.forward(T, dt)

    for i in range(plot_n_examples):
        title = f'Simulation run {i}.'
        plot_states(ts, states[i], speeds[i], planar_angles[i], nonplanar_angles[i], title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_states_sim_{i}.{img_extension}')

        plot_2d_trajectory(ts, Xs[i], title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_2d_sim_{i}.{img_extension}')

        plot_3d_trajectory(Xs[i], title)
        if save_plots:
            plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectory_3d_sim_{i}.{img_extension}')

    plot_3d_trajectories(Xs_sim=Xs)
    if save_plots:
        plt.savefig(LOGS_PATH / f'{START_TIMESTAMP}_trajectories_3d_all.{img_extension}')

    if show_plots:
        plt.show()


if __name__ == '__main__':
    if save_plots:
        os.makedirs(LOGS_PATH, exist_ok=True)

    # from simple_worm.plot3d import interactive
    # interactive()
    simulate()
