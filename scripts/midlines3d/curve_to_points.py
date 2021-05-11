import matplotlib.pyplot as plt
import numpy as np
import torch

from wormlab3d import N_WORM_POINTS

N_BASIS_FNS = 5


def curve_to_points(parameters: torch.Tensor) -> torch.Tensor:
    device = parameters.device
    bs = parameters.shape[0]

    # Extract amplitudes, frequencies and phases
    A_n = parameters[:, :, :N_BASIS_FNS]
    p_n = parameters[:, :, -N_BASIS_FNS:]
    w_n = torch.tensor([1 / 4, 1 / 2, 1, 2, 4]) * 2 * np.pi

    # Everything should be positive
    A_n = torch.exp(A_n)
    p_n = torch.exp(p_n)

    t = torch.linspace(0, 1, N_WORM_POINTS, device=device)
    s = (A_n.unsqueeze(-1) * torch.cos(torch.einsum('n,t->nt', w_n, t) + p_n.unsqueeze(-1))).sum(dim=2)

    return s


def plot_fourier_curve():
    from wormlab3d.toolkit.plot_utils import interactive_plots
    interactive_plots()

    A_n = torch.log(torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0.5, 0, 0, 0.5],
    ]))
    p_n = torch.log(torch.tensor([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]))

    parameters = torch.stack([
        torch.cat([A_n, p_n], dim=1)
    ])

    curve = curve_to_points(parameters)[0].numpy()

    fig, axes = plt.subplots(3)
    ts = np.linspace(0, 1, N_WORM_POINTS)

    for c in range(3):
        ax = axes[c]
        ax.plot(ts, curve[c])
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(curve[0], curve[1], curve[2])
    plt.show()


if __name__ == '__main__':
    plot_fourier_curve()
