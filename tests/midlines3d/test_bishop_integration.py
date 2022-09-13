import matplotlib.pyplot as plt
import numpy as np
import torch

from wormlab3d.midlines3d.mf_methods import integrate_curvature
from wormlab3d.postures.natural_frame import NaturalFrame
from wormlab3d.toolkit.util import to_numpy

show_plots = True


def main():
    """
    Check that the Bishop frame integrates to the same curve using different start points
    and different integration methods.
    """
    N = 128
    psi = np.linspace(0, 2 * np.pi, N)
    mc = 3 * np.pi * np.ones(N, dtype=np.complex64) * np.exp(1j * psi)
    NF = NaturalFrame(mc)
    l = torch.tensor([NF.length], dtype=torch.float32)
    K = torch.from_numpy(np.stack([mc.real, mc.imag], axis=-1)).unsqueeze(0).to(torch.float32) * (l / (N - 1))

    def ic_wrap(X0_, T0_, M10_, idx_, algorithm_):
        return integrate_curvature(
            X0=X0_,
            T0=T0_,
            l=l,
            K=K,
            M10=M10_,
            start_idx=idx_,
            integration_algorithm=algorithm_
        )

    # Use the NaturalFrame as the reference implementation
    X_orig = torch.from_numpy(NF.X_pos)
    T_orig = torch.from_numpy(NF.T)
    M1_orig = torch.from_numpy(NF.M1)

    # Test all integration algorithms
    for algorithm in ['euler', 'midpoint', 'rk4']:
        # Build the curve from the midpoint
        mp = int((N - 1) / 2)
        X0_mp = X_orig[mp].unsqueeze(0)
        T0_mp = T_orig[mp].unsqueeze(0)
        M10_mp = M1_orig[mp].unsqueeze(0)
        X_mp, T_mp, M1_mp = ic_wrap(X0_mp, T0_mp, M10_mp, mp, algorithm)

        # Integrate again from the head and tail
        X0_h = X_mp[:, 0]
        T0_h = T_mp[:, 0]
        M10_h = M1_mp[:, 0]
        X_h, T_h, M1_h = ic_wrap(X0_h, T0_h, M10_h, 0, algorithm)

        X0_t = X_mp[:, -1]
        T0_t = T_mp[:, -1]
        M10_t = M1_mp[:, -1]
        X_t, T_t, M1_t = ic_wrap(X0_t, T0_t, M10_t, N - 1, algorithm)

        # Calculate errors
        LXom = ((X_orig - X_mp)**2).sum()
        LXoh = ((X_orig - X_h)**2).sum()
        LXot = ((X_orig - X_t)**2).sum()
        LXmh = ((X_mp - X_h)**2).sum()
        LXmt = ((X_mp - X_t)**2).sum()
        LXht = ((X_h - X_t)**2).sum()
        LTmh = ((T_mp - T_h)**2).sum()
        LTmt = ((T_mp - T_t)**2).sum()
        LTht = ((T_h - T_t)**2).sum()
        LM1mh = ((M1_mp - M1_h)**2).sum()
        LM1mt = ((M1_mp - M1_t)**2).sum()
        LM1ht = ((M1_h - M1_t)**2).sum()

        print(f'\n==== Algorithm = {algorithm} ====')
        # print('LXom', LXom)
        # print('LXoh', LXoh)
        # print('LXot', LXot)
        print('LXmh', LXmh)
        print('LXmt', LXmt)
        print('LXht', LXht)
        print('LTmh', LTmh)
        print('LTmt', LTmt)
        print('LTht', LTht)
        print('LM1mh', LM1mh)
        print('LM1mt', LM1mt)
        print('LM1ht', LM1ht)

        if show_plots:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(projection='3d')
            ax.set_title(f'Algorithm={algorithm}')
            # ax.scatter(*X_orig.T, label='orig')
            ax.scatter(*to_numpy(X_mp[0]).T, label='mp')
            ax.scatter(*to_numpy(X_h[0]).T, label='h')
            ax.scatter(*to_numpy(X_t[0]).T, label='t')
            ax.legend()
            plt.show()


if __name__ == '__main__':
    # from simple_worm.plot3d import interactive
    # interactive()
    main()
