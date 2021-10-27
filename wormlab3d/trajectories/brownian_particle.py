import numpy as np


class BrownianParticle:
    """
    Brownian particle trajectory simulator.
    Adapted from: https://people.sc.fsu.edu/~jburkardt/py_src/brownian_motion_simulation/brownian_motion_simulation.html
    """

    def __init__(
            self,
            x0: np.ndarray = np.array([0, 0, 0]),
            D: float = 10.
    ):
        # Starting position
        self.x0 = x0

        # Diffusion coefficient
        self.D = D

    def generate_trajectory(self, n_steps: int, total_time: float) -> np.ndarray:
        """
        Generate a random trajectory as a Wiener process.
        """
        dt = total_time / float(n_steps - 1)
        m = self.x0.shape[0]
        x = np.zeros([n_steps, m])
        x[0] = self.x0

        for j in range(1, n_steps):
            # S is the step size
            s = np.sqrt(2.0 * m * self.D * dt) * np.random.randn(1)

            # Direction is random
            if m == 1:
                step = s * np.ones(1)
            else:
                dx = np.random.randn(m)
                norm_dx = np.linalg.norm(dx)
                step = s * dx / norm_dx

            x[j] = x[j - 1] + step

        return x
