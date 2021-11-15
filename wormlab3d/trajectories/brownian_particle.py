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

        # Current position (for momentum etc)
        self.x = x0

        # Diffusion coefficient
        self.D = D

        # Spatial dimension
        self.m = self.x0.shape[0]

    def generate_trajectory(self, n_steps: int, total_time: float) -> np.ndarray:
        """
        Generate a random trajectory as a Wiener process.
        """
        dt = total_time / float(n_steps - 1)
        x = np.zeros([n_steps, self.m])
        x[0] = self.x0

        for j in range(1, n_steps):
            step = self._step(dt)
            x[j] = x[j - 1] + step
            self.x = x[j]

        return x

    def _step(self, dt: float) -> np.ndarray:
        """
        Generate a step in a random direction.
        """

        # S is the step size
        s = np.sqrt(2.0 * self.m * self.D * dt) * np.random.randn(1)

        # Direction is random
        if self.m == 1:
            step = s * np.ones(1)
        else:
            dx = np.random.randn(self.m)
            norm_dx = np.linalg.norm(dx)
            step = s * dx / norm_dx

        return step


class ActiveParticle(BrownianParticle):
    def __init__(
            self,
            x0: np.ndarray = np.array([0, 0, 0]),
            D: float = 10.,
            momentum: float = 0.9
    ):
        super().__init__(x0, D)
        self.momentum = momentum

        # Update direction angles
        self.theta = 0
        self.phi = 0

        # Convert momentum into a variance value
        if 0 < self.momentum < 1:
            self.delta_angles_sigma = -np.log(self.momentum)
        else:
            self.delta_angles_sigma = 0

    def generate_trajectory(self, n_steps: int, total_time: float) -> np.ndarray:
        """
        Sample initial theta and phi angles at random and then update using delta-angles drawn from normal distribution.
        """
        self.theta = np.random.rand() * 2 * np.pi
        self.phi = np.random.rand() * np.pi

        return super().generate_trajectory(n_steps, total_time)

    def _step(self, dt: float) -> np.ndarray:
        """
        Generate a step in a random direction.
        """

        # Sample random step size
        r = np.sqrt(2.0 * self.m * self.D * dt) * np.abs(np.random.randn())

        # Update angles
        if self.momentum > 0:
            self.theta += np.random.normal(scale=self.delta_angles_sigma)
            self.phi += np.random.normal(scale=self.delta_angles_sigma)
        else:
            self.theta = np.random.rand() * 2 * np.pi
            self.phi = np.random.rand() * np.pi

        # Convert vector update to cartesian coordinates
        dx = np.array([
            r * np.cos(self.theta) * np.sin(self.phi),
            r * np.sin(self.theta) * np.sin(self.phi),
            r * np.cos(self.phi),
        ])

        return dx


class BoundedParticle(ActiveParticle):
    def __init__(
            self,
            x0: np.ndarray = np.array([0, 0, 0]),
            D: float = 10.,
            momentum: float = 0,
            bounds: np.ndarray = None,
    ):
        super().__init__(x0, D, momentum)
        if bounds is None:
            bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
        self.bounds = bounds

    def _step(self, dt: float) -> np.ndarray:
        """
        Generate the random step and then ensure it stays within bounds.
        """
        dx = super()._step(dt)

        # Check if the particle tries to escape the confines
        xj = self.x + dx
        dx2 = dx.copy()
        reflected = False
        for k in range(self.m):
            if not (self.bounds[k][0] < xj[k] < self.bounds[k][1]):
                dx2[k] = -dx[k]
                reflected = True

        # Update the angles if the particle was reflected back
        if reflected:
            r = np.linalg.norm(dx2)
            self.theta = np.arctan2(dx2[1], dx2[0])
            self.phi = np.arccos(dx2[2] / r)

        return dx2


class ConfinedParticle(BoundedParticle):
    def __init__(
            self,
            x0: np.ndarray = np.array([0, 0, 0]),
            D: float = 10.,
            momentum: float = 0,
            bounds: np.ndarray = None,

            unconfined_duration_mean: float = 10.,
            unconfined_duration_variance: float = 1.,
            confined_duration_mean: float = 10.,
            confined_duration_variance: float = 1.,
            D_confined: float = 1.,
    ):
        super().__init__(x0, D, momentum, bounds)
        self.D_confined = D_confined
        self.D_unconfined = D

        self.unconfined_duration_mean = unconfined_duration_mean
        self.unconfined_duration_variance = unconfined_duration_variance
        self.confined_duration_mean = confined_duration_mean
        self.confined_duration_variance = confined_duration_variance

        self.is_confined = False
        self.T_state = 0
        self.T_target = 0

    def generate_trajectory(self, n_steps: int, total_time: float) -> np.ndarray:
        """
        Sample initial unconfined duration.
        """
        self.T_target = np.random.normal(loc=self.unconfined_duration_mean, scale=self.unconfined_duration_variance)
        return super().generate_trajectory(n_steps, total_time)

    def _step(self, dt: float) -> np.ndarray:
        """
        Randomly enter confined state with reduced diffusivity.
        """

        # Toggle state if particle has been in the state for the target duration
        if self.T_state > self.T_target:
            self.T_state = 0

            # Generate a new dwell time for this state
            if self.is_confined:
                self.is_confined = False
                mu, sig = self.unconfined_duration_mean, self.unconfined_duration_variance
            else:
                self.is_confined = True
                mu, sig = self.confined_duration_mean, self.confined_duration_variance
            self.T_target = np.random.normal(loc=mu, scale=sig)

        # Vary the diffusivity depending on confined state
        if self.is_confined:
            self.D = self.D_confined
        else:
            self.D = self.D_unconfined

        dx = super()._step(dt)
        self.T_state += dt

        return dx
