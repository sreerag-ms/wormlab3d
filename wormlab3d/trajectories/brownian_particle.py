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

        # Current and previous positions (for momentum etc)
        self.x = x0
        self.x_prev = x0

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
            self.x_prev = self.x
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
        self.cone_angle = (1 - self.momentum) * np.pi

    def _step(self, dt: float) -> np.ndarray:
        """
        Sample a vector from a cone formed in the direction of travel.
        Adapted from https://stackoverflow.com/questions/38997302/create-random-unit-vector-inside-a-defined-conical-region
        """
        cone_dir = self.x - self.x_prev
        cone_dir_default = np.array([0, 0, 1])

        # If the cone direction is zeros then generate a random cone dir
        if np.allclose(cone_dir, np.zeros_like(cone_dir)):
            cone_dir = np.random.rand(3) * 2 - 1

        # Normalise cone axis
        cone_dir = cone_dir / np.linalg.norm(cone_dir)

        # Sample a random unit-length vector centred around the north pole
        theta = np.random.rand() * 2 * np.pi
        z = np.random.rand() * (1 - np.cos(self.cone_angle)) + np.cos(self.cone_angle)
        x = np.sqrt(1 - z**2) * np.cos(theta)
        y = np.sqrt(1 - z**2) * np.sin(theta)
        p = np.array([x, y, z])

        # Find the rotation axis and rotation angle
        u = np.cross(cone_dir_default, cone_dir)
        u = u / np.linalg.norm(u)
        rot = np.arccos(np.dot(cone_dir, cone_dir_default))

        # Convert rotation axis and angle to 3x3 rotation matrix
        # (See https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle)
        R = np.cos(rot) * np.eye(3) \
            + np.sin(rot) * np.cross(u, -np.eye(3)) \
            + (1 - np.cos(rot)) * (np.outer(u, u))

        # Rotate random vector
        v = np.matmul(R, p)

        # Scale by random step size
        r = np.sqrt(2.0 * self.m * self.D * dt) * np.abs(np.random.randn())
        dx = r * v

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
            unconfined_momentum: float = 0.9,
            confined_duration_mean: float = 10.,
            confined_duration_variance: float = 1.,
            confined_momentum: float = 0.9,
            D_confined: float = 1.,
    ):
        super().__init__(x0, D, momentum, bounds)
        self.D_confined = D_confined
        self.D_unconfined = D

        self.unconfined_duration_mean = unconfined_duration_mean
        self.unconfined_duration_variance = unconfined_duration_variance
        self.unconfined_momentum = unconfined_momentum
        self.confined_duration_mean = confined_duration_mean
        self.confined_duration_variance = confined_duration_variance
        self.confined_momentum = confined_momentum

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
            self.cone_angle = (1 - self.confined_momentum) * np.pi
        else:
            self.D = self.D_unconfined
            self.cone_angle = (1 - self.unconfined_momentum) * np.pi

        dx = super()._step(dt)
        self.T_state += dt

        return dx
