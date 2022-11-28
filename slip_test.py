import numpy as np
import matplotlib.pyplot as plt


class Robot:
    def __init__(self):
        # Constants

        self.L_0 = 1.0  # original spring length
        self.m = 80  # mass
        self.g = 9.8  # gravity
        # self.k = 10.7 * self.m * self.g / self.L_0  # spring constant
        self.k = 100

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                self.L_0,  # r
                0,  # theta
                0,  # r_dot
                0,  # theta_dot
            ]
        )
        # State
        self.x = self.x_0

    def f(self, x, u):
        """
        x_dot = f(x, u)
        """
        r_ddot = (
            x[0] * x[3] ** 2
            - self.g * np.cos(x[1])
            + self.k * (self.L_0 - x[0]) / self.m
        )
        theta_ddot = -2 * x[2] * x[3] / x[0] + self.g * np.sin(x[0]) / x[0]

        return np.array([x[2], x[3], r_ddot, theta_ddot])

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt


if __name__ == "__main__":
    # Instantiate robot
    robot = Robot()

    # Simulation parameters
    dt = 0.01
    t = 0
    t_max = 10

    # Graphing parameters
    robot_state_history = np.reshape(robot.x_0, (4, 1))
    t_history = []
    t_history.append(0)

    # Run simulation
    while t < t_max:
        u = None
        robot.step(x=robot.x, u=u, dt=dt)
        robot_state_history = np.hstack(
            (robot_state_history, np.reshape(robot.x, (4, 1)))
        )
        t += dt
        t_history.append(t)

    # Plot
    plt.figure()
    # plt.plot(
    #     -robot_state_history[0, :] * np.sin(robot_state_history[1, :]),
    #     robot_state_history[0, :] * np.cos(robot_state_history[1, :]),
    # )
    plt.plot(
        t_history,
        robot_state_history[0, :] * np.cos(robot_state_history[1, :]),
    )
    plt.xlabel("x-position")
    plt.ylabel("z-position")
    # plt.xlim(-6, 6)
    # plt.ylim(-6, 6)
    plt.legend()
    plt.show()
