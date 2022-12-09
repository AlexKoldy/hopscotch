import numpy as np
import matplotlib.pyplot as plt


class Robot:
    def __init__(self):
        # Constants
        self.k_1 = 10  # spring 1 constant
        self.k_2 = 0  # spring 2 constant
        self.L = 10  # original spring length
        self.m = 1  # mass
        self.g = np.array([0, 0, -9.8])  # gravity

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                0.0,  # x-position
                0,  # y-position
                5,  # z-position
                0,  # theta_1
                0,  # phi_1
                0,  # theta_2
                0,  # phi_2
                0,  # x-velocity
                0,  # y-velocity
                0,  # z-velocity
                0,  # theta_1 angular velocity
                0,  # phi_1 angular velocity
                0,  # theta_2 angular velocity
                0,  # phi_2 angular velocity
            ]
        )

        # State
        self.x = self.x_0

    def f(self, x, u):
        """
        x_dot = f(x, u)
        """
        # Calculate accelerations
        pos_ddot = (
            self.k_1
            * (self.L - np.sqrt(self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2))
            * np.array([np.cos(x[4]), np.sin(x[4]), np.cos(x[3])])
            + self.k_2
            * (self.L - np.sqrt(self.x[0] ** 2 + self.x[1] ** 2 + self.x[2] ** 2))
            * np.array([np.cos(x[6]), np.sin(x[6]), np.cos(x[5])])
            + self.m * self.g
        )

        # Calculate state derivative
        x_dot = np.zeros(x.shape)
        x_dot[:7] = x[7:]
        x_dot[7:10] = pos_ddot
        x_dot[10:] = u

        return x_dot

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt


if __name__ == "__main__":
    # Instantiate robot
    robot = Robot()

    # Simulation parameters
    dt = 0.01
    t = 0
    t_max = 1

    # Graphing parameters
    robot_state_history = np.reshape(robot.x_0, (14, 1))
    t_history = []
    t_history.append(0)

    # Run simulation
    while t < t_max:
        u = np.zeros((4,))
        robot.step(x=robot.x, u=u, dt=dt)
        robot_state_history = np.hstack(
            (robot_state_history, np.reshape(robot.x, (14, 1)))
        )
        t += dt
        t_history.append(t)

    # Plot
    plt.figure()
    plt.plot(robot_state_history[0, :], robot_state_history[2, :])
    plt.xlabel("x-position")
    plt.ylabel("z-position")
    # plt.xlim(-6, 6)
    # plt.ylim(-6, 6)
    plt.legend()
    plt.show()
