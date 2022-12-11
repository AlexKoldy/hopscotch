from pydrake.autodiffutils import AutoDiffXd
import numpy as np
import matplotlib.pyplot as plt


class ASLIP:
    def __init__(self):
        # Constants
        self.L_0 = 0.5  # original spring length
        self.m = 54.0  # mass
        self.g = -9.8  # gravity
        self.k = 10.7 * self.m * -self.g / self.L_0
        # self.k = 100000

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                0.0,  # Foot x-position (global frame)
                0.0,  # Foot y-position (global frame)
                0.0,  # Foot z-position (global frame)
                0.0,  # Actuator length
                0.1,  # CoM x-position (global frame)
                0.0,  # CoM y-position (global frame)
                0.8,  # CoM z-position (global frame)
                0.0,  # Foot x-velocity (global frame)
                0.0,  # Foot y-velocity (global frame)
                0.0,  # Foot z-velocity (global frame)
                0.0,  # Actuator velocity
                0.0,  # CoM x-velocity (global frame)
                0.0,  # CoM y-velocity (global frame)
                0.0,  # CoM z-velocity (global frame)
            ]
        )
        # State
        self.x = self.x_0

    def f(self, x, u):
        """
        x_dot = f(x, u)
        """
        p_foot = x[:3]
        q = x[3]
        p_com = x[4:7]

        d = np.sqrt(
            (p_com[0] - p_foot[0]) ** 2
            + (p_com[1] - p_foot[1]) ** 2
            + (p_com[2] - p_foot[2]) ** 2
        )

        F_s = (
            self.k
            * (self.L_0 - d + q)
            * np.array(
                [
                    (p_com[0] - p_foot[0]) / d,
                    (p_com[1] - p_foot[1]) / d,
                    (p_com[2] - p_foot[2]) / d,
                ],
                dtype=AutoDiffXd,
            )
        )

        a_com = (F_s / self.m) + np.array([0, 0, self.g])
        x_dot = np.zeros(x.shape, dtype=AutoDiffXd)
        x_dot[:7] = x[7:]
        x_dot[10] = u
        x_dot[11:] = a_com

        return x_dot

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt


# if __name__ == "__main__":
#     # Instantiate robot
#     robot = ASLIP()

#     # Simulation parameters
#     dt = 0.0001
#     t = 0
#     t_max = 2

#     # Graphing parameters
#     robot_state_history = np.reshape(robot.x_0, (14, 1))
#     t_history = []
#     t_history.append(0)

#     # Run simulation
#     while t < t_max:
#         if t < 2:
#             u = 0
#         elif 2 <= t and t <= 5:
#             u = 1
#         elif 5 < t:
#             u = 0
#         robot.step(x=robot.x, u=u, dt=dt)
#         robot_state_history = np.hstack(
#             (robot_state_history, np.reshape(robot.x, (14, 1)))
#         )
#         t += dt
#         t_history.append(t)

#     # Plot
#     plt.figure()
#     plt.plot(robot_state_history[4, :], robot_state_history[6, :])
#     plt.xlabel("x-position")
#     plt.ylabel("z-position")
#     plt.show()

#     plt.figure()
#     plt.plot(t_history, robot_state_history[6, :])
#     plt.xlabel("time")
#     plt.ylabel("z-position")
#     plt.show()
