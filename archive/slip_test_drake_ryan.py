import numpy as np
import matplotlib.pyplot as plt
from pydrake.solvers.mathematicalprogram import MathematicalProgram


class Robot:
    def __init__(self):
        # Constants

        self.L_0 = 10.0  # original spring length
        self.m = 1  # mass
        self.g = -9.8  # gravity
        self.k = 10

        # Initial state [q, q_dot]
        self.x_0 = np.array(
            [
                0.0,  # Foot x-position (global frame)
                0.0,  # Foot y-position (global frame)
                0.0,  # Foot z-position (global frame)
                0.0,  # Actuator length
                0.0,  # CoM x-position (global frame)
                0.0,  # CoM y-position (global frame)
                10.0,  # CoM z-position (global frame)
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
            ]
        )
        )

        # print((p_com[0] - p_foot[0]) / d)
        # print((p_com[1] - p_foot[1]) / d)
        # print((p_com[2] - p_foot[2]) / d)

        a_com = (F_s / self.m) + np.array([0, 0, self.g])
        # print(d)

        x_dot = np.zeros(x.shape)
        x_dot[:7] = x[7:]
        x_dot[10] = u
        x_dot[11:] = a_com

        return x_dot

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt

    def CollocationConstraintEvaluator(planar_arm, context, dt, x_i, u_i, x_ip1, u_ip1):
        n_x = planar_arm.num_positions() + planar_arm.num_velocities()
        h_i = np.zeros(n_x, )
        # TODO: Add f_i and f_ip1
        # You should make use of the EvaluateDynamics() function to compute f(x,u)
        s0 = x_i
        f_i = robot.f(x_i, u_i)
        s1 = f_i
        f_ip1 = robot.f(x_ip1, u_ip1)
        s2 = (3 * x_ip1 - 3 * x_i - 2 * f_i * dt - f_ip1 * dt) / (dt * dt)
        s3 = (-2 * x_ip1 + 2 * x_i + f_i * dt + f_ip1 * dt) / (dt * dt * dt)

        s_midpoint = (1 / 2) * (x_i + x_ip1) - (dt / 8) * (f_ip1 - f_i)
        sdot_midpoint = 3 / (2 * dt) * (x_ip1 - x_i) - 0.25 * (f_i + f_ip1)

        h_i = sdot_midpoint - robot.f(s_midpoint, (u_i + u_ip1) * 0.5)

        return h_i

    def AddCollocationConstraints(prog, planar_arm, context, N, x, u, timesteps):
        n_u = planar_arm.num_actuators()
        n_x = planar_arm.num_positions() + planar_arm.num_velocities()

        for i in range(N - 1):
            def CollocationConstraintHelper(vars):
                x_i = vars[:n_x]
                u_i = vars[n_x:n_x + n_u]
                x_ip1 = vars[n_x + n_u: 2 * n_x + n_u]
                u_ip1 = vars[-n_u:]
                return robot.CollocationConstraintEvaluator(planar_arm, context, timesteps[i + 1] - timesteps[i], x_i, u_i,
                                                      x_ip1,
                                                      u_ip1)

            # TODO: Within this loop add the dynamics constraints for segment i (aka collocation constraints)
            #       to prog
            # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
            # where vars = hstack(x[i], u[i], ...)
            lb = np.zeros(n_x)
            ub = lb
            vars = np.hstack((x[i], u[i], x[i + 1], u[i + 1]))
            prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)


if __name__ == "__main__":
    # Instantiate robot
    robot = Robot()

    prog = MathematicalProgram()

    # Simulation parameters
    dt = 0.0001
    t = 0
    t_max = 10

    # Graphing parameters
    robot_state_history = np.reshape(robot.x_0, (14, 1))
    t_history = []
    t_history.append(0)

    # Run simulation
    while t < t_max:
        if t < 2:
            u = 0
        elif 2 <= t and t <= 5:
            u = 1
        elif 5 < t:
            u = 0
        robot.step(x=robot.x, u=u, dt=dt)
        robot_state_history = np.hstack(
            (robot_state_history, np.reshape(robot.x, (14, 1)))
        )
        t += dt
        t_history.append(t)

    # Plot
    plt.figure()
    plt.plot(robot_state_history[4, :], robot_state_history[6, :])
    plt.xlabel("x-position")
    plt.ylabel("z-position")
    plt.show()

    plt.figure()
    plt.plot(t_history, robot_state_history[6, :])
    plt.xlabel("time")
    plt.ylabel("z-position")
    plt.show()
