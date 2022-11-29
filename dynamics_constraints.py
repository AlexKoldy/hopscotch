import numpy as np

from pydrake.solvers.mathematicalprogram import MathematicalProgram
from pydrake.autodiffutils import AutoDiffXd


def CollocationConstraintEvaluator(robot, dt, x_i, u_i, x_ip1, u_ip1):
    n_x = robot.num_positions() + robot.num_velocities()
    h_i = np.zeros(n_x, )
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


def AddCollocationConstraints(prog, robot, N, x, u, timesteps):
    n_u = robot.num_actuators()
    n_x = robot.num_positions() + robot.num_velocities()

    for i in range(N - 1):
        def CollocationConstraintHelper(vars):
            x_i = vars[:n_x]
            u_i = vars[n_x:n_x + n_u]
            x_ip1 = vars[n_x + n_u: 2 * n_x + n_u]
            u_ip1 = vars[-n_u:]
            return CollocationConstraintEvaluator(robot, timesteps[i + 1] - timesteps[i], x_i, u_i,
                                                        x_ip1,
                                                        u_ip1)

        # Hint: use prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
        # where vars = hstack(x[i], u[i], ...)
        lb = np.zeros(n_x)
        ub = lb
        vars = np.hstack((x[i], u[i], x[i + 1], u[i + 1]))
        prog.AddConstraint(CollocationConstraintHelper, lb, ub, vars)
