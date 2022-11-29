import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve
)

import kinematic_constraints
import dynamics_constraints

importlib.reload(kinematic_constraints)
importlib.reload(dynamics_constraints)
from kinematic_constraints import (
    AddFinalLandingPositionConstraint
)
from dynamics_constraints import (
  AddCollocationConstraints,
)


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

    # TODO: Add getter/setter functions for num positions/velocities/actuators

    def num_positions(self):
        return 7
    def num_velocities(self):
        return 7
    def num_actuators(self):
        return 1

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

        # x_dot = np.zeros(x.shape)
        # print(x[7:])
        # x_dot[:7] = x[7:]
        # x_dot[10] = u
        # x_dot[11:] = a_com

        x_dot = x[7:]
        x_dot = np.append(x_dot, 0)
        x_dot = np.append(x_dot, 0)
        x_dot = np.append(x_dot, 0)
        x_dot = np.append(x_dot, u)
        x_dot = np.append(x_dot, a_com)

        return x_dot

    def step(self, x, u, dt):
        self.x += self.f(x, u) * dt

if __name__ == "__main__":
    # Instantiate robot
    robot = Robot()

    # init constants
    N = 5  # number of knot points
    tf = 3.0  # last time
    n_q = robot.num_positions()
    n_v = robot.num_velocities()
    n_x = n_q + n_v
    n_u = robot.num_actuators()
    # TODO: define initial state and jump distance - rn it is ball throwing distance but i think it is the same
    # values are currently taken straight from HW5 main method at the bottom of find_throwing_trajectory.py
    initial_state = np.zeros(n_x)
    distance = 15.0
    final_configuration = np.array(
            [
                10.0,  # Foot x-position (global frame)
                0.0,  # Foot y-position (global frame)
                0.0,  # Foot z-position (global frame)
                0.0,  # Actuator length
                10.0,  # CoM x-position (global frame)
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

    # init drake mathematical program
    prog = MathematicalProgram()
    x = np.zeros((N, n_x), dtype="object")
    u = np.zeros((N, n_u), dtype="object")
    for i in range(N):
        x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
        u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

    t_land = prog.NewContinuousVariables(1, "t_land")

    t0 = 0.0
    timesteps = np.linspace(t0, tf, N)
    x0 = x[0]
    xf = x[-1]

    # Add constraints
    # Add the kinematic constraints (initial state, final state)
    prog.AddLinearEqualityConstraint(x0, initial_state)

    # Add the kinematic constraint on the final state
    AddFinalLandingPositionConstraint(prog, xf, distance, t_land)

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, robot, N, x, u, timesteps)

    # want to minimize u, quadratic cost
    g = 0
    for i in range(N - 1):
        g += 0.5 * (timesteps[1] - timesteps[0]) * (u[i].T @ u[i] + u[i + 1].T @ u[i + 1])
    prog.AddQuadraticCost(g)

    # TODO: Add bounding box constraints on the inputs and qdot - fake rn
    # ub = np.ones((N, n_q)) * joint_limits
    # lb = -ub
    # ubVel = np.ones((N, n_q)) * vel_limits
    # lbVel = -ubVel
    # prog.AddBoundingBoxConstraint(lb, ub, x[:, 0:n_q])
    # prog.AddBoundingBoxConstraint(lbVel, ubVel, x[:, n_q:N])
    # ubU = np.ones((N, n_u)) * effort_limits
    # lbU = -ubU
    # prog.AddBoundingBoxConstraint(lbU, ubU, u)

    # give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
    guess_u = np.random.rand(N, n_u)
    guess_x = np.zeros((N, n_x))
    guess_x[:, 0] = np.linspace(initial_state[0], final_configuration[0], N)
    guess_x[:, 1] = np.linspace(initial_state[1], final_configuration[1], N)

    print(guess_x)
    prog.SetInitialGuess(x, guess_x)  # guess x
    prog.SetInitialGuess(u, guess_u)  # guess x

    # prog.SetInitialGuess(u, effort_limits * guess_u - 0.5 * np.ones((N, 2)))  # guess u with effort limits

    # Set up solver
    # print(prog)
    result = Solve(prog)

    x_sol = result.GetSolution(x)
    u_sol = result.GetSolution(u)
    t_land_sol = result.GetSolution(t_land)

    print("RESULTS:")

    print('optimal cost: ', result.get_optimal_cost())
    print('x_sol: ', x_sol)
    print('u_sol: ', u_sol)
    print('t_land: ', t_land_sol)

    print(result.get_solution_result())

    # TODO: Reconstruct the trajectory
    # xdot_sol = np.zeros(x_sol.shape)
    # for i in range(N):
    #     xdot_sol[i] = EvaluateDynamics(plant, plant_context, x_sol[i], u_sol[i])
    #
    # x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, xdot_sol.T)
    # u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)


    # pre-drake but potentially useful/necessary

    # # Simulation parameters
    # dt = 0.0001
    # t = 0
    # t_max = 10
    #
    # # Graphing parameters
    # robot_state_history = np.reshape(robot.x_0, (14, 1))
    # t_history = [0]
    #
    # # Run simulation
    # while t < t_max:
    #     if 2 <= t <= 5:
    #         u = 1
    #     else:
    #         u = 0
    #     robot.step(x=robot.x, u=u, dt=dt)
    #     robot_state_history = np.hstack(
    #         (robot_state_history, np.reshape(robot.x, (14, 1)))
    #     )
    #     t += dt
    #     t_history.append(t)
    #
    # # Plot
    # plt.figure()
    # plt.plot(robot_state_history[4, :], robot_state_history[6, :])
    # plt.xlabel("x-position")
    # plt.ylabel("z-position")
    # plt.show()
    #
    # plt.figure()
    # plt.plot(t_history, robot_state_history[6, :])
    # plt.xlabel("time")
    # plt.ylabel("z-position")
    # plt.show()
