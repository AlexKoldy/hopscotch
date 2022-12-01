import matplotlib.pyplot as plt
import numpy as np
import importlib

from pydrake.all import (
    DiagramBuilder, Simulator, FindResourceOrThrow, MultibodyPlant, PiecewisePolynomial, SceneGraph,
    Parser, JointActuatorIndex, MathematicalProgram, Solve, AutoDiffXd
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
                15.0,  # CoM z-position (global frame)
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

        d = ((p_com[0] - p_foot[0]) ** 2
             + (p_com[1] - p_foot[1]) ** 2
             + (p_com[2] - p_foot[2]) ** 2)

        F_s = (
                self.k
                * (self.L_0 - np.sqrt(d + (1e-10)) + q)
                * np.array(
            [
                (p_com[0] - p_foot[0]) / d,
                (p_com[1] - p_foot[1]) / d,
                (p_com[2] - p_foot[2]) / d,
            ]
        )
        )

        a_com = (F_s / self.m) + np.array([0, 0, self.g])

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
    N = 25  # number of knot points
    tf = 8.0  # last time
    n_q = robot.num_positions()
    n_v = robot.num_velocities()
    n_x = n_q + n_v
    n_u = robot.num_actuators()
    # TODO: define initial state and jump distance
    initial_state = robot.x_0
    distance = 10.0
    final_configuration = np.array(
        [
            0.0,  # Foot x-position (global frame)
            0.0,  # Foot y-position (global frame)
            0.0,  # Foot z-position (global frame)
            0.0,  # Actuator length
            0.0,  # CoM x-position (global frame)
            0.0,  # CoM y-position (global frame)
            8.0,  # CoM z-position (global frame)
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


    # add final state constraint

    def add_final_state_constraint(
            robot: Robot,
            prog: MathematicalProgram,
            x_f: np.array,
            jumping_distance: np.array,
            t_land,
    ):
        """
        Creates a constraint on the final state of the CoM

        Arguments:
            prog - (MathematicalProgram) optimization setup
            x_f - (np.array) final state right before leaving ground
            jumping distance - (np.array) distance in x & y the CoM should jump [m]
            t_land - (float) final time where CoM "lands" [s]
        """

        def landing_constraint(vars: np.array) -> np.array:
            """
            Internal function to setup landing constraint

            Arguments:
                vars - (np.array) array of constraint variables [x, t_land]

            Returns:
                constraint_eval - (np.array) array of variables to contrain
            """
            # Setup constraint
            constraint_eval = np.zeros((4,), dtype=AutoDiffXd)

            # Get state variables
            q = vars[:7]
            q_dot = vars[7:14]

            # Get landing time
            t_land = vars[-1]  # landing time [s]

            # Foot state variables
            p_foot = q[:3]  # Foot position [m]

            # CoM state variabkes
            p_com = q[4:7]  # CoM position [m]
            v_com = q_dot[4:7]  # CoM velocity [m/s]

            # "Initial" states for kinematics
            x_foot, y_foot, z_foot = p_foot[0], p_foot[1], p_foot[2]
            x_com_i, y_com_i, z_com_i = p_com[0], p_com[1], p_com[2]
            v_com_x_i, v_com_y_i, v_com_z_i = v_com[0], v_com[1], v_com[2]

            # "Final" states for kinematics
            x_com_f = x_com_i + v_com_x_i * t_land
            y_com_f = y_com_i + v_com_y_i * t_land
            z_com_f = (
                    z_com_i + v_com_z_i * t_land + (1 / 2) * robot.g * t_land ** 2
            )

            # Setup constraints
            constraint_eval[0] = x_com_f
            constraint_eval[1] = y_com_f
            constraint_eval[2] = z_com_f
            constraint_eval[3] = np.sqrt(
                (x_com_i - x_foot) ** 2
                + (y_com_i - y_foot) ** 2
                + (z_com_i - z_foot) ** 2
            )

            # constraint_eval = [constraint_eval[i] for i in range(4)]
            print(type(constraint_eval))

            return constraint_eval

        # Kinematic constraints
        x_jump = jumping_distance[0]  # TODO
        y_jump = jumping_distance[1]
        bounds = np.array(
            [x_jump, y_jump, robot.L_0, robot.L_0]
        )  # TODO z constraint
        evaluated_constraints = landing_constraint(np.append(x_f, t_land))

        # Constraint on landing time
        prog.AddConstraint(t_land[0] >= 0)


    # add kinematic constraints
    jump = np.array([distance, distance])
    add_final_state_constraint(robot, prog, xf, jump, t_land)

    # Add the collocation aka dynamics constraints
    AddCollocationConstraints(prog, robot, N, x, u, timesteps)

    # want to minimize u, quadratic cost
    g = 0
    for i in range(N - 1):
        g += 0.5 * (timesteps[1] - timesteps[0]) * (u[i].T @ u[i] + u[i + 1].T @ u[i + 1])
    prog.AddQuadraticCost(g)

    # TODO: Add bounding box constraints on the inputs and qdot - fake rn

    # x[0],  # Foot x-position (global frame)
    # x[1],  # Foot y-position (global frame)
    # x[2],  # Foot z-position (global frame)
    # x[3],  # Actuator length
    # x[4],  # CoM x-position (global frame)
    # x[5],  # CoM y-position (global frame)
    # x[6],  # CoM z-position (global frame)

    print(x.shape)
    prog.AddBoundingBoxConstraint(0, 0, x[:, :3])
    prog.AddBoundingBoxConstraint(0, 1e10, x[:, 6])
    for i in range(N):
        d = (x[i, 4] - x[i, 0]) ** 2 + (x[i, 5] - x[i, 1]) ** 2 + (x[i, 6] - x[i, 2]) ** 2
        prog.AddConstraint(d <= (robot.L_0) ** 2)

    # ubVel = np.ones((N, n_q)) * vel_limits
    # lbVel = -ubVel
    # prog.AddBoundingBoxConstraint(lb, ub, x[:, 0:n_q])
    # prog.AddBoundingBoxConstraint(lbVel, ubVel, x[:, n_q:N])
    # ubU = np.ones((N, n_u)) * effort_limits
    # lbU = -ubU
    # prog.AddBoundingBoxConstraint(lbU, ubU, u)

    # give the solver an initial guess for x and u using prog.SetInitialGuess(var, value)
    # guess_u = np.random.rand(N, n_u)
    # guess_x = np.zeros((N, n_x))
    # guess_x[:, 0] = np.linspace(initial_state[0], final_configuration[0], N)
    # guess_x[:, 1] = np.linspace(initial_state[1], final_configuration[1], N)
    #
    # print(guess_x)
    # prog.SetInitialGuess(x, guess_x)  # guess x
    # prog.SetInitialGuess(u, guess_u)  # guess u
    # prog.SetInitialGuess(u, effort_limits * guess_u - 0.5 * np.ones((N, 2)))  # guess u with effort limits

    for i in range(N):
        prog.SetInitialGuess(x[i, :], robot.x_0)
        prog.SetInitialGuess(u[i], np.array([0]))

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

    # Reconstruct trajectory
    x_dot_sol = np.zeros(x_sol.shape)
    for i in range(N):
        x_dot_sol[i] = robot.f(x_sol[i], u_sol[i])
    x_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, x_dot_sol.T)
    u_sol = np.reshape(u_sol, (N, 1))
    u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

    plt.figure()
    plt.plot(timesteps, x_sol[:, 6])
    plt.xlabel("time")
    plt.ylabel("Z")
    plt.show()
    # plt.figure()
    # plt.plot(x_sol[:, 4], x_sol[:, 6])
    # plt.xlabel("x")
    # plt.ylabel("Z")
    # plt.show()

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
