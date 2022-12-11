from pydrake.all import MathematicalProgram, Solve, PiecewisePolynomial
from pydrake.autodiffutils import AutoDiffXd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline, CubicSpline

import importlib
import aslip

importlib.reload(aslip)
from aslip import ASLIP

# import latexify


class OfflinePlanner:
    def __init__(self):
        """
        Offline center of mass (CoM) trajectory optimizer
        for jumping biped. Uses actuated SLIP (aSLIP) model
        """
        # Model
        self.aslip = ASLIP()

        # State parameters
        self.x_0 = self.aslip.x_0  # initial state

        # Collocation parameters
        self.N = 20  # number of knot points

        self.d_0 = np.sqrt(
            (self.aslip.x_0[4] - self.aslip.x_0[0]) ** 2
            + (self.aslip.x_0[5] - self.aslip.x_0[1]) ** 2
            + (self.aslip.x_0[6] - self.aslip.x_0[2]) ** 2
        )

        self.aslip.L_0 = self.d_0
        # self.x_0[3] = self.d_0 - self.aslip.L_0

    def find_com_trajectory(
        self, final_state: np.array, t_f: float, jumping_distance: np.array
    ) -> list:
        """
        Finds the CoM trajectory
        Arguments:
            final_state - (np.array) final configuration
            t_f - (float) final time
            jumping distance - (np.array) distance in x & y the CoM should jump [m]
        Returns:
            [x_traj, u_traj] - (list) trajectory generated by optimizer
        """
        # Create the mathematical program
        prog = MathematicalProgram()

        # Initialize decision variables
        n_x = self.x_0.shape[0]  # size of state vector
        n_u = 1  # size of input vector TODO: potentiall change this

        x = np.zeros((self.N, n_x), dtype="object")
        u = np.zeros((self.N, n_u), dtype="object")

        for i in range(self.N):
            x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
            u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

        x_f = x[-1]

        # Intialize time parameters
        t_0 = 0.3  # initial time [s]
        # self.t_0
        t_land = prog.NewContinuousVariables(1, "t_land")  # [s]
        timesteps = np.linspace(t_0, t_f, self.N)  # [s]

        # Initial and final state constraints
        prog.AddLinearEqualityConstraint(x[0], self.x_0)  # constraint on initial state
        self.add_final_state_constraint(
            prog, x_f, jumping_distance, t_land
        )  # constraint on final state

        d_f = np.sqrt(
            (x[-1, 4] - x[-1, 0]) ** 2
            + (x[-1, 5] - x[-1, 1]) ** 2
            + (x[-1, 6] - x[-1, 2]) ** 2
        )
        print("d_f: ", d_f)
        prog.AddConstraint(d_f == self.d_0)

        # Dynamics constraints
        self.add_dynamics_constraint(prog, x, u, timesteps)

        # Add foot constraint
        bounds = np.array([0, 0, 0])
        for i in range(self.N):
            prog.AddLinearEqualityConstraint(x[i, :3], bounds)

        # Add spring constraint
        for i in range(self.N - 1):
            d = np.sqrt(
                (x[i, 4] - x[i, 0]) ** 2
                + (x[i, 5] - x[i, 1]) ** 2
                + (x[i, 6] - x[i, 2]) ** 2
            )
            prog.AddConstraint(d - x[i, 3] <= self.aslip.L_0)
            prog.AddConstraint(d >= 0.1)
            prog.AddConstraint(d <= self.d_0)
            prog.AddConstraint(x[i, 3] >= 0)

            # Add constraint on final actuator velocity
            # prog.AddConstraint(x[i, 3] == 0)
            # prog.AddConstraint(x[i, 10] == 0)

            # Add final state constraint
            # prog.AddConstraint(x[i, 4] <= 1.0)

        # Find delta t
        dt = timesteps[1] - timesteps[0]  # [s]

        # Add cost
        cost = 0
        for i in range(self.N - 1):
            u_i = u[i]
            u_ip1 = u[i + 1]
            cost += (dt / 2.0) + (u_i.T @ u_i + u_ip1.T @ u_ip1)
        prog.AddQuadraticCost(cost)

        # constrain the actuator velocity to positive then negative
        # for i in range(15):
        #     prog.AddConstraint(x[i, 10] >= 0)
        # for i in np.arange(15, self.N):
        #     prog.AddConstraint(x[i, 10] <= 0)

        # effort_limits = 500
        # for i in range(self.N):
        #     prog.AddBoundingBoxConstraint(-effort_limits, effort_limits, u[i])

        # Add initial guess
        for i in range(self.N):
            prog.SetInitialGuess(x[i, :], self.x_0)
            prog.SetInitialGuess(u[i, 0], np.array([0]))

        # Solve optimiation problem
        result = Solve(prog)

        x_sol = result.GetSolution(x)
        u_sol = result.GetSolution(u)
        t_land_sol = result.GetSolution(t_land)

        # Reconstruct trajectory
        x_dot_sol = np.zeros(x_sol.shape)
        for i in range(self.N):
            x_dot_sol[i] = self.aslip.f(x_sol[i], u_sol[i])

        # timesteps = np.linspace(0.3, 0.6, self.N)
        x_traj2 = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, x_dot_sol.T)
        print(type(x_traj2))

        x_traj = CubicHermiteSpline(timesteps, x_sol[:, 4], x_dot_sol[:, 4])
        r_traj = CubicHermiteSpline(timesteps, x_sol[:, 3], x_dot_sol[:, 3])
        z_traj = CubicHermiteSpline(timesteps, x_sol[:, 6], x_dot_sol[:, 6])
        v_x_traj = CubicHermiteSpline(timesteps, x_sol[:, 11], x_dot_sol[:, 11])
        v_z_traj = CubicHermiteSpline(timesteps, x_sol[:, 13], x_dot_sol[:, 13])
        f_x_traj = CubicHermiteSpline(timesteps, x_sol[:, 0], x_dot_sol[:, 0])
        f_z_traj = CubicHermiteSpline(timesteps, x_sol[:, 2], x_dot_sol[:, 2])
        f_vx_traj = CubicHermiteSpline(timesteps, x_sol[:, 7], x_dot_sol[:, 7])
        f_vz_traj = CubicHermiteSpline(timesteps, x_sol[:, 9], x_dot_sol[:, 9])
        # u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

        return (
            x_traj,
            z_traj,
            t_land_sol,
            v_x_traj,
            v_z_traj,
            f_x_traj,
            f_z_traj,
            f_vx_traj,
            f_vz_traj,
            r_traj,
        )

    def add_final_state_constraint(
        self,
        prog: MathematicalProgram,
        x_f: np.array,
        jumping_distance: np.array,
        t_land: float,
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
                z_com_i + v_com_z_i * t_land + (1 / 2) * self.aslip.g * t_land**2
            )

            # Setup constraints
            constraint_eval[0] = x_com_f
            constraint_eval[1] = y_com_f
            constraint_eval[2] = z_com_f
            constraint_eval[3] = (
                np.sqrt(
                    (x_com_i - x_foot) ** 2
                    + (y_com_i - y_foot) ** 2
                    + (z_com_i - z_foot) ** 2
                )
                - q[3]
            )

            # constraint_eval = [constraint_eval[i] for i in range(4)]

            return constraint_eval

        # Kinematic constraints
        x_jump = jumping_distance[0]  # TODO
        y_jump = jumping_distance[1]
        bounds = np.array([x_jump, y_jump, self.d_0, self.aslip.L_0])
        prog.AddConstraint(landing_constraint, bounds, bounds, np.hstack((x_f, t_land)))

        # Constraint on landing time
        prog.AddConstraint(t_land[0] >= 0)

    def add_dynamics_constraint(
        self, prog: MathematicalProgram, x: np.array, u: np.array, timesteps: np.array
    ):
        """
        Adds aSLIP dynamics constraints
        Arguments:
            prog - (MathematicalProgram) optimization setup
            x - (np.array) state decision variables
            u - (np.array) input decision variables
            timesteps - (np.array) timesteps array [s]
        """

        def collocation_constraint_evaluator(
            dt: float, x_i: np.array, u_i: float, x_ip1: np.array, u_ip1: float
        ) -> np.array:
            """
            Establishes collocation constraint variables
            Arguments:
                x_i - (np.array) aSLIP state at ith timestep
                u_i - (np.array) input at ith timestep
                x_ip1 - (np.array) aSLIP state at (i+1)th timestep
                u_ip1 - (np.array) input at (i+1)th timestep
            Returns:
                h_i - (np.array) collocation constraints
            """
            n_x = self.x_0.shape[0]
            # h_i = np.zeros((n_x,))

            f_i = self.aslip.f(x_i, u_i[0])
            f_ip1 = self.aslip.f(x_ip1, u_ip1[0])

            s_dot_i = (1.5 / dt) * (x_ip1 - x_i) - 0.25 * (f_i + f_ip1)
            s_i = 0.5 * (x_i + x_ip1) - (dt / 8.0) * (f_ip1 - f_i)

            h_i = s_dot_i - self.aslip.f(s_i, 0.5 * (u_ip1[0] + u_i[0]))

            return h_i

        # Get state and input sizes
        n_x = self.x_0.shape[0]
        n_u = 1

        # Add dynamics (collocation) constraints
        for i in range(self.N - 1):

            def collocation_constraint_helper(vars) -> np.array:
                """
                Assists in collocation constraints
                Arguments:
                    vars - (np.array) array of constraint variables [x_i, u_i, x_ip1, u_ip1]
                Returns:
                    collocation_constraints - (np.array) collocation setup
                """
                x_i = vars[:n_x]
                u_i = vars[n_x : n_x + n_u]
                x_ip1 = vars[n_x + n_u : 2 * n_x + n_u]
                u_ip1 = vars[-n_u:]
                collocation_constraints = collocation_constraint_evaluator(
                    timesteps[i + 1] - timesteps[i], x_i, u_i, x_ip1, u_ip1
                )
                return collocation_constraints

            bounds = np.zeros((n_x,))
            bounds = np.zeros((n_x,))
            prog.AddConstraint(
                collocation_constraint_helper,
                bounds,
                bounds,
                np.hstack((x[i], u[i], x[i + 1], u[i + 1])),
            )


if __name__ == "__main__":

    planner = OfflinePlanner()

    print("d_0: ", planner.d_0)

    t_last = 0.6
    distance = np.array([1, 0])

    (
        x_traj,
        z_traj,
        t_land,
        v_x_traj,
        v_z_traj,
        f_x_traj,
        f_z_traj,
        f_vx_traj,
        f_vz_traj,
        r_traj,
    ) = planner.find_com_trajectory(planner.aslip.x_0 * 2, t_last, distance)

    # print(planner.find_com_trajectory(planner.aslip.x_0 * 2, t_last, distance))

    t = np.linspace(0, t_last, 100)
    xx = x_traj(t)[-1]
    zz = z_traj(t)[-1]

    fx = f_x_traj(t)[-1]
    fz = f_z_traj(t)[-1]

    d_ff = np.sqrt((xx - fz) ** 2 + (zz - fz) ** 2)
    print(f"d_f: {d_ff}")

    v_xx = v_x_traj(t)[-1]
    v_zz = v_z_traj(t)[-1]
    print(v_zz)
    print(t_land)

    dt = t[1] - t[0]

    tt = t_last

    x_com = []
    z_com = []
    x_com.append(xx)
    z_com.append(zz)
    # print(t_land)
    # print(dt)

    t_max = t_last + t_land[0]
    # t_max = 20

    while tt < t_max:
        xx += v_xx * dt
        zz += v_zz * dt + (1 / 2) * planner.aslip.g * dt**2
        v_zz += planner.aslip.g * dt
        x_com.append(xx)
        z_com.append(zz)
        tt += dt

    tttt = np.array([0, 0.3])
    xxxx = np.array([0, planner.aslip.x_0[4]])
    yyyy = np.array([0, 0])
    zzzz = np.array([0.8, planner.aslip.x_0[6]])

    xxx = CubicSpline(tttt, xxxx)
    yyy = CubicSpline(tttt, yyyy)
    zzz = CubicSpline(tttt, zzzz)

    # print(len(x_com))

    t = np.linspace(0.3, t_last, 100)
    tt = np.linspace(0, 0.3, 100)

    plt.figure()
    plt.title("com x vs com z")
    plt.plot(x_traj(t), z_traj(t))
    plt.plot(x_com, z_com)
    plt.plot(xxx(tt), zzz(tt))
    plt.xlabel("CoM x")
    plt.ylabel("CoM z")
    plt.show()

    # t2 = np.linspace(t_last, t_max, (t_max - t_last) / dt + 2)

    # plt.figure()
    # plt.title("time vs CoM z")
    # plt.plot(t, z_traj(t))
    # plt.plot(t2, z_com)
    # plt.xlabel("time")
    # plt.ylabel("CoM z")
    # plt.show()

    plt.figure()
    plt.title("r")
    plt.plot(t, r_traj(t))
    plt.show()

# plot foot z versus time (pre and post jump)
# plt.figure()
# plt.plot(t, f_z_traj(t))
# plt.plot(t2, f_z_traj(t2))
# plt.title("Foot z vs t (pre and post jump)")
# plt.xlabel("time")
# plt.ylabel("Foot z")
# plt.show()


# foot x vs foot z without post jump
# plt.figure()
# plt.plot(f_x_traj(t), f_z_traj(t))
# plt.xlabel("Foot x")
# plt.ylabel("Foot z")
# plt.show()


# plt.figure()
# plt.title("angle vs time")
# plt.plot(t, 180 / np.pi * np.arctan2(z_traj(t), x_traj(t)))
# plt.show()

# plt.figure()
# plt.plot(t, z_traj(t))
# plt.show()
#     planner = OfflinePlanner()
#     x_traj, z_traj, t_land, v_x_traj, v_z_traj = planner.find_com_trajectory(
#         planner.aslip.x_0 * 2, 2, np.array([12.0, 0])
#     )
#     t = np.linspace(0, 2, 100)
#     xx = x_traj(t)[-1]
#     zz = z_traj(t)[-1]

#     v_xx = v_x_traj(t)[-1]
#     v_zz = v_z_traj(t)[-1]
#     print(v_zz)
#     print(t_land)

#     dt = t[1] - t[0]

#     tt = 2

#     x_com = []
#     z_com = []
#     x_com.append(xx)
#     z_com.append(zz)
#     # print(t_land)
#     # print(dt)

#     t_max = 2 + t_land[0]
#     # t_max = 20

#     while tt < t_max:
#         xx += v_xx * dt
#         zz += v_zz * dt + (1 / 2) * planner.aslip.g * dt**2
#         v_zz += planner.aslip.g * dt
#         x_com.append(xx)
#         z_com.append(zz)
#         tt += dt

#     # print(len(x_com))

#     # print(x_traj.shape)
#     t = np.linspace(0, 2, 100)
#     plt.figure()
#     plt.plot(x_traj(t), z_traj(t))
#     plt.plot(x_com, z_com)
#     plt.show()

#     plt.figure()
#     plt.plot(t, 180 / np.pi * np.arctan2(z_traj(t), x_traj(t)))
#     plt.show()

#     # plt.figure()
#     # plt.plot(t, z_traj(t))
#     # plt.show()
