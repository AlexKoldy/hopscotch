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
    
    def find_trajectories(self, final_state: np.array, t_f: float, jumping_distance: np.array
    ) -> list:
        jump_duration = 0.3
        t_begin_motion = 0.3 # robot starts moving after 0.3 seconds
        t_takeoff = 0.3 # takeoff within 0.3 seconds
        t_balance = 1 # duration of balancing/stabilization
        distance = np.array([1, 0])
        x_traj, y_traj, z_traj, t_land, v_x_traj, v_y_traj, v_z_traj, f_x_traj, f_z_traj, f_vx_traj, f_vz_traj, r_traj, com_mode_1 = self.find_com_trajectory(self.aslip.x_0 * 2, t_begin_motion+t_takeoff, distance
    )
        
        #LEFT FOOT !!!
        # MODE 0: lean forward
        t_0 = np.array([0, t_begin_motion])
        x_foot_0 = np.array([0.0, 0.0])
        y_foot_0 = np.array([0.2, 0.2])
        z_foot_0 = np.array([0.0, 0.0])

        x_foot_mode0 = CubicSpline(t_0, x_foot_0)
        y_foot_mode0 = CubicSpline(t_0, y_foot_0)
        z_foot_mode0 = CubicSpline(t_0, z_foot_0)

        # MODE 1: TAKEOFF (foot still on ground)
        t_1 = np.array([t_begin_motion, t_takeoff + t_begin_motion])

        x_foot_1 = np.array([0, 0])
        y_foot_1 = np.array([0.2, 0.2])
        z_foot_1 = np.array([0.0, 0.0])

        x_foot_mode1 = CubicSpline(t_1, x_foot_1)
        y_foot_mode1 = CubicSpline(t_1, y_foot_1)
        z_foot_mode1 = CubicSpline(t_1, z_foot_1)

        # MODE 2: PROJECTILE MOTION
        jump_duration = 0.3
        distance = np.array([1, 0])
        # (x_traj, z_traj, t_land, v_x_traj, v_z_traj, f_x_traj, f_z_traj, f_vx_traj, f_vz_traj, r_traj,) = planner.find_com_trajectory(planner.aslip.x_0 * 2, t_begin_motion+t_takeoff, distance)
        # print("land: ", t_land)

        t_2 = np.array([t_1[-1], t_1[-1] + t_land[0]])

        left_foot_final_x = distance[0]
        x_foot_2 = np.array([0, 0.5 * left_foot_final_x, left_foot_final_x])

        # x_foot_mode2 = CubicSpline(t_2, x_foot_2)
        z_foot_2 = np.array([0, 0.5, 0.3])
        z_foot_mode2 = CubicSpline(np.array([t_1[-1], (t_1[-1] + t_land[0] + t_1[-1])/2, t_1[-1] + t_land[0]]), z_foot_2)

        y_foot_2 = np.array([0.2, 0.2, 0.2])
        # y_foot_mode2 = CubicSpline(t_2, y_foot_2)
        # # MODE 3: STABILIZE --> osc


        #RIGHT FOOT !!!
        # MODE 0: lean forward
        # t_0 = np.array([0, t_begin_motion])
        xR_foot_0 = np.array([0.0, 0.0])
        yR_foot_0 = np.array([-0.2, -0.2])
        zR_foot_0 = np.array([0.0, 0.0])

        xR_foot_mode0 = CubicSpline(t_0, xR_foot_0)
        yR_foot_mode0 = CubicSpline(t_0, yR_foot_0)
        zR_foot_mode0 = CubicSpline(t_0, zR_foot_0)

        # MODE 1: TAKEOFF (foot still on ground)
        # t_1 = np.array([t_begin_motion, t_takeoff + t_begin_motion])

        xR_foot_1 = np.array([0, 0])
        yR_foot_1 = np.array([-0.2, -0.2])
        zR_foot_1 = np.array([0.0, 0.0])

        xR_foot_mode1 = CubicSpline(t_1, xR_foot_1)
        yR_foot_mode1 = CubicSpline(t_1, yR_foot_1)
        zR_foot_mode1 = CubicSpline(t_1, zR_foot_1)

        # MODE 2: PROJECTILE MOTION
        
        # t_2 = np.array([t_1[-1], t_1[-1] + t_land[0]])

        right_foot_final_x = distance[0]
        xR_foot_2 = np.array([0, 0.5*right_foot_final_x, right_foot_final_x])

        # xR_foot_mode2 = CubicSpline(t_2, xR_foot_2)
        zR_foot_2 = np.array([0, 0.5, 0.0])
        zR_foot_mode2 = CubicSpline(np.array([t_1[-1], (t_1[-1] + t_land[0] + t_1[-1])/2, t_1[-1] + t_land[0]]), zR_foot_2)

        yR_foot_2 = np.array([-0.2, -0.2, -0.2])
        # yR_foot_mode2 = CubicSpline(t_2, yR_foot_2)
        # # MODE 3: STABILIZE --> osc


        ## COM !!!

        t_com = np.linspace(0, t_begin_motion + t_takeoff, 100)
        xx = x_traj(t_com)[-1]
        yy = y_traj(t_com)[-1]
        zz = z_traj(t_com)[-1]

        v_xx = v_x_traj(t_com)[-1]
        v_zz = v_z_traj(t_com)[-1]
        v_yy = v_y_traj(t_com)[-1]

        dt = t_com[1] - t_com[0]
        tt = t_begin_motion + t_takeoff

        x_com = []
        y_com = []
        z_com = []
        t_com_proj = []
        x_com.append(xx)
        y_com.append(yy)
        z_com.append(zz)
        t_com_proj.append(tt)

        t_max = t_begin_motion + t_takeoff + t_land[0]

        while tt < t_max:
            xx += v_xx * dt
            yy += v_yy * dt
            zz += v_zz * dt + (1 / 2) * self.aslip.g * dt**2
            v_zz += self.aslip.g * dt
            x_com.append(xx)
            y_com.append(yy)
            z_com.append(zz)
            tt += dt
            t_com_proj.append(tt)
            

        tttt = np.array([0, 0.3])
        xxxx = np.array([0, self.aslip.x_0[4]])
        yyyy = np.array([0, 0])
        zzzz = np.array([0.8, self.aslip.x_0[6]])

        xxx = CubicSpline(tttt, xxxx)
        yyy = CubicSpline(tttt, yyyy)
        zzz = CubicSpline(tttt, zzzz)

        t_com_t = np.linspace(t_begin_motion, t_takeoff + t_begin_motion, 100)
        t_com_tt = np.linspace(0, t_begin_motion, 100)


        # now input it into leg spline generators
        lf_mode_0 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                t_0, np.array([x_foot_0, y_foot_0, z_foot_0]).reshape((3,2)), np.zeros((3,)), np.zeros((3,))
            )
        lf_mode_1 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                t_1, np.array([x_foot_1, y_foot_1, z_foot_1]).reshape((3,2)), np.zeros((3,)), np.zeros((3,))
            )
        lf_mode_2 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([t_1[-1], (t_1[-1] + t_land[0] + t_1[-1])/2, t_1[-1] + t_land[0]]) , np.array([x_foot_2, y_foot_2, z_foot_2]).reshape((3,3)), np.zeros((3,)), np.zeros((3,))
            )
        
        rf_mode_0 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                t_0, np.array([xR_foot_0, yR_foot_0, zR_foot_0]).reshape((3,2)), np.zeros((3,)), np.zeros((3,))
            )
        rf_mode_1 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                t_1, np.array([xR_foot_1, yR_foot_1, zR_foot_1]).reshape((3,2)), np.zeros((3,)), np.zeros((3,))
            )
        rf_mode_2 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([t_1[-1], (t_1[-1] + t_land[0] + t_1[-1])/2, t_1[-1] + t_land[0]]) , np.array([xR_foot_2, yR_foot_2, zR_foot_2]).reshape((3,3)), np.zeros((3,)), np.zeros((3,))
            )

        com_mode_0 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([0, t_begin_motion]), np.array([xxxx, yyyy, zzzz]).reshape((3,2)), np.zeros((3,)), np.zeros((3,))
            )
        # com_mode_1 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        #         t_com_t, np.array([x_traj(t_com_t), y_traj(t_com_t), z_traj(t_com_t)]).reshape((3,100)), np.zeros((3,)), np.zeros((3,))
        #     )
        com_mode_2 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([t_com_proj[0], t_com_proj[-1]]) , np.array([[x_com[0],x_com[-1]], [y_com[0], y_com[-1]], [z_com[0], z_com[-1]]]).reshape((3,2)), np.zeros((3,)), np.zeros((3,))
            )


        return com_mode_0, com_mode_1, com_mode_2, lf_mode_0, lf_mode_1, lf_mode_2, rf_mode_0, rf_mode_1, rf_mode_2

    

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
        t_0 = 0.30  # initial time [s]
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
        com_traj = PiecewisePolynomial.CubicHermite(timesteps, x_sol[:, 4:7].T, x_dot_sol[:, 4:7].T)
        # print(type(x_traj2))

        x_traj = CubicHermiteSpline(timesteps, x_sol[:, 4], x_dot_sol[:, 4])
        y_traj = CubicHermiteSpline(timesteps, x_sol[:, 5], x_dot_sol[:, 5])
        r_traj = CubicHermiteSpline(timesteps, x_sol[:, 3], x_dot_sol[:, 3])
        z_traj = CubicHermiteSpline(timesteps, x_sol[:, 6], x_dot_sol[:, 6])
        v_x_traj = CubicHermiteSpline(timesteps, x_sol[:, 11], x_dot_sol[:, 11])
        v_y_traj = CubicHermiteSpline(timesteps, x_sol[:, 12], x_dot_sol[:, 12])
        v_z_traj = CubicHermiteSpline(timesteps, x_sol[:, 13], x_dot_sol[:, 13])
        f_x_traj = CubicHermiteSpline(timesteps, x_sol[:, 0], x_dot_sol[:, 0])
        f_z_traj = CubicHermiteSpline(timesteps, x_sol[:, 2], x_dot_sol[:, 2])
        f_vx_traj = CubicHermiteSpline(timesteps, x_sol[:, 7], x_dot_sol[:, 7])
        f_vz_traj = CubicHermiteSpline(timesteps, x_sol[:, 9], x_dot_sol[:, 9])
        # u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

        return (
            x_traj,
            y_traj,
            z_traj,
            t_land_sol,
            v_x_traj,
            v_y_traj,
            v_z_traj,
            f_x_traj,
            f_z_traj,
            f_vx_traj,
            f_vz_traj,
            r_traj,
            com_traj
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






# from pydrake.all import MathematicalProgram, Solve, PiecewisePolynomial
# from pydrake.autodiffutils import AutoDiffXd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicHermiteSpline

# import importlib
# import aslip 
# importlib.reload(aslip)
# from aslip import ASLIP

# # import latexify


# class OfflinePlanner:
#     def __init__(self):
#         """
#         Offline center of mass (CoM) trajectory optimizer
#         for jumping biped. Uses actuated SLIP (aSLIP) model
#         """
#         # Model
#         self.aslip = ASLIP()

#         # State parameters
#         self.x_0 = self.aslip.x_0  # initial state
        

#         # Collocation parameters
#         self.N = 20  # number of knot points

#         self.d_0 = np.sqrt(
#                 (self.aslip.x_0[4] - self.aslip.x_0[0]) ** 2
#                 + (self.aslip.x_0[5] - self.aslip.x_0[1]) ** 2
#                 + (self.aslip.x_0[6] - self.aslip.x_0[2]) ** 2
#             )

#         self.aslip.L_0 = self.d_0
#         # self.x_0[3] = self.d_0 - self.aslip.L_0
        

#     def find_com_trajectory(
#         self, final_state: np.array, t_f: float, jumping_distance: np.array
#     ) -> list:
#         """
#         Finds the CoM trajectory

#         Arguments:
#             final_state - (np.array) final configuration
#             t_f - (float) final time
#             jumping distance - (np.array) distance in x & y the CoM should jump [m]

#         Returns:
#             [x_traj, u_traj] - (list) trajectory generated by optimizer
#         """
#         # Create the mathematical program
#         prog = MathematicalProgram()

#         # Initialize decision variables
#         n_x = self.x_0.shape[0]  # size of state vector
#         n_u = 1  # size of input vector TODO: potentiall change this

#         x = np.zeros((self.N, n_x), dtype="object")
#         u = np.zeros((self.N, n_u), dtype="object")

#         for i in range(self.N):
#             x[i] = prog.NewContinuousVariables(n_x, "x_" + str(i))
#             u[i] = prog.NewContinuousVariables(n_u, "u_" + str(i))

#         x_f = x[-1]

#         # Intialize time parameters
#         t_0 = 0.0  # initial time [s]
#         t_land = prog.NewContinuousVariables(1, "t_land")  # [s]
#         timesteps = np.linspace(t_0, t_f, self.N)  # [s]

#         # Initial and final state constraints
#         prog.AddLinearEqualityConstraint(x[0], self.x_0)  # constraint on initial state
#         self.add_final_state_constraint(
#             prog, x_f, jumping_distance, t_land
#         )  # constraint on final state

#         d_f = np.sqrt(
#                 (x[-1, 4] - x[-1, 0]) ** 2
#                 + (x[-1, 5] - x[-1, 1]) ** 2
#                 + (x[-1, 6] - x[-1, 2]) ** 2
#             )
#         print("d_f: ", d_f)
#         prog.AddConstraint(d_f == self.d_0)
        
#         # Dynamics constraints
#         self.add_dynamics_constraint(prog, x, u, timesteps)

#         # Add foot constraint
#         bounds = np.array([0, 0, 0])
#         for i in range(self.N):
#             prog.AddLinearEqualityConstraint(x[i, :3], bounds)

#         # Add spring constraint
#         for i in range(self.N - 1):
#             d = np.sqrt(
#                 (x[i, 4] - x[i, 0]) ** 2
#                 + (x[i, 5] - x[i, 1]) ** 2
#                 + (x[i, 6] - x[i, 2]) ** 2
#             )
#             prog.AddConstraint(d - x[i, 3] <= self.aslip.L_0)
#             prog.AddConstraint(d >= 0.1)
#             prog.AddConstraint(d <= self.d_0)
#             prog.AddConstraint(x[i, 3] >= 0)

#             # Add constraint on final actuator velocity
#             # prog.AddConstraint(x[i, 3] == 0)
#             # prog.AddConstraint(x[i, 10] == 0)

#             # Add final state constraint
#             # prog.AddConstraint(x[i, 4] <= 1.0)

#         # Find delta t
#         dt = timesteps[1] - timesteps[0]  # [s]

#         # Add cost
#         cost = 0
#         for i in range(self.N - 1):
#             u_i = u[i]
#             u_ip1 = u[i + 1]
#             cost += (dt / 2.0) + (u_i.T @ u_i + u_ip1.T @ u_ip1)
#         prog.AddQuadraticCost(cost)

#         # constrain the actuator velocity to positive then negative
#         for i in range(15):
#             prog.AddConstraint(x[i, 10] >= 0)
#         for i in np.arange(15, self.N):
#             prog.AddConstraint(x[i, 10] <= 0)
            
#         # effort_limits = 500
#         # for i in range(self.N):
#         #     prog.AddBoundingBoxConstraint(-effort_limits, effort_limits, u[i])

#         # Add initial guess
#         for i in range(self.N):
#             prog.SetInitialGuess(x[i, :], self.x_0)
#             prog.SetInitialGuess(u[i, 0], np.array([0]))

#         # Solve optimiation problem
#         result = Solve(prog)

#         x_sol = result.GetSolution(x)
#         u_sol = result.GetSolution(u)
#         t_land_sol = result.GetSolution(t_land)

#         # Reconstruct trajectory
#         x_dot_sol = np.zeros(x_sol.shape)
#         for i in range(self.N):
#             x_dot_sol[i] = self.aslip.f(x_sol[i], u_sol[i])
#         x_traj2 = PiecewisePolynomial.CubicHermite(timesteps, x_sol.T, x_dot_sol.T)
#         print(type(x_traj2))

#         x_traj = CubicHermiteSpline(timesteps, x_sol[:, 4], x_dot_sol[:, 4])
#         z_traj = CubicHermiteSpline(timesteps, x_sol[:, 6], x_dot_sol[:, 6])
#         v_x_traj = CubicHermiteSpline(timesteps, x_sol[:, 11], x_dot_sol[:, 11])
#         v_z_traj = CubicHermiteSpline(timesteps, x_sol[:, 13], x_dot_sol[:, 13])
#         f_x_traj = CubicHermiteSpline(timesteps, x_sol[:, 0], x_dot_sol[:, 0])
#         f_z_traj = CubicHermiteSpline(timesteps, x_sol[:, 2], x_dot_sol[:, 2])
#         f_vx_traj = CubicHermiteSpline(timesteps, x_sol[:, 7], x_dot_sol[:, 7])
#         f_vz_traj = CubicHermiteSpline(timesteps, x_sol[:, 9], x_dot_sol[:, 9])
#         r_traj = CubicHermiteSpline(timesteps, x_sol[:, 4], x_dot_sol[:, 3])
#         # u_traj = PiecewisePolynomial.ZeroOrderHold(timesteps, u_sol.T)

#         return x_traj, z_traj, t_land_sol, v_x_traj, v_z_traj, f_x_traj, f_z_traj, f_vx_traj, f_vz_traj, r_traj

#     def add_final_state_constraint(
#         self,
#         prog: MathematicalProgram,
#         x_f: np.array,
#         jumping_distance: np.array,
#         t_land: float,
#     ):
#         """
#         Creates a constraint on the final state of the CoM

#         Arguments:
#             prog - (MathematicalProgram) optimization setup
#             x_f - (np.array) final state right before leaving ground
#             jumping distance - (np.array) distance in x & y the CoM should jump [m]
#             t_land - (float) final time where CoM "lands" [s]
#         """

#         def landing_constraint(vars: np.array) -> np.array:
#             """
#             Internal function to setup landing constraint

#             Arguments:
#                 vars - (np.array) array of constraint variables [x, t_land]

#             Returns:
#                 constraint_eval - (np.array) array of variables to contrain
#             """
#             # Setup constraint
#             constraint_eval = np.zeros((4,), dtype=AutoDiffXd)

#             # Get state variables
#             q = vars[:7]
#             q_dot = vars[7:14]

#             # Get landing time
#             t_land = vars[-1]  # landing time [s]

#             # Foot state variables
#             p_foot = q[:3]  # Foot position [m]

#             # CoM state variabkes
#             p_com = q[4:7]  # CoM position [m]
#             v_com = q_dot[4:7]  # CoM velocity [m/s]

#             # "Initial" states for kinematics
#             x_foot, y_foot, z_foot = p_foot[0], p_foot[1], p_foot[2]
#             x_com_i, y_com_i, z_com_i = p_com[0], p_com[1], p_com[2]
#             v_com_x_i, v_com_y_i, v_com_z_i = v_com[0], v_com[1], v_com[2]

#             # "Final" states for kinematics
#             x_com_f = x_com_i + v_com_x_i * t_land
#             y_com_f = y_com_i + v_com_y_i * t_land
#             z_com_f = (
#                 z_com_i + v_com_z_i * t_land + (1 / 2) * self.aslip.g * t_land**2
#             )

#             # Setup constraints
#             constraint_eval[0] = x_com_f
#             constraint_eval[1] = y_com_f
#             constraint_eval[2] = z_com_f
#             constraint_eval[3] = (
#                 np.sqrt(
#                     (x_com_i - x_foot) ** 2
#                     + (y_com_i - y_foot) ** 2
#                     + (z_com_i - z_foot) ** 2
#                 )
#                 - q[3]
#             )

#             # constraint_eval = [constraint_eval[i] for i in range(4)]

#             return constraint_eval

#         # Kinematic constraints
#         x_jump = jumping_distance[0]  # TODO
#         y_jump = jumping_distance[1]
#         bounds = np.array([x_jump, y_jump, self.d_0, self.aslip.L_0])
#         prog.AddConstraint(landing_constraint, bounds, bounds, np.hstack((x_f, t_land)))

#         # Constraint on landing time
#         prog.AddConstraint(t_land[0] >= 0)

#     def add_dynamics_constraint(
#         self, prog: MathematicalProgram, x: np.array, u: np.array, timesteps: np.array
#     ):
#         """
#         Adds aSLIP dynamics constraints

#         Arguments:
#             prog - (MathematicalProgram) optimization setup
#             x - (np.array) state decision variables
#             u - (np.array) input decision variables
#             timesteps - (np.array) timesteps array [s]
#         """

#         def collocation_constraint_evaluator(
#             dt: float, x_i: np.array, u_i: float, x_ip1: np.array, u_ip1: float
#         ) -> np.array:
#             """
#             Establishes collocation constraint variables

#             Arguments:
#                 x_i - (np.array) aSLIP state at ith timestep
#                 u_i - (np.array) input at ith timestep
#                 x_ip1 - (np.array) aSLIP state at (i+1)th timestep
#                 u_ip1 - (np.array) input at (i+1)th timestep

#             Returns:
#                 h_i - (np.array) collocation constraints
#             """
#             n_x = self.x_0.shape[0]
#             # h_i = np.zeros((n_x,))

#             f_i = self.aslip.f(x_i, u_i[0])
#             f_ip1 = self.aslip.f(x_ip1, u_ip1[0])

#             s_dot_i = (1.5 / dt) * (x_ip1 - x_i) - 0.25 * (f_i + f_ip1)
#             s_i = 0.5 * (x_i + x_ip1) - (dt / 8.0) * (f_ip1 - f_i)

#             h_i = s_dot_i - self.aslip.f(s_i, 0.5 * (u_ip1[0] + u_i[0]))

#             return h_i

#         # Get state and input sizes
#         n_x = self.x_0.shape[0]
#         n_u = 1

#         # Add dynamics (collocation) constraints
#         for i in range(self.N - 1):

#             def collocation_constraint_helper(vars) -> np.array:
#                 """
#                 Assists in collocation constraints

#                 Arguments:
#                     vars - (np.array) array of constraint variables [x_i, u_i, x_ip1, u_ip1]

#                 Returns:
#                     collocation_constraints - (np.array) collocation setup
#                 """
#                 x_i = vars[:n_x]
#                 u_i = vars[n_x : n_x + n_u]
#                 x_ip1 = vars[n_x + n_u : 2 * n_x + n_u]
#                 u_ip1 = vars[-n_u:]
#                 collocation_constraints = collocation_constraint_evaluator(
#                     timesteps[i + 1] - timesteps[i], x_i, u_i, x_ip1, u_ip1
#                 )
#                 return collocation_constraints

#             bounds = np.zeros((n_x,))
#             bounds = np.zeros((n_x,))
#             prog.AddConstraint(
#                 collocation_constraint_helper,
#                 bounds,
#                 bounds,
#                 np.hstack((x[i], u[i], x[i + 1], u[i + 1])),
#             )


# # if __name__ == "__main__":
# #     planner = OfflinePlanner()
# #     x_traj, z_traj, t_land, v_x_traj, v_z_traj = planner.find_com_trajectory(
# #         planner.aslip.x_0 * 2, 2, np.array([12.0, 0])
# #     )
# #     t = np.linspace(0, 2, 100)
# #     xx = x_traj(t)[-1]
# #     zz = z_traj(t)[-1]

# #     v_xx = v_x_traj(t)[-1]
# #     v_zz = v_z_traj(t)[-1]
# #     print(v_zz)
# #     print(t_land)

# #     dt = t[1] - t[0]

# #     tt = 2

# #     x_com = []
# #     z_com = []
# #     x_com.append(xx)
# #     z_com.append(zz)
# #     # print(t_land)
# #     # print(dt)

# #     t_max = 2 + t_land[0]
# #     # t_max = 20

# #     while tt < t_max:
# #         xx += v_xx * dt
# #         zz += v_zz * dt + (1 / 2) * planner.aslip.g * dt**2
# #         v_zz += planner.aslip.g * dt
# #         x_com.append(xx)
# #         z_com.append(zz)
# #         tt += dt

# #     # print(len(x_com))

# #     # print(x_traj.shape)
# #     t = np.linspace(0, 2, 100)
# #     plt.figure()
# #     plt.plot(x_traj(t), z_traj(t))
# #     plt.plot(x_com, z_com)
# #     plt.show()

# #     plt.figure()
# #     plt.plot(t, 180 / np.pi * np.arctan2(z_traj(t), x_traj(t)))
# #     plt.show()

# #     # plt.figure()
# #     # plt.plot(t, z_traj(t))
# #     # plt.show()
