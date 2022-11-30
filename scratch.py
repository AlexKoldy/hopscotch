from pydrake.all import MathematicalProgram, Solve, PiecewisePolynomial
from pydrake.autodiffutils import AutoDiffXd
import numpy as np

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
                z_com_i + v_com_z_i * t_land + (1 / 2) * self.aslip.g * t_land ** 2
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
        [x_jump, y_jump, self.aslip.L_0, self.aslip.L_0]
    )  # TODO z constraint
    prog.AddConstraint(landing_constraint, bounds, bounds, np.hstack((x_f, t_land)))

    # Constraint on landing time
    prog.AddConstraint(t_land[0] >= 0)