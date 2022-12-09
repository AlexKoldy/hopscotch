import numpy as np
from typing import Tuple

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.math import RigidTransform
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.solvers import MathematicalProgram, OsqpSolver
from pydrake.multibody.all import JacobianWrtVariable

from osc_gains import OscGains
from point_on_frame import PointOnFrame
from tracking_objectives.point_tracking_objective import PointPositionTrackingObjective
from tracking_objectives.center_of_mass_position_tracking_objective import (
    CenterOfMassPositionTrackingObjective,
)


class OperationalSpaceController(LeafSystem):
    def __init__(self, gains: OscGains):
        """
        TODO
        """
        LeafSystem.__init__(self)

        # Set gains
        self.gains = gains

        # Load multibody plant
        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModelFromFile("3d_biped.urdf")
        # self.plant.WeldFrames(
        #     self.plant.world_frame(),
        #     self.plant.GetBodyByName("base").body_frame(),
        #     RigidTransform.Identity(),
        # )
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # Assign contact frames
        self.contact_points = {
            "left leg": PointOnFrame(
                self.plant.GetBodyByName("left_lower_leg").body_frame(),
                np.array([0, 0, -0.5]),
            ),
            "right leg": PointOnFrame(
                self.plant.GetBodyByName("right_lower_leg").body_frame(),
                np.array([0, 0, -0.5]),
            ),
        }

        # Initiliaze tracking objectives
        self.tracking_objectives = {
            "com_traj": CenterOfMassPositionTrackingObjective(
                self.plant,
                self.plant_context,
                self.gains.k_p_com,
                self.gains.k_d_com,
            ),
            # "left_foot_traj": PointPositionTrackingObjective(
            #     self.plant,
            #     self.plant_context,
            #     self.gains.k_p_left_foot,
            #     self.gains.k_d_left_foot,
            #     self.contact_points["left leg"],
            # ),
            # "right_foot_traj": PointPositionTrackingObjective(
            #     self.plant,
            #     self.plant_context,
            #     self.gains.k_p_right_foot,
            #     self.gains.k_d_right_foot,
            #     self.contact_points["right leg"],
            # ),
        }
        self.tracking_costs = {
            "com_traj": self.gains.w_com,
            # "left_foot_traj": self.gains.w_left_foot,
            # "right_foot_traj": self.gains.w_right_foot,
        }
        self.trajs = self.tracking_objectives.keys()

        # Declare input ports
        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "x", self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()  # state input port

        trj = PiecewisePolynomial()
        self.traj_input_ports = {
            "com_traj": self.DeclareAbstractInputPort(
                "com_traj", AbstractValue.Make(trj)
            ).get_index(),
            # "left_foot_traj": self.DeclareAbstractInputPort(
            #     "left_foot_traj", AbstractValue.Make(trj)
            # ).get_index(),
            # "right_foot_traj": self.DeclareAbstractInputPort(
            #     "right_foot_traj", AbstractValue.Make(trj)
            # ).get_index(),
        }  # trajectory input ports

        # Define the output ports
        self.torque_output_port = self.DeclareVectorOutputPort(
            "u", self.plant.num_actuators(), self.CalcTorques
        )

        self.u = np.zeros((self.plant.num_actuators()))

    def get_traj_input_port(self, traj_name):
        return self.get_input_port(self.traj_input_ports[traj_name])

    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)

    def CalculateContactJacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        For a given finite state, LEFT_STANCE or RIGHT_STANCE, calculate the
        Jacobian terms for the contact constraint, J and Jdot * v.

        As an example, see CalcJ and CalcJdotV in PointPositionTrackingObjective

        use self.contact_points to get the PointOnFrame for the current stance foot
        """
        J = np.zeros((6, self.plant.num_velocities()))
        JdotV = np.zeros((6,))

        # TODO - STUDENT CODE HERE:
        pt_to_track = self.contact_points["left leg"]
        J_left = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            pt_to_track.frame,
            pt_to_track.pt,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

        JdotV_left = self.plant.CalcBiasTranslationalAcceleration(
            self.plant_context,
            JacobianWrtVariable.kV,
            pt_to_track.frame,
            pt_to_track.pt,
            self.plant.world_frame(),
            self.plant.world_frame(),
        ).ravel()

        pt_to_track = self.contact_points["right leg"]
        J_right = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            pt_to_track.frame,
            pt_to_track.pt,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

        JdotV_right = self.plant.CalcBiasTranslationalAcceleration(
            self.plant_context,
            JacobianWrtVariable.kV,
            pt_to_track.frame,
            pt_to_track.pt,
            self.plant.world_frame(),
            self.plant.world_frame(),
        ).ravel()

        J[:3, :] = J_left
        J[3:, :] = J_right

        JdotV[:3] = JdotV_left
        JdotV[3:] = JdotV_right

        return J, JdotV

    def SetupAndSolveQP(
        self, context: Context
    ) -> Tuple[np.ndarray, MathematicalProgram]:

        # First get the state, time, and fsm state
        x = self.EvalVectorInput(context, self.robot_state_input_port_index).get_value()
        t = context.get_time()

        # Update the plant context with the current position and velocity
        self.plant.SetPositionsAndVelocities(self.plant_context, x)

        # Update tracking objectives
        for traj_name in self.trajs:
            traj = self.EvalAbstractInput(
                context, self.traj_input_ports[traj_name]
            ).get_value()
            self.tracking_objectives[traj_name].Update(t, traj)

        """Set up and solve the QP """
        prog = MathematicalProgram()

        # Make decision variables
        u = prog.NewContinuousVariables(self.plant.num_actuators(), "u")
        vdot = prog.NewContinuousVariables(self.plant.num_velocities(), "vdot")
        lambda_c = prog.NewContinuousVariables(6, "lambda_c")

        for traj_name in self.trajs:
            obj = self.tracking_objectives[traj_name]
            yddot_cmd_i = obj.yddot_cmd
            J_i = obj.J
            JdotV_i = obj.JdotV
            W_i = self.tracking_costs[traj_name]

            Q_i = 2 * J_i.T @ W_i @ J_i
            b_i_T = (
                -yddot_cmd_i.T @ W_i @ J_i
                - yddot_cmd_i.T @ W_i.T @ J_i
                + JdotV_i.T @ W_i @ J_i
                + JdotV_i.T @ W_i.T @ J_i
            )
            c_i = (
                yddot_cmd_i.T @ W_i @ yddot_cmd_i
                - yddot_cmd_i.T @ W_i @ JdotV_i
                - JdotV_i.T @ W_i @ yddot_cmd_i
                + JdotV_i.T @ W_i @ JdotV_i
            )

            prog.AddQuadraticCost(Q_i, b_i_T, c_i, vdot, is_convex=True)

        Q_eye = 0.00001 * np.eye(14)
        prog.AddQuadraticCost(2 * Q_eye, np.zeros(14), vdot, is_convex=True)

        # Calculate terms in the manipulator equation
        J_c, J_c_dot_v = self.CalculateContactJacobian()
        M = self.plant.CalcMassMatrix(self.plant_context)
        Cv = self.plant.CalcBiasTerm(self.plant_context)
        # Drake gives gravity as an external force, but we take the negative to match
        # our derivation of the manipulator equations
        G = -self.plant.CalcGravityGeneralizedForces(self.plant_context)
        B = self.plant.MakeActuationMatrix()

        prog.AddLinearEqualityConstraint(
            M @ vdot + Cv + G - B @ u - J_c.T @ lambda_c, np.zeros((14, 1))
        )

        prog.AddLinearEqualityConstraint(J_c_dot_v + J_c @ vdot, np.zeros((6, 1)))

        A = np.array(
            [
                [-1, 0, -1, 0, 0, 0],
                [-1, 0, -1, 0, 0, 0],
                [1, 0, -1, 0, 0, 0],
                [0, 0, 0, -1, 0, -1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, -1],
            ]
        )
        b = np.array([0, 0, 0, 0, 0, 0])
        prog.AddLinearConstraint((A @ lambda_c)[0] <= b[0])
        prog.AddLinearConstraint((A @ lambda_c)[1] <= b[1])
        prog.AddLinearConstraint((A @ lambda_c)[2] <= b[2])
        prog.AddLinearConstraint((A @ lambda_c)[3] <= b[3])
        prog.AddLinearConstraint((A @ lambda_c)[4] <= b[4])
        prog.AddLinearConstraint((A @ lambda_c)[5] <= b[5])

        solver = OsqpSolver()
        prog.SetSolverOption(solver.id(), "max_iter", 2000)

        result = solver.Solve(prog)

        # If we exceed iteration limits use the previous solution
        if not result.is_success():
            usol = self.u
        else:
            usol = result.GetSolution(u)
            self.u = usol

        return usol, prog

    def CalcTorques(self, context: Context, output: BasicVector) -> None:
        usol, _ = self.SetupAndSolveQP(context)
        # usol = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        output.SetFromVector(usol)


if __name__ == "__main__":
    Kp = np.diag([100, 0, 100])
    Kd = np.diag([10, 0, 10])
    W = np.diag([1, 0, 1])

    Wcom = np.eye(3)

    gains = OscGains(
        Kp,
        Kd,
        Wcom,
        Kp,
        Kd,
        W,
        Kp,
        Kd,
        W,
    )
    osc = OperationalSpaceController(gains)
