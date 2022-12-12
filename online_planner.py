import numpy as np
from typing import Tuple

from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import Context, LeafSystem, BasicVector
from pydrake.math import RigidTransform
from pydrake.trajectories import Trajectory, PiecewisePolynomial
from pydrake.common.value import AbstractValue
from pydrake.solvers import MathematicalProgram

from osc_gains import OscGains
from point_on_frame import PointOnFrame
from point_tracking_objective import PointPositionTrackingObjective
from center_of_mass_position_tracking_objective import (
    CenterOfMassPositionTrackingObjective,
)


class OnlinePlanner(LeafSystem):
    def __init__(
        self,
        com_mode_0_traj,
        com_mode_1_traj,
        com_mode_2_traj,
        lf_mode_0_traj,
        lf_mode_1_traj,
        lf_mode_2_traj,
        rf_mode_0_traj,
        rf_mode_1_traj,
        rf_mode_2_traj,
        fsm
    ):
        LeafSystem.__init__(self)
        
        self.fsm = fsm
        self.com_mode_0_traj = com_mode_0_traj
        self.com_mode_1_traj = com_mode_1_traj
        self.com_mode_2_traj = com_mode_2_traj
        self.lf_mode_0_traj = lf_mode_0_traj
        self.lf_mode_1_traj = lf_mode_1_traj
        self.lf_mode_2_traj = lf_mode_2_traj
        self.rf_mode_0_traj = rf_mode_0_traj
        self.rf_mode_1_traj = rf_mode_1_traj
        self.rf_mode_2_traj = rf_mode_2_traj

        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModelFromFile("3d_biped.urdf")
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

        self.robot_state_input_port_index = self.DeclareVectorInputPort(
            "x", self.plant.num_positions() + self.plant.num_velocities()
        ).get_index()

        self.com_traj_output_port_index = self.DeclareAbstractOutputPort(
            "com_traj",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcComTraj,
        ).get_index()

        self.right_foot_traj_output_port_index = self.DeclareAbstractOutputPort(
            "right_foot_traj",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcRFTraj,
        ).get_index()
        self.left_foot_traj_output_port_index = self.DeclareAbstractOutputPort(
            "left_foot_traj",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcLFTraj,
        ).get_index()

        self.stance_foot_pos = None

    def get_com_traj_output_port(self):
        return self.get_output_port(self.com_traj_output_port_index)

    def get_left_foot_traj_output_port(self):
        return self.get_output_port(self.left_foot_traj_output_port_index)

    def get_right_foot_traj_output_port(self):
        return self.get_output_port(self.right_foot_traj_output_port_index)

    def get_state_input_port(self):
        return self.get_input_port(self.robot_state_input_port_index)

    def CalcComTraj(self, context: Context, output) -> None:
        fsm = self.fsm
        com_traj = self.com_mode_0_traj
        # Lean forward
        if fsm == 0:
            com_traj = self.com_mode_0_traj
        # Start jump
        elif fsm == 1:
            com_traj = self.com_mode_1_traj
        # Projectile motion
        elif fsm == 2:
            com_traj = self.com_mode_2_traj
        # Stabilize
        elif fsm == 3:
            if self.stance_foot_pos:
                stance_foot_pos = self.stance_foot_pos
            else:
                stance_foot = self.contact_points["right leg"]
                stance_foot_pos = self.plant.CalcPointsPositions(
                    self.plant_context,
                    stance_foot.frame,
                    stance_foot.pt,
                    self.plant.world_frame(),
                ).ravel()
                self.stance_foot_pos = stance_foot_pos
            traj = np.zeros((3, 2))
            traj[0] = np.array([stance_foot_pos[0], stance_foot_pos[0]])  # x_i & x_f
            traj[1] = np.array([stance_foot_pos[1], stance_foot_pos[1]])
            traj[2] = np.array([0.8, 0.8])  # z_i & z_f
            t_start = 0  # [s]
            t_end = 2  # [s]
            com_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([t_start, t_end]), traj, np.zeros((3,)), np.zeros((3,))
            )

        output.set_value(com_traj)

    def CalcLFTraj(self, context: Context, output) -> None:
        fsm = self.fsm
        lf_traj = self.lf_mode_0_traj
        # Lean forward
        if fsm == 0:
            lf_traj = self.lf_mode_0_traj
        # Start jump
        elif fsm == 1:
            lf_traj = self.lf_mode_1_traj
        # Projectile motion
        elif fsm == 2:
            lf_traj = self.lf_mode_2_traj
        # Stabilize
        elif fsm == 3:
            if self.stance_foot_pos:
                stance_foot_pos = self.stance_foot_pos
            else:
                stance_foot = self.contact_points["right leg"]
                stance_foot_pos = self.plant.CalcPointsPositions(
                    self.plant_context,
                    stance_foot.frame,
                    stance_foot.pt,
                    self.plant.world_frame(),
                ).ravel()
            traj = np.zeros((3, 2))
            traj[0] = np.array([stance_foot_pos[0], stance_foot_pos[0]])  # x_i & x_f
            traj[1] = np.array([0.2, 0.2])
            traj[2] = np.array([0.3, 0.3])  # z_i & z_f
            t_start = 0  # [s]
            t_end = 2  # [s]
            lf_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([t_start, t_end]), traj, np.zeros((3,)), np.zeros((3,))
            )

        output.set_value(lf_traj)

    def CalcRFTraj(self, context: Context, output) -> None:
        fsm = self.fsm
        rf_traj = self.rf_mode_0_traj
        # Lean forward
        if fsm == 0:
            rf_traj = self.rf_mode_0_traj
        # Start jump
        elif fsm == 1:
            rf_traj = self.rf_mode_1_traj
        # Projectile motion
        elif fsm == 2:
            rf_traj = self.rf_mode_2_traj
        # Stabilize
        elif fsm == 3:
            if self.stance_foot_pos:
                stance_foot_pos = self.stance_foot_pos
            else:
                stance_foot = self.contact_points["right leg"]
                stance_foot_pos = self.plant.CalcPointsPositions(
                    self.plant_context,
                    stance_foot.frame,
                    stance_foot.pt,
                    self.plant.world_frame(),
                ).ravel()
            traj = np.zeros((3, 2))
            traj[0] = np.array([stance_foot_pos[0], stance_foot_pos[0]])  # x_i & x_f
            traj[1] = np.array([0, 0])
            traj[2] = np.array([0, 0])  # z_i & z_f
            t_start = 0  # [s]
            t_end = 2  # [s]
            rf_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
                np.array([t_start, t_end]), traj, np.zeros((3,)), np.zeros((3,))
            )

        output.set_value(rf_traj)