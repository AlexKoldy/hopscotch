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
from tracking_objectives.point_tracking_objective import PointPositionTrackingObjective
from tracking_objectives.center_of_mass_position_tracking_objective import (
    CenterOfMassPositionTrackingObjective,
)


class ComPlanner(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModelFromFile("3d_biped.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        self.com_traj_output_port_index = self.DeclareAbstractOutputPort(
            "com_traj",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcComTraj,
        ).get_index()

    def get_com_traj_output_port(self):
        return self.get_output_port(self.com_traj_output_port_index)

    def CalcComTraj(self, context: Context, output) -> None:
        Y = np.zeros((3, 2))
        Y[0] = 0 * np.ones((2,))
        Y[2] = 0.9 * np.ones((2,))
        com_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            np.array([0, 10]), Y, np.zeros((3,)), np.zeros((3,))
        )
        output.set_value(com_traj)
