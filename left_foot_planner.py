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


class LFPlanner(LeafSystem):
    def __init__(self, offline_traj):
    # def __init__(self):
        LeafSystem.__init__(self)

        self.offline_traj = offline_traj # ASSUMES THIS IS ONLY THE CALCULATED TRAJ

        self.plant = MultibodyPlant(0.0)
        self.parser = Parser(self.plant)
        self.parser.AddModelFromFile("3d_biped.urdf")
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        self.LF_traj_output_port_index = self.DeclareAbstractOutputPort(
            "LF_traj",
            lambda: AbstractValue.Make(PiecewisePolynomial()),
            self.CalcLFTraj,
        ).get_index()

    def get_LF_traj_output_port(self):
        return self.get_output_port(self.LF_traj_output_port_index)

    def CalcLFTraj(self, context: Context, output) -> None:
        
        t_start = 0  # [s]
        t_takeoff = 2  # [s]
        t_landing = 

        # if we're in mode 0: PREJUMP
        # lean forward
        traj = np.zeros((3, 2))
        traj[0] = np.array([0, 0.0])  # x_i & x_f
        traj[2] = np.array([1, 0.8])  # z_i & z_f
   
        LF_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            np.array([t_start, t_end]), traj, np.zeros((3,)), np.zeros((3,))
        )

        # if we're in mode 1:
        # so we've hit t_end
        # follow a parabola ~0.8 below the offline planner trajectory


        # once we hit the end of this trajectory
        # generate the LF projectile motion spline
        # just generate random spline or something

        # then, figure out when landed (probably cant use
        # time here)
        # start trying to stabilize the CoM
        # (see above)

        output.set_value(com_traj)
