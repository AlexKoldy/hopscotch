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
    def __init__(self, offline_traj, projectile_traj):
    # def __init__(self):
        LeafSystem.__init__(self)

        self.offline_traj = offline_traj # x_traj, z_traj, t_land, v_x_traj,  v_z_traj, f_x_traj, f_z_traj, f_vx_traj, f_vz_traj
        self.projectile_traj = projectile_traj

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

    def CalcLFTraj(self, context: Context, output) -> list:
        
        t_start = 0  # [s]
        t_begin_motion = t_start + 1 # CHANGE THIS
        t_takeoff = t_begin_motion + 0.3  # [s] # CAN CHANGE THIS based on time allocated for jump
        t_landing = self.offline_traj.t_land # SOLVE FOR THIS VIA PROJECTILE MOTION
        t_end = t_landing + 10 # if we want 10 seconds after the robot lands


        # if we're in mode 0: PREJUMP
        # lean forward
        traj_0 = np.zeros((3, 2))
        # traj[0] = np.array([0.0, 0.0])  # x_i & x_f
        traj_0[1] = np.array([0.2, 0.2]) # y_i & y_f !!! fix these values later
        # traj[2] = np.array([0.0, 0.0])  # z_i & z_f
        
        t_0 = np.array([t_start, t_begin_motion])
        LF_traj_0 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            t_0, traj_0, np.zeros((3,)), np.zeros((3,))
        )

        # mode 1: blue part of trajectory (jump)
        x_traj, z_traj, t_land, v_x_traj,  v_z_traj, f_x_traj, f_z_traj, f_vx_traj, f_vz_traj = self.offline_traj
        t_1 = np.array([t_begin_motion, t_takeoff])
        traj_1 = np.zeros((3,2))
        traj_1[0] = np.array([x_traj(0), x_traj(t_takeoff - t_begin_motion)]) + traj_0
        traj_1[0] = np.array([traj_0[1,0], 0]) #below the CoM
        traj_1[2] = np.array([z_traj(0), z_traj(t_takeoff - t_begin_motion)]) - 0.5 #leg length below CoM

        LF_traj_1 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            t_1, traj_1, np.zeros((3,)), np.zeros((3,))
        )


        # if we're in mode 2:
        # so we've hit t_takeoff
        # follow a parabola ~0.8 below the offline planner trajectory
        x_com, z_com = self.projectile_traj # this is the orange part of the 

        t_2 = np.array([t_takeoff, t_landing])
        # traj_2 = np.zeros((3,2))
        # traj_2[0] = np.array(x_com[0], x_com[1])
        # traj_2[1] # keep y under the CoM for the jump
        y_com = np.zeros(x_com.shape)
        traj_2 = np.array([[x_com], [y_com], [z_com] - 0.5])

        LF_traj_2 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(t_2, traj_2, np.zeros((3,)), np.zeros((3,))


        # once we hit the end of this trajectory
        # just generate random spline or something
        t_3 = np.array([t_landing, t_end])
        traj_3 = np.zeros((3,2))
        traj_3[0] = np.array([x_com[-1], x_com[-1]])
        traj_3[2] = np.array([0.3, 0.3])
        
        LF_traj_3 = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            t_3, traj_3, np.zeros((3,)), np.zeros((3,))
        )

        # then, figure out when landed (probably cant use
        # time here)
        # start trying to stabilize the CoM
        # (see above)
    

        output.set_value(com_traj)

        return LF_traj_0, LF_traj_1, LF_traj_2, LF_traj_3
