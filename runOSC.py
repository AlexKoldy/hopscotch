import pydot
import numpy as np
from IPython.display import SVG, display

from pydrake.all import (
    # ConnectMeshcatVisualizer,
    JointIndex,
    Simulator,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    RigidTransform,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    ConstantVectorSource,
    ConstantValueSource,
    PiecewisePolynomial,
    AbstractValue,
    HalfSpace,
    CoulombFriction,
    StartMeshcat,
)

from osc_gains import OscGains


import importlib

import osc
importlib.reload(osc)
from osc import OperationalSpaceController

import online_planner
importlib.reload(online_planner)
from online_planner import OnlinePlanner

import offline_planner
importlib.reload(offline_planner)
from offline_planner import OfflinePlanner
from finite_state_machine import FiniteStateMachine


import point_tracking_objective
importlib.reload(point_tracking_objective)

import operational_space_tracking_objective
importlib.reload(operational_space_tracking_objective)

import center_of_mass_position_tracking_objective
importlib.reload(center_of_mass_position_tracking_objective)

# # Start meshcat simulation software
# meshcat = StartMeshcat()

# Create block diagram of system
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0005)
X_WG = HalfSpace.MakePose(
np.array([0, 0, 1]),
np.zeros(3),
)
plant.RegisterCollisionGeometry(
plant.world_body(), X_WG, HalfSpace(), "collision", CoulombFriction(1.0, 1.0)
)
parser = Parser(plant)
parser.AddModelFromFile("3d_biped.urdf")
plant.Finalize()
num_positions = plant.num_positions()
num_joints = plant.num_joints()

print("floating base:")
for j in plant.GetFloatingBaseBodies():
    body = plant.get_body(j)
    start = body.floating_positions_start()
    print(body.name())

print("joints:")
# Print names of each joint
for i in range(num_joints):
    index = JointIndex(i)
    j = plant.get_joint(index)
    print(j.name())

# Print number of positions
print(f"number of positions: {num_positions}")
print(f"number of joints: {num_joints}")

# Get OSC gains
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

# Setup offline trajectory optimization
offline_planner = OfflinePlanner()

t_begin_motion = 0.3 # robot starts moving after 0.3 seconds
t_takeoff = 0.3 # takeoff within 0.3 seconds
distance = np.array([1.0, 0])
(
com_mode_0_traj,
com_mode_1_traj,
com_mode_2_traj,
lf_mode_0_traj,
lf_mode_1_traj,
lf_mode_2_traj,
rf_mode_0_traj,
rf_mode_1_traj,
rf_mode_2_traj,
) = offline_planner.find_trajectories(offline_planner.aslip.x_0 * 2, t_begin_motion+t_takeoff, distance)

fsm = FiniteStateMachine()

planner = builder.AddSystem(
OnlinePlanner(com_mode_0_traj, com_mode_1_traj, com_mode_2_traj, lf_mode_0_traj,lf_mode_1_traj,lf_mode_2_traj,rf_mode_0_traj,rf_mode_1_traj,rf_mode_2_traj, fsm)
)

# Wire plant to planner
builder.Connect(plant.get_state_output_port(), planner.get_state_input_port())

# Wire OSC to plant
osc = builder.AddSystem(OperationalSpaceController(gains, fsm))
builder.Connect(osc.get_output_port(), plant.get_actuation_input_port())

# Wire OSC inputs
builder.Connect(plant.get_state_output_port(), osc.get_state_input_port())

builder.Connect(
planner.get_com_traj_output_port(), osc.get_traj_input_port("com_traj")
)
builder.Connect(
planner.get_left_foot_traj_output_port(), osc.get_traj_input_port("left_foot_traj")
)
builder.Connect(
planner.get_right_foot_traj_output_port(), osc.get_traj_input_port("right_foot_traj")
)

# TODO: Adjust target wlaking speed here
# walking_speed = 0.5  # walking speed in m/s

# osc = builder.AddSystem(OperationalSpaceController(gains))
# planner = builder.AddSystem(footstep_planner.LipTrajPlanner())
# speed_src = builder.AddSystem(ConstantVectorSource(np.array([walking_speed])))
# base_traj_src = builder.AddSystem(
#     ConstantValueSource(
#         AbstractValue.Make(
#             PiecewisePolynomial(
#                 np.zeros(
#                     1,
#                 )
#             )
#         )
#     )
# )

# Wire planner inputs
# builder.Connect(plant.get_state_output_port(), planner.get_state_input_port())
# builder.Connect(speed_src.get_output_port(), planner.get_walking_speed_input_port())

# Wire OSC inputs
# builder.Connect(plant.get_state_output_port(), osc.get_state_input_port())
# builder.Connect(
#     planner.get_swing_foot_traj_output_port(),
#     osc.get_traj_input_port("swing_foot_traj"),
# )
# builder.Connect(
#     planner.get_com_traj_output_port(), osc.get_traj_input_port("com_traj")
# )
# builder.Connect(
#     base_traj_src.get_output_port(), osc.get_traj_input_port("base_joint_traj")
# )

# Add the visualizer
vis_params = MeshcatVisualizerParams(publish_period=0.01)
MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params=vis_params)

# Wire OSC to plant
# builder.Connect(osc.get_output_port(), plant.get_actuation_input_port())

# simulate
diagram = builder.Build()
display(
SVG(
    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=2))[
        0
    ].create_svg()
)
)

sim_time = 10.0
simulator = Simulator(diagram)
simulator.Initialize()
simulator.set_target_realtime_rate(1)

plant_context = diagram.GetMutableSubsystemContext(
plant, simulator.get_mutable_context()
)
# print(plant_context)
q = np.zeros((num_positions,))

# quaternion
q[0] = 1
q[1] = 0
q[2] = 0
q[3] = 0

# x, y, z
q[4] = 0
q[5] = 0
q[6] = 1.15

q[8] = 0.2
q[12] = -0.2

q[10] = 0  # left knee
q[14] = 0  # right knee

plant.SetPositions(plant_context, q)

import time

time.sleep(10)

simulator.AdvanceTo(sim_time)

# Set the robot state
# plant_context = diagram.GetMutableSubsystemContext(
#     plant, simulator.get_mutable_context())
# q = np.zeros((plant.num_positions(),))
# q[1] = 0.8
# theta = -np.arccos(q[1])
# q[3] = theta
# q[4] = -2 * theta
# q[5] = theta
# q[6] = -2 * theta
# plant.SetPositionsplant_context, q)

# Simulate the robot
# simulator.AdvanceTo(sim_time)
