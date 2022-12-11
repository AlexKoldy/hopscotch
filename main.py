"""
RUN offline_planner.py and generate some plots
"""

import numpy as np
import matplotlib.pyplot as plt

import importlib
import offline_planner
importlib.reload(offline_planner)
from offline_planner import OfflinePlanner


planner = OfflinePlanner()

print("d_0: ", planner.d_0)

t_last = 0.375
distance = np.array([1.50, 0])

x_traj, z_traj, t_land, v_x_traj,  v_z_traj, f_x_traj, f_z_traj, f_vx_traj, f_vz_traj = planner.find_com_trajectory(planner.aslip.x_0 * 2, t_last, distance)


# print(planner.find_com_trajectory(planner.aslip.x_0 * 2, t_last, distance))

t = np.linspace(0, t_last, 100)
xx = x_traj(t)[-1]
zz = z_traj(t)[-1]

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

# print(len(x_com))

t = np.linspace(0, t_last, 100)

plt.figure()
plt.title("com x vs com z")
plt.plot(x_traj(t), z_traj(t))
plt.plot(x_com, z_com)
plt.xlabel("CoM x")
plt.ylabel("CoM z")
plt.show()

t2 = np.linspace(t_last, t_max, (t_max - t_last)/dt + 2)

plt.figure()
plt.title("time vs CoM z")
plt.plot(t, z_traj(t))
plt.plot(t2, z_com)
plt.xlabel("time")
plt.ylabel("CoM z")
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
