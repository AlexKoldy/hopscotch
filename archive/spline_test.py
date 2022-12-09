from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np

t = np.array([0, 1, 2])
x = np.array([0, 1, 2])
y = np.array([0, 1, 2])
z = np.array([0, 1, 0])

x = CubicSpline(t, x)
y = CubicSpline(t, y)
z = CubicSpline(t, z)

t = np.linspace(0, 2, 100)
plt.figure()
plt.plot(t, z(t))
plt.plot(t, z(t, 1))
plt.show()
