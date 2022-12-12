import numpy as np
from abc import ABC, abstractmethod
from typing import List


from pydrake.trajectories import Trajectory
from pydrake.systems.framework import Context
from pydrake.multibody.plant import MultibodyPlant


class OperationalSpaceTrackingObjective(ABC):
    """
    Abstract class representing a general operational space tracking objective
    Specific task spaces should implement
    - GetY
    - GetYdot
    - GetJ
    - GetJdotV
    With the assumption that the context will already be set to the correct state
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
        kp: np.ndarray,
        kd: np.ndarray,
    ):
        self.kp = kp
        self.kd = kd
        self.plant = plant
        self.context = plant_context

        self.J = None
        self.JdotV = None
        self.yddot_cmd = None

    def Update(self, t: float, y_des_traj: Trajectory):
        y = self.CalcY()
        ydot = self.CalcYdot()

        self.J = self.CalcJ()
        self.JdotV = self.CalcJdotV()

        yd = y_des_traj.value(t).ravel()
        yd_dot = y_des_traj.derivative(1).value(t).ravel()
        yd_ddot = y_des_traj.derivative(2).value(t).ravel()

        self.yddot_cmd = yd_ddot - self.kp @ (y - yd) - self.kd @ (ydot - yd_dot)

    def GetJ(self):
        return self.J

    def GetJdotV(self):
        return self.JdotV

    def GetYddotCmd(self):
        return self.yddot_cmd

    @abstractmethod
    def CalcJ(self) -> np.ndarray:
        pass

    @abstractmethod
    def CalcJdotV(self) -> np.ndarray:
        pass

    @abstractmethod
    def CalcY(self) -> np.ndarray:
        pass

    @abstractmethod
    def CalcYdot(self) -> np.ndarray:
        pass
