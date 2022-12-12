import numpy as np

from pydrake.systems.framework import Context
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.all import JacobianWrtVariable

from operational_space_tracking_objective import (
    OperationalSpaceTrackingObjective,
)


class CenterOfMassPositionTrackingObjective(OperationalSpaceTrackingObjective):
    """
    Track the center of mass of a robot
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
        kp: np.ndarray,
        kd: np.ndarray,
    ):
        super().__init__(plant, plant_context, kp, kd)

    def CalcY(self) -> np.ndarray:
        return self.plant.CalcCenterOfMassPositionInWorld(self.context).ravel()

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJ(self) -> np.ndarray:
        return self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.context,
            JacobianWrtVariable.kV,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

    def CalcJdotV(self) -> np.ndarray:
        return self.plant.CalcBiasCenterOfMassTranslationalAcceleration(
            self.context,
            JacobianWrtVariable.kV,
            self.plant.world_frame(),
            self.plant.world_frame(),
        ).ravel()
