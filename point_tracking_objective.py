import numpy as np

from pydrake.systems.framework import Context
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.all import JacobianWrtVariable

from point_on_frame import PointOnFrame
from operational_space_tracking_objective import (
    OperationalSpaceTrackingObjective,
)


class PointPositionTrackingObjective(OperationalSpaceTrackingObjective):
    """
    Track the position of a point as measured in the world frame
    """

    def __init__(
        self,
        plant: MultibodyPlant,
        plant_context: Context,
        kp: np.ndarray,
        kd: np.ndarray,
        pt_to_track: PointOnFrame,
    ):
        super().__init__(plant, plant_context, kp, kd)
        self.pt_to_track = pt_to_track

    def CalcY(self) -> np.ndarray:
        return self.plant.CalcPointsPositions(
            self.context,
            self.pt_to_track.frame,
            self.pt_to_track.pt,
            self.plant.world_frame(),
        ).ravel()

    def CalcJ(self) -> np.ndarray:
        pt_to_track = self.pt_to_track
        return self.plant.CalcJacobianTranslationalVelocity(
            self.context,
            JacobianWrtVariable.kV,
            pt_to_track.frame,
            pt_to_track.pt,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

    def CalcYdot(self) -> np.ndarray:
        return (self.CalcJ() @ self.plant.GetVelocities(self.context)).ravel()

    def CalcJdotV(self) -> np.ndarray:
        pt_to_track = self.pt_to_track
        return self.plant.CalcBiasTranslationalAcceleration(
            self.context,
            JacobianWrtVariable.kV,
            pt_to_track.frame,
            pt_to_track.pt,
            self.plant.world_frame(),
            self.plant.world_frame(),
        ).ravel()
