import numpy as np
from dataclasses import dataclass

from pydrake.multibody.tree import Frame


@dataclass
class PointOnFrame:
    """
    Wrapper class which holds a BodyFrame and a vector, representing a point
    expressed in the BodyFrame
    """

    frame: Frame
    pt: np.ndarray
