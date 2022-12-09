import numpy as np
from dataclasses import dataclass


@dataclass
class OscGains:
    k_p_com: np.ndarray  # CoM proportional constant
    k_d_com: np.ndarray  # CoM derivative constant
    w_com: np.ndarray  # CoM weight
    k_p_left_foot: np.ndarray  # left foot proportional constant
    k_d_left_foot: np.ndarray  # left foot derivative constant
    w_left_foot: np.ndarray  # left foot weight
    k_p_right_foot: np.ndarray  # right foot proportional constant
    k_d_right_foot: np.ndarray  # right foot derivative constant
    w_right_foot: np.ndarray  # right foot weight
