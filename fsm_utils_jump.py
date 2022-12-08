import numpy as np

# Global variables for gait timing
PRE_JUMP = 0
FLIGHT = 1
POST_JUMP = 2

HAS_JUMPED = False



def get_fsm(z, height) -> int:
    if z <= height and not HAS_JUMPED:
        return PRE_JUMP
    elif z >= height:
        HAS_JUMPED = True
        return FLIGHT
    else:
        return POST_JUMP
