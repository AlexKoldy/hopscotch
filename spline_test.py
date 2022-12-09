import numpy as np
from scipy.linalg import expm

from pydrake.trajectories import Trajectory, PiecewisePolynomial
from scipy.interpolate import CubicHermiteSpline


def make_exponential_spline(
    K: np.ndarray,
    A: np.ndarray,
    alpha: np.ndarray,
    pp_part: PiecewisePolynomial,
    n: int = 10,
) -> PiecewisePolynomial:
    """
    Helper function to approximate the solution to a linear system as
    a cubic spline, since ExponentialPlusPiecewisePolynomial doesn't have python bindings
    (https://drake.mit.edu/doxygen_cxx/classdrake_1_1trajectories_1_1_exponential_plus_piecewise_polynomial.html)
    """

    time_vect = np.linspace(pp_part.start_time(), pp_part.end_time(), n).tolist()
    knots = [
        np.expand_dims(
            K @ expm((t - time_vect[0]) * A) @ alpha + pp_part.value(t).ravel(), axis=1
        )
        for t in time_vect
    ]
    return PiecewisePolynomial.CubicShapePreserving(time_vect, knots)


if __name__ == "__main__":
    # Y = np.zeros((3, 2))
    # Y[0] = 0 * np.ones((2,))
    # Y[2] = 0.9 * np.ones((2,))

    # if 2 - 0 <= 1e-5:
    #     t_start = 2 - 1e-4

    # pp_part = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
    #     np.array([0, 2]), Y, np.zeros((3,)), np.zeros((3,))
    # )
    # A = np.fliplr(np.diag([1.0 / (50 * 0.9), 50 * 9.81]))
    # K = np.zeros((3, 2))
    # K[:2] = np.eye(2)
    # spline = make_exponential_spline(K, A, alip_state, pp_part)
    Y = np.zeros((3, 2))
    Y[0] = 0 * np.ones((2,))
    Y[2] = 0.9 * np.ones((2,))
    x_traj = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
        np.array([0, 10]), Y, np.zeros((3,)), np.zeros((3,))
    )

    # x_traj2 = PiecewisePolynomial.CubicHermite(
    #     np.array([0, 2]), np.array([10, 10]), np.array([0, 0])
    # )
    import matplotlib.pyplot as plt

    t = np.linspace(0, 2, 100)
    plt.figure()
    plt.plot(t, x_traj(t))
    plt.show()
