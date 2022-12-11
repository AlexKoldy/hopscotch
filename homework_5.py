import control

import numpy as np

m = 1.0  # [kg]
M = 5.0  # [kg]
l = 2.0  # [m]
g = 9.8  # [m/s^2]

A = np.array(
    [
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, m * g / M, 0, 0],
        [0, (m + M) * g / (M * l), 0, 0],
    ]
)

B = np.array([[0], [0], [1 / M], [1 / (M * l)]])

# C = np.hstack((B, A @ B, A @ A @ B, A @ A @ A @ B))


C_s = np.array([[1.0, 0, 0, 0]])
C_theta = np.array([[0, 1.0, 0, 0]])

O_s = np.vstack((C_s, C_s @ A, C_s @ A @ A, C_s @ A @ A @ A))
O_theta = np.vstack((C_theta, C_theta @ A, C_theta @ A @ A, C_theta @ A @ A @ A))

# print(np.linalg.matrix_rank(O_s))
# print(np.linalg.matrix_rank(O_theta))

Q = np.eye(4)

R = 1

K, S, E = control.lqr(A, B, Q, R)
print(K)

C_both = np.array([[1, 1, 0, 0]])
L_both, S, E = control.lqr(A.T, np.array([[1, 1, 0, 0]]).T, Q, R)
L_both = L_both.T


L_s, S, E = control.lqr(A.T, np.array([[1, 0, 0, 0]]).T, Q, R)
L_s = L_s.T

print("eigenvalues")
# print(A - B @ K)
print("(i)")
print(np.linalg.eigvals(A - B @ K))

print("(ii)")
K = -K
L_both = -L_both
L_s = -L_s

A_both = np.vstack(
    (np.hstack((A, B @ K)), np.hstack((-L_both @ C_both, A + B @ K + L_both @ C_both)))
)
print(np.linalg.eigvals(A_both))

print("(iii)")
A_s = np.vstack((np.hstack((A, B @ K)), np.hstack((-L_s @ C_s, A + B @ K + L_s @ C_s))))

print(np.linalg.eigvals(A_s))
