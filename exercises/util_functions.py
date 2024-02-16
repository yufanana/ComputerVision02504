"""
Utilty functions commonly used in exercises.
"""
import numpy as np
import itertools as it


# Ex 1.12
def Pi(ph):
    """
    Converts coordinates from homogeneous to inhomogeneous.
    ph : 4xn np.array
    p : 3xn np.array
    """
    p = ph[:-1] / ph[-1]  # divide by and remove last coordinate
    return p


# Ex 1.12
def PiInv(p):
    """
    Converts coordinates from inhomogeneous to homogeneous.
    p : 3xn np.array
    ph : 4xn np.array
    """
    ph = np.vstack((p, np.ones(p.shape[1])))
    return ph


def box3d(n=16):
    """Generate 3D points inside a cube with n-points along each edge"""
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


# Ex 1.13
# def projectpoints(K, R, t, Q):
#     """
#     Project 3D points in Q onto a 2D plane of a camera

#     K : 3 x 3, intrinsic camera matrix
#     R : 3 x 3, rotation matrix
#     t: 3 x 1, translation matrix
#     Q: 3 x n, 3D points matrix

#     P : 2 x n, 2D points matrix
#     """
#     Qh = PiInv(Q)  # 4 x n
#     pose = np.hstack((R, t))  # 3 x 4
#     Ph = K @ pose @ Qh  # 3 x n
#     P = Pi(Ph)  # 2 x n
#     return P


# Ex 2.2
def projectpoints(K, R, t, Q, distCoeffs=[]):
    """
    Project 3D points in Q onto a 2D plane of a camera with distortion.

    K : 3 x 3, intrinsic camera matrix
    R : 3 x 3, rotation matrix
    t: 3 x 1, translation matrix
    Q: 3 x n, 3D points matrix
    distCoeffs: [k3,k5,k7,...] distortion coefficients

    P : 2 x n, 2D points matrix
    """
    if Q.shape[0] != 3:
        raise ValueError("Q must be 3 x n")
    if K.shape != (3, 3):
        raise ValueError("K must be 3 x 3")
    if R.shape != (3, 3):
        raise ValueError("R must be 3 x 3")
    if t.shape != (3,1):
        raise ValueError("t must be 3 x 1")

    Qh = PiInv(Q)  # 4 x n
    Rt = np.hstack((R, t))  # 3 x 4
    qh = Rt @ Qh  # 3 x n
    q = Pi(qh)  # 2 x n
    qd = distort(q, distCoeffs)  # 2 x n
    Ph = K @ PiInv(qd)  # 3 x n
    P = Pi(Ph)  # 2 x n
    return P


def camera_intrinsic(f, c, alpha=1, beta=0):
    """
    Create a camera intrinsic matrix

    f : float, focal length
    c : 2D principal point (x,y)
    alpha : float, skew
    beta : float, aspect ratio
    """
    K = np.array([[f, beta * f, c[0]], [0, alpha * f, c[1]], [0, 0, 1]])
    return K


# Ex 2.2
def distort(q, distCoeffs):
    """
    Apply distortion to a 2D point on the image plane
    """
    r = np.sqrt((q[0]) ** 2 + (q[1]) ** 2)
    correction = 1
    for i in range(len(distCoeffs)):
        exp = 2 * (i + 1)
        correction += distCoeffs[i] * r**exp
    qd = q * correction
    return qd
