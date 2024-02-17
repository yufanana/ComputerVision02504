"""
Utilty functions commonly used in exercises and quizzes.
"""
import numpy as np
import itertools as it
import matplotlib.pyplot as plt


# Ex 1.7
def point_line_distance(line, p):
    """
    Calculate shortest distance d between line l and 2D homogenous point p.

    line : 3x1 vector
    p : 3x1 vector
    """
    if p.shape != (3,1):
        raise ValueError("p must be a 3x1 homogenous vector")
    if line.shape != (3,1):
        raise ValueError("line must be a 3x1 homogenous vector")

    d = abs(line.T @ p) / (abs(p[2]) * np.sqrt(line[0] ** 2 + line[1] ** 2))
    return d

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


def projection_matrix(K, R, t):
    """
    Create a projection matrix from camera parameters.

    K : 3x3 np.array, intrinsic camera matrix
    R : 3x3 np.array, rotation matrix
    t : 3x1 np.array, translation matrix

    P : 3x4 np.array, projection matrix
    """
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 matrix")
    if R.shape != (3, 3):
        raise ValueError("R must be a 3x3 matrix")
    if t.shape != (3, 1):
        raise ValueError("t must be a 3x1 matrix")

    P = K @ np.hstack((R, t))
    return P


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


# Ex 3.2
def CrossOp(r):
    '''Returns the cross product operator of r.'''
    if r.shape == (3,1):
        r = r.flatten()
    if r.shape != (3,):
        raise ValueError('r must be a 3x1 vector')

    return np.array([
        [0,-r[2],r[1]],
        [r[2],0,-r[0]],
        [-r[1],r[0],0],
    ])


# Ex 3.3
def essential_matrix(R, t):
    '''Returns the essential matrix.'''
    return CrossOp(t) @ R

def fundamental_matrix(K1, R1, t1, K2, R2, t2):
    '''
    Returns the fundamental matrix, assuming camera 1 coordinates are
    on top of global coordinates.
    '''
    if R1.shape != (3,3) or R2.shape != (3,3):
        raise ValueError('R1 and R2 must be 3x3 matrices')
    if t1.shape == (3,) or t2.shape == (3,):
        t1 = t1.reshape(-1,1)
        t2 = t2.reshape(-1,1)
    if t1.shape != (3,1) or t2.shape != (3,1):
        raise ValueError('t1 and t2 must be 3x1 matrices')
    if K1.shape != (3,3) or K2.shape != (3,3):
        raise ValueError('K1 and K2 must be 3x3 matrices')

    # When the {camera1} and {camera2} are not aligned with {global}
    R_tilde = R2 @ R1.T
    t_tilde = t2 - R_tilde @ t1

    E = essential_matrix(R_tilde, t_tilde)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


# Ex 3.9
def DrawLine(l, shape):
    # Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame

    def in_frame(l_im):
        '''Returns the intersection point of the line with the image frame.'''
        q = np.cross(l.flatten(), l_im) # intersection point
        q = q[:2]/q[2]                  # convert to inhomogeneous
        if all(q>=0) and all(q+1<=shape[1::-1]):
            return q

    # 4 edge lines of the image
    lines = [
        [1, 0, 0],             # x = 0
        [0, 1, 0],             # y = 0
        [1, 0, 1-shape[1]],    # x = shape[1]
        [0, 1, 1-shape[0]],
    ]    # y = shape[0]

    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]

    if (len(P)==0):
        print("Line is completely outside image")
    plt.plot(*np.array(P).T)


# Ex 3.11
def triangulate(q_list,P_list):
    '''
    Triangulate a single 3D point seen by n cameras.

    q_list : nx2x1 list of 2x1 pixel points
    P_list : nx3x4, list of 3x4 camera projection matrices
    '''
    B = []  # 2n x 4 matrix
    for i in range(len(P_list)):
        qi = q_list[i]
        P_i = P_list[i]
        B.append(P_i[2]*qi[0]-P_i[0])
        B.append(P_i[2]*qi[1]-P_i[1])
    B = np.array(B)
    U, S, Vt = np.linalg.svd(B)
    Q = Vt[-1,:-1]/Vt[-1,-1]
    return Q
