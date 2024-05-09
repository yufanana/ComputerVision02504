"""
Utilty functions commonly used in exercises and quizzes.
"""
import cv2
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib as mpl


# Ex 1.7
def point_line_distance(line: np.ndarray, p: np.ndarray):
    """
    Calculate shortest distance d between line l and 2D homogenous point p.

    Args:
        line: homogenous line, shape (3, 1)
        p: 3x1 vector, shape (3, 1)

    Returns:
        d (float): distance
    """
    if p.shape != (3, 1):
        raise ValueError("p must be a 3x1 homogenous vector")
    if line.shape != (3, 1):
        raise ValueError("line must be a 3x1 homogenous vector")

    d = abs(line.T @ p) / (abs(p[2]) * np.sqrt(line[0] ** 2 + line[1] ** 2))
    return d


# Ex 1.12
def Pi(ph: np.ndarray):
    """
    Converts coordinates from homogeneous to inhomogeneous.

    Args:
        ph (np.array): shape (n+1,m)

    Returns:
        p (np.array): shape (n,m)
    """
    p = ph[:-1] / ph[-1]  # divide by and remove last coordinate
    return p


# Ex 1.12
def PiInv(p: np.ndarray):
    """
    Converts coordinates from inhomogeneous to homogeneous.

    Args:
        p (np.array): shape (n,m)

    Returns:
        ph (np.array): shape (n+1,m)
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


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """
    Create a projection matrix from camera parameters.

    Args:
        K (np.array): intrinsic camera matrix, shape (3, 3)
        R (np.array): rotation matrix, shape (3, 3)
        t (np.array): translation matrix, shape (3, 1)

    Returns:
        P (np.array): projection matrix, shape (3, 4)
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
def project_points(
    K: np.ndarray, R: np.ndarray, t: np.ndarray, Q: np.ndarray, distCoeffs=[]
):
    """
    Project 3D points in Q onto a 2D plane of a camera with distortion.

    Args:
        K (np.ndarray): intrinsic camera matrix, shape (3, 3)
        R (np.ndarray): rotation matrix, shape (3, 3)
        t (np.ndarray): translation matrix. shape (3, 1)
        Q (np.ndarray): 3D points matrix, shape (3, n)
        distCoeffs: [k3,k5,k7,...] distortion coefficients

    Returns:
        P : 2D points matrix, shape (2, n)
    """
    if Q.shape[0] != 3:
        raise ValueError("Q must be 3 x n")
    if K.shape != (3, 3):
        raise ValueError("K must be 3 x 3")
    if R.shape != (3, 3):
        raise ValueError("R must be 3 x 3")
    if t.shape != (3, 1):
        raise ValueError("t must be 3 x 1")

    Qh = PiInv(Q)  # 4 x n
    Rt = np.hstack((R, t))  # 3 x 4
    qh = Rt @ Qh  # 3 x n
    q = Pi(qh)  # 2 x n
    qd = distort(q, distCoeffs)  # 2 x n
    Ph = K @ PiInv(qd)  # 3 x n
    P = Pi(Ph)  # 2 x n
    return P


def camera_intrinsic(f: float, c: tuple[float, float], alpha=1.0, beta=0.0):
    """
    Create a camera intrinsic matrix

    Args:
        f (float): focal length
        c (tuple): 2D principal point (x,y)
        alpha (float): skew
        beta (float): aspect ratio

    Returns:
        K (np.array): intrinsic camera matrix, shape (3, 3)
    """
    K = np.array(
        [
            [f, beta * f, c[0]],
            [0, alpha * f, c[1]],
            [0, 0, 1],
        ]
    )
    return K


# Ex 2.2
def distort(q: np.ndarray, distCoeffs: list):
    """
    Apply distortion to a 2D point on the image plane

    Args:
        q (np.array): 2D points, shape (2, n)
        distCoeffs (list[float]): list of distortion coefficients

    Returns:
        qd (np.array): distorted 2D points, shape (2, n)
    """
    r = np.sqrt((q[0]) ** 2 + (q[1]) ** 2)
    correction = 1
    for i in range(len(distCoeffs)):
        exp = 2 * (i + 1)
        correction += distCoeffs[i] * r**exp
    qd = q * correction
    return qd


# Ex 2.4
def undistort_image(im, distCoeffs: list, K: np.ndarray):
    """
    Returns an undistorted image using the camera matrix K and
    distortion coefficients distCoeffs.

    Args:
        im : input image
        distCoeffs (list[float]) : list of distortion coefficients
        K (np.ndarray): intrinsic camera matrix

    Returns:
        im_undistorted: undistorted image
    """
    # meshgrid: placeholder for pixel coordinates
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
    p = np.stack((x, y, np.ones(x.shape))).reshape(3, -1)  # 3 x n, homogenous

    q = Pi(np.linalg.inv(K) @ p)  # 2 x n
    qd = distort(q, distCoeffs)  # 2 x n
    p_d = K @ PiInv(qd)  # 3 x n

    x_d = p_d[0].reshape(x.shape).astype(np.float32)
    y_d = p_d[1].reshape(y.shape).astype(np.float32)
    assert (p_d[2] == 1).all(), "You did a mistake somewhere"
    # use bilinear interpolation to remap the image
    im_undistorted = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)
    return im_undistorted


# Ex 2.7
def normalize2d(q: np.ndarray):
    """
    Normalize 2D points to have mean 0 and sd 1.

    Args:
        q : 2 x n, 2D points

    Returns
        qn : 2 x n, normalized 2D points
        T : 3 x 3, normalization matrix
    """
    if q.shape[0] != 2:
        raise ValueError("q must have 2 rows")
    if q.shape[1] < 2:
        raise ValueError("At least 2 points are required to normalize")

    mu = np.mean(q, axis=1).reshape(-1, 1)
    mu_x = mu[0].item()
    mu_y = mu[1].item()
    std = np.std(q, axis=1).reshape(-1, 1)
    std_x = std[0].item()
    std_y = std[1].item()
    Tinv = np.array([[std_x, 0, mu_x], [0, std_y, mu_y], [0, 0, 1]])
    T = np.linalg.inv(Tinv)
    qn = T @ PiInv(q)
    qn = Pi(qn)
    return qn, T


# Ex 2.8
def hest(q1: np.ndarray, q2: np.ndarray, normalize=False):
    """
    Calculate the homography matrix from n sets of 2D points

    Args:
        q1 : 2 x n, 2D points in the first image
        q2 : 2 x n, 2D points in the second image
        normalize : bool, whether to normalize the points

    Returns:
        H : 3 x 3, homography matrix
    """
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("Number of points in q1 and q2 must be equal")
    if q1.shape[1] < 4:
        raise ValueError(
            "At least 4 points are required to estimate a homography",
        )
    if q1.shape[0] != 2 or q2.shape[0] != 2:
        raise ValueError("q1 and q2 must have 2 rows")

    if normalize:
        q1, T1 = normalize2d(q1)
        q2, T2 = normalize2d(q2)

    n = q1.shape[1]
    B = []
    for i in range(n):
        x1, y1 = q1[:, i]
        x2, y2 = q2[:, i]
        Bi = np.array(
            [
                [0, -x2, x2 * y1, 0, -y2, y2 * y1, 0, -1, y1],
                [x2, 0, -x2 * x1, y2, 0, -y2 * x1, 1, 0, -x1],
                [-x2 * y1, x2 * x1, 0, -y2 * y1, y2 * x1, 0, -y1, x1, 0],
            ],
        )
        B.append(Bi)
    B = np.array(B).reshape(-1, 9)
    U, S, Vt = np.linalg.svd(B)
    H = Vt[-1].reshape(3, 3)
    H = H.T
    if normalize:
        H = np.linalg.inv(T1) @ H @ T2
    return H


# Ex 9.1
def R(theta_x: float, theta_y: float, theta_z: float):
    """
    Angles in radians.

    Returns : Rz @ Ry @ Rx
    """
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ],
    )
    Ry = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ],
    )
    Rz = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ],
    )
    return Rz @ Ry @ Rx


# Ex 3.2
def CrossOp(r: np.ndarray):
    """
    Cross product operator of r

    Args:
        r : 3x1 vector

    Returns:
        R : 3x3 matrix
    """
    if r.shape == (3, 1):
        r = r.flatten()
    if r.shape != (3,):
        raise ValueError("r must be a 3x1 vector")

    return np.array(
        [
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0],
        ],
    )


# Ex 3.3
def essential_matrix(R: np.ndarray, t: np.ndarray):
    """
    Returns the essential matrix.

    Args:
        R : 3x3 matrix, rotation matrix
        t : 3x1 matrix, translation matrix

    Returns:
        E : 3x3 matrix, essential matrix
    """
    return CrossOp(t) @ R


def fundamental_matrix(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
):
    """
    Returns the fundamental matrix, assuming camera 1 coordinates are
    on top of global coordinates.

    Args:
        K1 : 3x3 matrix, intrinsic matrix of camera 1
        R1 : 3x3 matrix, rotation matrix of camera 1
        t1 : 3x1 matrix, translation matrix of camera 1
        K2 : 3x3 matrix, intrinsic matrix of camera 2
        R2 : 3x3 matrix, rotation matrix of camera 2
        t2 : 3x1 matrix, translation matrix of camera 2

    Returns:
        F : 3x3 matrix, fundamental matrix
    """
    if R1.shape != (3, 3) or R2.shape != (3, 3):
        raise ValueError("R1 and R2 must be 3x3 matrices")
    if t1.shape == (3,) or t2.shape == (3,):
        t1 = t1.reshape(-1, 1)
        t2 = t2.reshape(-1, 1)
    if t1.shape != (3, 1) or t2.shape != (3, 1):
        raise ValueError("t1 and t2 must be 3x1 matrices")
    if K1.shape != (3, 3) or K2.shape != (3, 3):
        raise ValueError("K1 and K2 must be 3x3 matrices")

    # When the {camera1} and {camera2} are not aligned with {global}
    R_tilde = R2 @ R1.T
    t_tilde = t2 - R_tilde @ t1

    E = essential_matrix(R_tilde, t_tilde)
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    return F


# Ex 3.9
def DrawLine(l: np.ndarray, shape: tuple[int, int]):
    """
    Draws a line on an the 2nd plot.

    Args:
        l (np.ndarray): epipolar line in homogenous coordinates, shape (3,1)
        shape (tuple[int,int]) : shape of the image
    """
    # Checks where the line intersects the four sides of the image
    # and finds the two intersections that are within the frame

    def in_frame(l_im):
        """Returns the intersection point of the line with the image frame."""
        q = np.cross(l.flatten(), l_im)  # intersection point
        q = q[:2] / q[2]  # convert to inhomogeneous
        if all(q >= 0) and all(q + 1 <= shape[1::-1]):
            return q

    # 4 edge lines of the image
    lines = [
        [1, 0, 0],  # x = 0
        [0, 1, 0],  # y = 0
        [1, 0, 1 - shape[1]],  # x = shape[1]
        [0, 1, 1 - shape[0]],  # y = shape[0]
    ]

    P = [in_frame(l_im) for l_im in lines if in_frame(l_im) is not None]

    if len(P) == 0:
        print("Line is completely outside image")
    plt.plot(*np.array(P).T)


# Ex 3.11
def triangulate(
    q_list: list[np.ndarray], P_list: list[np.ndarray]
) -> np.ndarray:
    """
    Triangulate a single 3D point seen by n cameras.

    Args:
        q_list : nx2x1 list of 2x1 pixel points
        P_list : nx3x4, list of 3x4 camera projection matrices

    Returns:
        Q : 3x1 vector, 3D point
    """
    B = []  # 2n x 4 matrix
    for i in range(len(P_list)):
        qi = q_list[i]
        P_i = P_list[i]
        B.append(P_i[2] * qi[0] - P_i[0])
        B.append(P_i[2] * qi[1] - P_i[1])
    B = np.array(B)
    U, S, Vt = np.linalg.svd(B)
    Q = Vt[-1, :-1] / Vt[-1, -1]
    Q = Q.reshape(3, 1)
    return Q


# Ex 4.2
def pest(Q: np.ndarray, q: np.ndarray, normalize=False):
    """
    Estimate projection matrix using direct linear transformation.

    Args:
        Q : 3 x n array of 3D points
        q : 2 x n array of 2D points
        normalize : bool, whether to normalize the 2D points

    Returns:
        P : 3 x 4 projection matrix
    """
    if Q.shape[0] != 3:
        raise ValueError("Q must be a 3 x n array of 3D points")
    if q.shape[0] != 2:
        raise ValueError("q must be a 2 x n array of 2D points")

    if normalize:
        q, T = normalize2d(q)

    q = PiInv(q)  # 3 x n
    Q = PiInv(Q)  # 4 x n
    n = Q.shape[1]  # number of points
    B = []
    for i in range(n):
        Qi = Q[:, i]
        qi = q[:, i]
        Bi = np.kron(Qi, CrossOp(qi))
        B.append(Bi)
    B = np.array(B).reshape(3 * n, 12)
    U, S, Vt = np.linalg.svd(B)
    P = Vt[-1].reshape(4, 3)
    P = P.T
    if normalize:
        P = np.linalg.inv(T) @ P
    return P


def compute_rmse(q_true: np.ndarray, q_est: np.ndarray):
    """
    Returns the root mean square error between the true and estimated 2D points.

    Args:
        q_true: 2 x n array of true 2D points
        q_est: 2 x n array of estimated 2D points
    """
    if q_true.shape[0] != 2 or q_est.shape[0] != 2:
        raise ValueError("q_true and q_est must be 2 in the first dimension")
    if q_true.shape[1] != q_est.shape[1]:
        raise ValueError("q_true and q_est must have the same number of points")
    se = (q_est - q_true) ** 2
    return np.sqrt(np.mean((se)))


# Ex 4.3
def checkerboard_points(n: int, m: int):
    """
    Generate 3D points of a checkerboard with n x m squares.

    Returns:
        points : 3 x (n*m) array of 3D points
    """
    points = np.array(
        [
            (
                i - (n - 1) / 2,
                j - (m - 1) / 2,
                0,
            )
            for i in range(n)
            for j in range(m)
        ],
    ).T
    return points


# Ex 4.5
def estimate_homographies(Q_omega, qs):
    """
    Estimate homographies for each view.

    Args:
        Q_omega : 3 x (nxm) array of untransformed 3D points
        qs : list of 2xn arrays corresponding to each view, e.g. [qa, qb, qc]

    Returns:
        Hs : list of 3x3 homographies for each view
    """
    Hs = []
    Q = Q_omega[:2]  # remove 3rd row of zeros
    for q in qs:
        H = hest(q, Q)  # TODO: why hest(q, Q) instead of hest(Q, q)?
        Hs.append(H)
    return Hs


# Ex 4.6
def form_vi(H: np.ndarray, a: int, b: int):
    """
    Form 1x6 vector vi using H and indices alpha, beta.

    Args:
        H (np.ndarray) : 3x3 homography
        a, b (int) : indices alpha, beta

    Returns:
        vi (np.ndarray) : 1x6 vector
    """
    # Use zero-indexing here. Notes uses 1-indexing.
    a = a - 1
    b = b - 1
    vi = np.array(
        [
            H[0, a] * H[0, b],
            H[0, a] * H[1, b] + H[1, a] * H[0, b],
            H[1, a] * H[1, b],
            H[2, a] * H[0, b] + H[0, a] * H[2, b],
            H[2, a] * H[1, b] + H[1, a] * H[2, b],
            H[2, a] * H[2, b],
        ],
    )
    vi = vi.reshape(1, 6)
    return vi


# Ex 4.6
def estimate_b(Hs: np.ndarray):
    """
    Estimate b matrix used Zhang's method for camera calibration.

    Args:
        Hs (np.ndarray) : list of 3x3 homographies for each view

    Returns:
        b (np.ndarray) : 6x1 vector
    """
    V = []  # coefficient matrix
    # Create constraints in matrix form
    for H in Hs:
        vi_11 = form_vi(H, 1, 1)
        vi_12 = form_vi(H, 1, 2)
        vi_22 = form_vi(H, 2, 2)
        v = np.vstack((vi_12, vi_11 - vi_22))  # 2 x 6
        V.append(v)
    # V = np.array(V) creates the wrong array shape
    V = np.vstack(V)  # 2n x 6
    U, S, bt = np.linalg.svd(V.T @ V)
    b = bt[-1].reshape(6, 1)
    return b


def b_from_B(B: np.ndarray):
    """
    Returns the 6x1 vector b from the 3x3 matrix B.

    b = [B11 B12 B22 B13 B23 B33].T
    """
    if B.shape != (3, 3):
        raise ValueError("B must be a 3x3 matrix")

    b = np.array((B[0, 0], B[0, 1], B[1, 1], B[0, 2], B[1, 2], B[2, 2]))
    b = b.reshape(6, 1)
    return b


# Ex 4.7
def estimate_intrinsics(Hs:list[np.ndarray]):
    """
    Estimate intrinsic matrix using Zhang's method for camera calibration.

    Args:
        Hs : list of 3x3 homographies for each view

    Returns:
        K : 3x3 intrinsic matrix
    """
    b = estimate_b(Hs)
    B11, B12, B22, B13, B23, B33 = b
    # Appendix B of Zhang's paper
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0 * (B12 * B13 - B11 * B23)) / B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11 * B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = lambda_ * v0 / beta - B13 * alpha**2 / lambda_
    # above values are sequences [value], so using [0] below is needed
    K = np.array([[alpha[0], gamma[0], u0[0]], [0, beta[0], v0[0]], [0, 0, 1]])
    return K


# Ex 4.8
def estimate_extrinsics(K, Hs):
    """
    Estimate extrinsic parameters using Zhang's method for camera calibration.

    Args:
        K : 3x3 intrinsic matrix
        Hs : list of 3x3 homographies for each view

    Returns:
        Rs : list of 3x3 rotation matrices
        ts : list of 3x1 translation vectors
    """
    Kinv = np.linalg.inv(K)
    Rs = []
    ts = []
    for H in Hs:  # H = [h1|h2|h3]
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = H[:, 2]
        lambda_ = np.linalg.norm(Kinv @ h1, 2)
        r1 = 1 / lambda_ * Kinv @ h1  # (3,)
        r2 = 1 / lambda_ * Kinv @ h2
        r3 = np.cross(r1, r2)
        t = np.array(1 / lambda_ * Kinv @ h3).reshape(3, 1)  # 3 x 1
        R = np.vstack((r1, r2, r3)).T  # 3 x 3 [r1|r2|r3]
        Rs.append(R)
        ts.append(t)
    Rs = np.array(Rs)
    ts = np.array(ts)
    return Rs, ts


# Ex 4.8
def calibrate_camera(qs, Q):
    """
    Calibrate camera using Zhang's method for camera calibration.

    Args:
        qs : list of 2xn arrays corresponding to each view
        Q : 3 x (nxm) array of untransformed 3D points

    Returns:
        K : 3x3 intrinsic matrix
        Rs : list of 3x3 rotation matrices
        ts : list of 3x1 translation vectors
    """
    Hs = estimate_homographies(Q, qs)
    K = estimate_intrinsics(Hs)
    Rs, ts = estimate_extrinsics(K, Hs)
    return K, Rs, ts


# Ex 5.3
def triangulate_nonlin(q_list: list[np.ndarray], P_list: list[np.ndarray]):
    """
    Nonlinear triangulation of a single 3D point seen by n cameras.

    Args:
        q_list : nx2x1 list of 2x1 pixel points
        P_list : nx3x4, list of 3x4 camera projection matrices

    Returns:
        Q : 3x1 vector, 3D point
    """

    def compute_residuals(Q: np.ndarray):
        """
        Helper function.

        Args:
            Q : 3-vector, 3D point (parameters to optimize)

        Returns:
            residuals : 2n-vector, residuals
                        (numbers to minimize sum of squares)
        """
        Q = Q.reshape(-1, 1)
        Qh = PiInv(Q)
        # least_squares() expects 1D array
        residuals = np.zeros(2 * len(q_list))  # 2n-vector
        # Compute difference in projection
        for i, q in enumerate(q_list):
            projected_2D = Pi(P_list[i] @ Qh)
            res = q - projected_2D  # difference in projection
            residuals[2 * i : 2 * (i + 1)] = res.reshape(-1)
        return residuals

    # Initial guess with linear approach
    x0 = triangulate(q_list, P_list)  # 3x1
    x0 = x0.reshape(-1)  # least_squares() expects 1D array

    # Least squares optimization
    from scipy.optimize import least_squares

    result = least_squares(compute_residuals, x0)
    return result.x


# Ex 6.1
def gaussian_1D_kernel(sigma: int):
    """
    Returns the 1D Gaussian kernel.

    Args:
        sigma : width of Gaussian kernel

    Returns:
        g : 1D Gaussian kernel
        gd : 1D Gaussian kernel derivative
    """
    if sigma == 0:
        return [1], [0]
    rule = 5
    x = np.arange(-rule * sigma, rule * sigma + 1)
    g = np.exp(-(x**2) / (2 * sigma**2))
    g = g / np.sum(g)  # normalize
    gd = -x / (sigma**2) * g
    return g, gd


# Ex 6.2
def gaussian_smoothing(im, sigma: int):
    """
    Smooths the input image with a 1D Gaussian kernel.

    Args:
        im : input image
        sigma : width of Gaussian kernel

    Returns:
        I : smoothed image
        Ix : image derivative in x-direction
        Iy : image derivative in y-direction
    """
    g, gd = gaussian_1D_kernel(sigma)
    I = cv2.sepFilter2D(im, -1, g, g)
    Ix = cv2.sepFilter2D(im, -1, gd, g)
    Iy = cv2.sepFilter2D(im, -1, g, gd)
    return I, Ix, Iy


# Ex 6.3
def structure_tensor(im, sigma, epsilon):
    """
    Computes the structure tensor C(x,y) of the input image.

    Args:
        im : input image
        sigma : Gaussian width to compute derivatives
        epsilon : Gaussian width to compute the structure tensor

    Returns:
        J : structure tensor
    """
    I, Ix, Iy = gaussian_smoothing(im, sigma)

    g_eps, g_eps_d = gaussian_1D_kernel(epsilon)
    C = np.zeros((2, 2))
    C00 = cv2.sepFilter2D(Ix**2, -1, g_eps, g_eps)
    C01 = cv2.sepFilter2D(Ix * Iy, -1, g_eps, g_eps)
    C10 = C01
    C11 = cv2.sepFilter2D(Iy**2, -1, g_eps, g_eps)
    C = np.array([[C00, C01], [C10, C11]])

    return C


# Ex 6.4
def harris_measure(im, sigma, epsilon, k):
    """
    Computes the Harris measure R(x,y) of the input image.

    Args:
        im : (h,w) input image
        sigma : Gaussian width to compute derivatives
        epsilon : Gaussian width to compute the structure tensor
        k : sensitivity factor

    Returns:
        r : (h,w), Harris measure
    """
    C = structure_tensor(im, sigma, epsilon)
    a = C[0, 0]
    b = C[1, 1]
    c = C[0, 1]
    r = a * b - c**2 - k * (a + b) ** 2
    return r


# Ex 6.5
def corner_detector(im, sigma, epsilon, tau, k):
    """
    Detects corners in the input image using the Harris measure
    with non-max suprrssion and thresholding.

    Args:
        im : input image
        sigma : Gaussian width to compute derivatives
        epsilon : Gaussian width to compute the structure tensor
        tau : threshold for Harris measure
        k : sensitivity factor

    Returns:
        c : list of corner coordinates
    """
    r = harris_measure(im, sigma, epsilon, k)
    print(f"r: [{r.max():.2f}, {r.min():.2f}], tau = {tau/r.max():.2f}*r.max")

    # Perform 4-neigbourhood non-max suppression
    c = []
    for i in range(1, r.shape[0] - 1):
        for j in range(1, r.shape[1] - 1):
            if (
                r[i, j] > r[i + 1, j]
                and r[i, j] >= r[i - 1, j]
                and r[i, j] > r[i, j + 1]
                and r[i, j] >= r[i, j - 1]
                and r[i, j] > tau
            ):  # Threshold
                c.append([i, j])
    return c


# Ex 7.6
def fit_line(p1: np.ndarray, p2: np.ndarray):
    """
    Fits a line given 2 points.

    Args:
        p1, p2 (np.array) : 2x1 inhomogenous coordinates

    Returns:
        l : 3x1, line in homogenous coordinates
    """
    if p1.shape == (2,):
        p1 = p1.reshape(2, 1)
        p2 = p2.reshape(2, 1)
    if p1.shape != (2, 1) or p2.shape != (2, 1):
        raise ValueError("Points must be 2x1 np.array")

    p1h = PiInv(p1)
    p2h = PiInv(p2)
    # cross() requires input as vectors
    l = np.cross(p1h.squeeze(), p2h.squeeze())
    return l


# Ex 7.7
def find_inliners_outliers(l: np.ndarray, points: np.ndarray, tau: float):
    """
    Args:
        l : equation of line in homogenous coordinates
        points : 2xn, set of 2D points
        tau : threshold for inliners

    Returns:
        inliners (np.array) : 2xa, set of inliner points
        outliers (np.array) : 2xb, set of outlier points
    """
    inliners = []
    outliers = []
    for p in points.T:
        p = p.reshape(2, 1)
        ph = PiInv(p)
        d = abs(l.T @ ph) / (abs(ph[2]) * np.sqrt(l[0] ** 2 + l[1] ** 2))
        if d <= tau:  # inliner
            inliners.append(p)
        else:  # outlier
            outliers.append(p)
    inliners = np.array(inliners).squeeze().T
    outliers = np.array(outliers).squeeze().T
    return inliners, outliers


# Ex 7.8
def consensus(l, points, tau: float) -> int:
    """
    Returns the number of inliners.

    Args:
        l : equation of line in homogenous coordinates
        points : 2xn, set of points
        tau : threshold for inliners

    Returns:
        consensus : number of inliners
    """
    consensus = 0
    inliners, _ = find_inliners_outliers(l, points, tau)
    consensus = inliners.shape[1]
    return consensus


# Ex 7.9
def sample_points(points: np.ndarray):
    """
    Randomly sample two of n 2D points without replacement.

    Args:
        points : nx2, set of points
        n : number of points to sample

    Returns:
        sample : 2xn, set of sampled points
    """
    n_points = 2
    sample = np.random.permutation(points.T).T[:, :n_points]
    return sample


# Ex 7.10
def ransac(points, iters, tau, t=5):
    """
    Random sample consensus (RANSAC) algorithm to fit
    a line to a set of points

    Args:
        points : nx2, set of points
        iters : number of iterations
        tau : threshold for inliners
        t : number of inliners to accept the model

    Returns:
        best_l : equation of line in homogenous coordinates
        best_inliners : ax2, set of inliner points
    """
    best_l = None
    best_inliners = None
    best_consensus = 0
    for _ in range(iters):
        sample = sample_points(points)
        l = fit_line(sample[:, 0], sample[:, 1])
        consensus_ = consensus(l, points, tau)
        if consensus_ > t and consensus_ > best_consensus:
            best_l = l
            best_inliners, _ = find_inliners_outliers(l, points, tau)
            best_consensus = consensus_
    return best_l, best_inliners


def visualize_ransac(points, best_inliners, best_l) -> None:
    """
    Visualizes the RANSAC algorithm for 2D lines.

    Args:
        points : nx2, set of points
        best_inliners : ax2, set of inliner points
        best_l : equation of line in homogenous coordinates
    """

    plt.scatter(points[0], points[1], c="b", label="all points")
    plt.scatter(best_inliners[0], best_inliners[1], c="r", label="inliners")

    # Draw line created by samples
    slope = -best_l[0] / best_l[1]
    intercept = -best_l[2] / best_l[1]
    x = np.linspace(min(points[0]), max(points[0]), 100)
    y = slope * x + intercept
    plt.plot(x, y, c="g", label="best line")

    plt.xlim(min(points[0]) - 1, max(points[0]) + 1)
    plt.ylim(min(points[1]) - 1, max(points[1]) + 1)

    plt.legend()
    plt.title("RANSAC")
    plt.show()


# Ex 8.1
def scale_spaced(im: np.ndarray, sigma: int, n: int, visualize=False):
    """
    Naive implementation of the scale space pyramid with no downsampling.

    Args:
        im (np.ndarray) : input image
        sigma (int) : standard deviation of the Gaussian kernel
        n (int) : number of scales

    Returns:
        im_scales : list containing the scale space pyramid of the input image
        scales : list containing the scales used in the pyramid
    """
    scales = [sigma * 2**i for i in range(n)]  # ratio = 2
    im_scales = []
    im_scale = im
    for scale in scales:
        # Apply Gaussian filter on the previously scaled image
        g, _ = gaussian_1D_kernel(scale)
        im_scale = cv2.sepFilter2D(
            src=im_scale,
            ddepth=-1,
            kernelX=g,
            kernelY=g,
        )
        im_scales.append(im_scale)

    if visualize:
        for i, im_scale in enumerate(im_scales):
            plt.imshow(im_scale, cmap="gray")
            plt.axis("off")
            plt.show()
        plt.show()
    return im_scales, scales


# Ex 8.2
def difference_of_gaussians(im: np.ndarray, sigma: int, n: int):
    """
    Implementation of the difference of Gaussians.

    Args:
        im (np.ndarray) : input image
        sigma (int) : standard deviation of the Gaussian kernel
        n (int) : number of scales

    Returns:
        DoG : list of scale space DoGs of im
        scales : list containing the scales used in the pyramid
    """
    im_scales, scales = scale_spaced(im, sigma, n)
    DoG = []
    for i in range(1, n):
        diff = im_scales[i] - im_scales[i - 1]
        DoG.append(diff)
    return DoG, scales


# Ex 8.3
def detect_blobs(im: np.ndarray, sigma: int, n: int, tau: float):
    """
    Implementation of the blob detector.

    Args:
        im (np.ndarray) : input image
        sigma (int) : standard deviation of the Gaussian kernel
        n (int) : number of scales
        tau (float) : threshold for blob detection

    Returns:
        blobs : list of detected blobs in the format (x, y, scale)
    """
    DoG, scales = difference_of_gaussians(im, sigma, n)
    DoG = np.array(DoG)

    # Obtain max value in a 3x3 neighborhood of each pixel in DoG
    MaxDoG = [cv2.dilate(abs(dog), np.ones((3, 3))) for dog in DoG]

    # Thresholding & non-max suppression
    blobs = []
    prev_blobs = 0
    for i in range(len(DoG)):  # for each DoG
        if i == 0:
            prev_MaxDoG = np.zeros(DoG[0].shape)
            next_MaxDoG = MaxDoG[i + 1]
        elif i == len(DoG) - 1:
            prev_MaxDoG = MaxDoG[i - 1]
            next_MaxDoG = np.zeros(DoG[0].shape)
        else:
            prev_MaxDoG = MaxDoG[i - 1]
            next_MaxDoG = MaxDoG[i + 1]

        for j in range(im.shape[0]):  # for each row
            for k in range(im.shape[1]):  # for each column
                # take abs() to find max and min
                if (
                    abs(DoG[i][j, k]) > tau  # thresholding
                    and abs(DoG[i][j, k])
                    == MaxDoG[i][j, k]  # max in current DoG
                    and abs(DoG[i][j, k])
                    > prev_MaxDoG[j, k]  # max in previous DoG
                    and abs(DoG[i][j, k]) > next_MaxDoG[j, k]  # max in next DoG
                ):
                    blobs.append((j, k, scales[i]))
        # Calculate how many new blobs detected in this DoG
        print(f"No. of blobs detected in DoG {i}: {len(blobs)-prev_blobs}")
        prev_blobs = len(blobs)
    return blobs


def visualize_blobs(blobs, im):
    """
    Args:
        blobs : list of detected blobs in the format (x, y, scale)
        im : BGR input image
    """
    gray_img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # To draw colored shapes on a gray img
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for x, y, scale in blobs:
        cv2.circle(
            bgr_img,
            (y, x),
            radius=int(scale),
            color=(255, 0, 0),
            thickness=2,
        )
    plt.axis("off")
    plt.imshow(bgr_img)
    plt.show()


# Ex 8.4
def transform_im(im: np.ndarray, theta: float, s: float):
    """
    Rotate an image by theta degrees and scale by s.

    Args:
        im : input image
        theta : angle of rotation
        s : scaling factor

    Returns:
        r_im : rotated and scaled image
    """
    rows, cols = im.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, s)
    r_im = cv2.warpAffine(im, M, (cols, rows))
    return r_im


# Ex 9.1
def Fest_8point(q1: np.ndarray, q2: np.ndarray):
    """
    Estimate the fundamental matrix using the 8-point linear algorithm.

    Args:
        q1 (np.ndarray): 2D points in image 1, shape (2, 8).
        q2 (np.ndarray): 2D points in image 2, shape (2, 8).

    Returns:
        F (np.ndarray): The estimated fundamental matrix, shape (3, 3).
    """
    # Construct B vector
    B = np.zeros((q1.shape[1], 9))
    for i in range(q1.shape[1]):
        x1, y1 = q1[:, i]
        x2, y2 = q2[:, i]
        Bi = np.array([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])
        B[i] = Bi

    # Solve for F
    U, S, V = np.linalg.svd(B)
    F = V[-1].reshape(3, 3)

    return F


# Ex 9.2
def find_features(im1: np.ndarray, im2: np.ndarray, plot=False):
    """
    Find matching features between two images.

    Args:
        im1 (np.ndarray): The first image.
        im2 (np.ndarray): The second image.

    Returns:
        matches (list): Matching features (cv2.DMatch objects).
        kp1 (list): Keypoints in the first image.
        kp2 (list): Keypoints in the second image.
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # Create BFMatcher object
    bf = cv2.BFMatcher(crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # ascending

    # Draw first 10 matches.
    if plot:
        img3 = cv2.drawMatches(
            im1,
            kp1,
            im2,
            kp2,
            matches[:10],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        mpl.rcParams["figure.figsize"] = [15, 10]
        plt.imshow(img3)
        plt.axis("off")
        plt.title("Closest 10 matches")
        plt.show()
        mpl.rcParams["figure.figsize"] = [8, 6]

    return matches, kp1, kp2


# Ex 9.3
def sampsons_distance(F: np.ndarray, p1: np.ndarray, p2: np.ndarray):
    """
    Compute the Sampson distance for the given fundamental matrix and point correspondences.

    Args:
        F (np.array): The fundamental matrix, shape (3, 3).
        p1 (np.array): 2D points in image 1, shape (2, N).
        p2 (np.array): 2D points in image 2, shape (2, N).

    Returns:
        dist (np.array): The Sampson distance for each point, shape (N,).
    """
    # if p1.shape[0] == 3 and p2.shape[0] == 3:
    #     # Normalize points
    #     p1 = p1[:2] / p1[2]
    #     p2 = p2[:2] / p2[2]
    # if p1.shape[0] != 2 or p2.shape[0] != 2:
    #     raise ValueError("p1 and p2 must have shape (2, N).")

    # Make homogeneous to multiply with F
    p1 = PiInv(p1)
    p2 = PiInv(p2)
    distances = np.zeros((p1.shape[1], 1))

    # For each pair of points
    for i in range(p1.shape[1]):
        p1i, p2i = p1[:, i], p2[:, i]
        num = (p2i.T @ F @ p1i) ** 2
        denom = (
            (p2i.T @ F[0]) ** 2
            + (p2i.T @ F[1]) ** 2
            + (F @ p1i)[0] ** 2
            + (F @ p1i)[1] ** 2
        )
        dist = num / denom
        distances[i, :] = dist
    return distances


def ransac_fundamental_matrix(im1, im2, threshold: float, iters=1000):
    """
    Estimate the fundamental matrix using RANSAC.

    Args:
        q1 (np.ndarray): 2D points in image 1, shape (2, N).
        q2 (np.ndarray): 2D points in image 2, shape (2, N).
        threshold (float): The threshold used for the RANSAC algorithm.
        iters (int): The number of iterations to run the RANSAC algorithm.

    Returns:
        F (np.ndarray): The estimated fundamental matrix, shape (3, 3).
        inliers (np.ndarray): The inliers used to estimate F, shape (N,).
    """
    matches, kp1, kp2 = find_features(im1, im2)
    im1_matches = np.array(
        [(kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]) for m in matches],
    )  # (N, 2)
    im2_matches = np.array(
        [(kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]) for m in matches],
    )  # (N, 2)

    best_consensus = 0
    best_inliers = None
    best_F = None

    for i in range(iters):
        # 1. sample 8 random matches
        # 2. use Fest_8point to estimate F matrix from the 8 matches
        # 3. compute sampson's distance
        # 4. find inliers if sampson's distance < threshold
        # 5. update best F and inliers
        # 6. iterate

        # Sample 8 random matches
        match_samples = np.random.choice(matches, 8, replace=False)

        # Extract x-y coordinates of samples
        im1_samples = np.array(
            [
                (kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1])
                for m in match_samples
            ],
        )  # (8, 2)
        im2_samples = np.array(
            [
                (kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1])
                for m in match_samples
            ],
        )  # (8, 2)

        F = Fest_8point(im1_samples.T, im2_samples.T)  # requires (2, 8) input

        # Calculate the dist for all matches
        dist_samp = sampsons_distance(F, im1_matches.T, im2_matches.T)
        inliers = [
            idx for idx, dist in enumerate(dist_samp) if dist < threshold
        ]
        inliers = np.array([im1_matches[inliers], im2_matches[inliers]])
        # inliers shape : (2, M, 2)

        consensus = inliers.shape[1]
        # Update best F and inliers
        if consensus > best_consensus:
            best_inliers = inliers
            best_consensus = consensus
            best_F = F

    if best_F is None:
        raise ValueError("RANSAC did not find any inliers.")

    # Refit the model using all inliers
    print(f"Best consesus of {best_consensus} out of {len(matches)} matches.")
    best_F = Fest_8point(best_inliers[0].T, best_inliers[1].T)

    return best_F, best_inliers


# Ex 10.2
def Hest_dist(H: np.ndarray, q1: np.ndarray, q2: np.ndarray):
    """
    Approximate distance of observed points to a homography matrix.

    Args:
        H: Homography matrix
        q1 (np.array): observed point in image 1, shape (N,2)
        q2 (np.array): observed point in image 2, shape (N,2)

    Returns:
        dist_approx (np.array): Approximate distance
    """
    p1 = PiInv(q1)
    p2 = PiInv(q2)
    distances = np.zeros((p1.shape[1], 1))

    # For each pair of points
    for i in range(p1.shape[1]):
        p1i, p2i = p1[:, i], p2[:, i]
        q1i, q2i = q1[:, i], q2[:, i]
        temp1 = (q1i - Pi(H @ p2i)) ** 2
        temp1 = np.sqrt(np.sum(temp1))
        temp2 = (q2i - Pi(np.linalg.inv(H) @ p1i)) ** 2
        temp2 = np.sqrt(np.sum(temp2))
        distances[i] = temp1 + temp2
    return distances


# Ex 10.2
def ransac_homography(
    im1: np.ndarray, im2: np.ndarray, threshold: float, iters=200, plot=False
):
    """
    Estimate homography matrix using RANSAC.

    Args:
        im1 (np.array): Image 1
        im2 (np.array): Image 2
        threshold (float|int): Distance threshold for inliers
        iters (int): Number of iterations
        plot (bool): Visualize inlier matches

    Returns:
        best_H (np.array): Best homography matrix, shape (3,3)
        best_inliers (np.array): Inliers of best homography (M,2,2)
    """

    # Find features
    matches, kp1, kp2 = find_features(im1, im2)

    # Extract matched keypoints
    im1_pts = np.float32([kp1[m.queryIdx].pt for m in matches])  # n x 2
    im2_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    # RANSAC
    best_consensus = 0
    best_inliers = None
    best_H = None

    for i in range(iters):
        # Randomly select 4 points
        idx = np.random.choice(len(im1_pts), 4, replace=False)
        q1 = im1_pts[idx]
        q2 = im2_pts[idx]

        # Estimate homography
        H = hest(q1.T, q2.T)

        # Calculate distance
        distances = Hest_dist(H, im1_pts.T, im2_pts.T)
        inliers_idx = [
            idx for idx, dist in enumerate(distances) if dist < threshold
        ]
        inliers = np.array(
            [[im1_pts[idx], im2_pts[idx]] for idx in inliers_idx],
        )  # (M,2,2)
        consensus = inliers.shape[0]

        if consensus > best_consensus:
            best_inliers_idx = inliers_idx
            best_inliers = inliers
            best_consensus = consensus
            best_H = H

    if best_H is None:
        raise ValueError("RANSAC did not find any inliers.")

    # Refit homography to inliers
    print(f"Best consensus of {best_consensus} out of {len(im1_pts)} points.")
    best_H = hest(best_inliers[:, 0].T, best_inliers[:, 1].T)

    if plot:
        # Parallel lines should be expected
        plt.imshow(
            cv2.drawMatches(
                im1,
                kp1,
                im2,
                kp2,
                np.array(matches)[best_inliers_idx],
                None,
            ),
        )
        plt.axis("off")
        plt.show()

    return best_H, best_inliers


# Ex 13.3
def unwrap(imgs, n1):
    """
    Unwrap the measured phases. Adapter from Collister.

    Args:
        imgs (List[np.ndarray]): List of images from the cameras.
        n1 (int): Period of the primary pattern.

    Returns:
        theta_est (np.ndarray) : The phase of the primary pattern.
    """
    # Primary pattern
    primary_images = imgs[2:18]  # 16 images
    fft_primary = np.fft.rfft(primary_images, axis=0)
    fourier_primary = fft_primary[1]
    theta_primary = np.angle(fourier_primary)

    # Secondary pattern
    secondary_images = imgs[18:26]
    fft_secondary = np.fft.rfft(secondary_images, axis=0)
    fourier_secondary = fft_secondary[1]
    theta_secondary = np.angle(fourier_secondary)

    # Compute phase cue using heterodyne principle
    theta_c = np.mod(theta_secondary - theta_primary, 2 * np.pi)

    # Order of primary phase
    o_primary = np.rint((n1 * theta_c - theta_primary) / (2 * np.pi))

    # Estimate the phase
    theta_est = np.mod((2 * np.pi * o_primary + theta_primary) / n1, 2 * np.pi)
    return theta_est
