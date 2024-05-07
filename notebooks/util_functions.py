"""
Utilty functions commonly used in exercises and quizzes.
"""
import cv2
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
import matplotlib as mpl


# Ex 1.7
def point_line_distance(line, p):
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
def Pi(ph):
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
def PiInv(p):
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


def projection_matrix(K, R, t):
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
def project_points(K, R, t, Q, distCoeffs=[]):
    """
    Project 3D points in Q onto a 2D plane of a camera with distortion.

    Args:
        K: intrinsic camera matrix, shape (3, 3)
        R: rotation matrix, shape (3, 3)
        t: translation matrix. shape (3, 1)
        Q: 3D points matrix, shape (3, n)
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


def camera_intrinsic(f, c, alpha=1, beta=0):
    """
    Create a camera intrinsic matrix

    Args:
        f (float): focal length
        c (tuple): 2D principal point (x,y)
        alpha (float): skew
        beta (float): aspect ratio

    Returns:
        K (np.array0): intrinsic camera matrix, shape (3, 3)
    """
    K = np.array([[f, beta * f, c[0]], [0, alpha * f, c[1]], [0, 0, 1]])
    return K


# Ex 2.2
def distort(q, distCoeffs):
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


# Ex 2.7
def normalize2d(q):
    """
    Normalize 2D points.

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
def hest(q1, q2, normalize=False):
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
def R(theta_x, theta_y, theta_z):
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
def CrossOp(r):
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
def essential_matrix(R, t):
    """
    Returns the essential matrix.

    Args:
        R : 3x3 matrix, rotation matrix
        t : 3x1 matrix, translation matrix

    Returns:
        E : 3x3 matrix, essential matrix
    """
    return CrossOp(t) @ R


def fundamental_matrix(K1, R1, t1, K2, R2, t2):
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
def DrawLine(l, shape):
    """
    Draws a line on an the 2nd plot.

    Args:
        l : 3x1 vector, epipolar line in homogenous coordinates
        shape : tuple, shape of the image
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
def triangulate(q_list, P_list):
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
def compute_rmse(q_true, q_est):
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
def checkerboard_points(n, m):
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
        qs : list of 2xn arrays corresponding to each view

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
def form_vi(H, a, b):
    """
    Form 1x6 vector vi using H and indices alpha, beta.

    Args:
        H : 3x3 homography
        a, b : indices alpha, beta

    Returns:
        vi : 1x6 vector
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
def estimate_b(Hs):
    """
    Estimate b matrix used Zhang's method for camera calibration.

    Args:
        Hs : list of 3x3 homographies for each view

    Returns:
        b : 6x1 vector
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


# Ex 4.7
def estimate_intrinsics(Hs):
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


# Ex 6.1
def gaussian_1D_kernel(sigma):
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
def gaussian_smoothing(im, sigma):
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


# Ex 9.2
def find_features(im1, im2, plot=False):
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
