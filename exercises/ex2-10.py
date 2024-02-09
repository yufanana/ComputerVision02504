import numpy as np
import cv2
import matplotlib.pyplot as plt


def Pi(ph):
    """
    Converts coordinates from homogeneous to inhomogeneous.
    ph : 4xn np.array
    p : 3xn np.array
    """
    p = ph[:-1] / ph[-1]  # divide by and remove last coordinate
    return p


def PiInv(p):
    """
    Converts coordinates from inhomogeneous to homogeneous.
    p : 3xn np.array
    ph : 4xn np.array
    """
    ph = np.vstack((p, np.ones(p.shape[1])))
    return ph


def normalize2d(q):
    """
    Normalize 2D points
    q : 2 x n, 2D points
    qn : 2 x n, normalized 2D points
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


def hest(q1, q2, normalize=False):
    """
    Calculate the homography matrix from n sets of 2D points
    q1 : 2 x n, 2D points in the first image
    q2 : 2 x n, 2D points in the second image
    H : 3 x 3, homography matrix
    """
    if q1.shape[1] != q2.shape[1]:
        raise ValueError("Number of points in q1 and q2 must be equal")
    if q1.shape[1] < 4:
        raise ValueError("At least 4 points are required to estimate a homography")
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
    if normalize:
        H = np.linalg.inv(T1) @ H @ T2
    return H


def hest_from_image(im1, im2, n):
    """
    Estimate homography from n pairs of points in two images
    im1 : np.array, first image
    im2 : np.array, second image
    n : int, number of pairs of points
    H : 3 x 3, homography matrix
    """
    plt.imshow(im1)
    p1 = plt.ginput(n)
    plt.close()
    plt.imshow(im2)
    p2 = plt.ginput(n)
    plt.close()
    H = hest(np.array(p1).T, np.array(p2).T, True)
    return H


def htrans(H, im1, im2):
    """
    Apply a homography to a selected point.
    im1 : np.array, first image
    im2 : np.array, second image
    H : 3 x 3, homography matrix
    """
    # Select a point in the second image
    plt.imshow(im2)
    p2 = plt.ginput(1)
    p2 = np.array(p2).T
    plt.close()

    q2 = PiInv(p2)
    q1 = H @ q2
    p1 = Pi(q1)

    # Plot the points
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(im1)
    ax1.plot(p1[0], p1[1], "ro")
    ax2.imshow(im2)
    ax2.plot(p2[0], p2[1], "ro")
    plt.show()


im1 = cv2.imread("exercises/imageA.jpg")[:, :, ::-1]
im2 = cv2.imread("exercises/imageB.jpg")[:, :, ::-1]
H = hest_from_image(im1, im2, 4)

with np.printoptions(precision=3, suppress=True):
    print(f"H:\n{H}")

htrans(H, im1, im2)
