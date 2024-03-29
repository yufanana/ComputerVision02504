#!/usr/bin/env python
import numpy as np
import cv2
from util_functions import (
    gaussian_1D_kernel,
)


# Ex 8.1
def scale_spaced(im, sigma: float, n: int):
    """
    Naive implementation of the scale space pyramid with no downsampling.

    Args:
        im : input image
        sigma : standard deviation of the Gaussian kernel
        n : number of scales

    Returns:
        im_scales : list containing the scale space pyramid of the input image
        scales : list of scales used in the pyramid
    """
    print("Creating scale space pyramid...")
    exp = np.linspace(0, n - 1, n)
    scales = sigma * np.power(2, exp)
    im_scales = []
    for sd in scales:
        g, _ = gaussian_1D_kernel(sd)
        I = cv2.sepFilter2D(im, -1, g, g)
        im_scales.append(I)
    print("Done with scale space pyramid.")
    return im_scales, scales


# Ex 8.2
def difference_of_gaussians(im, sigma: float, n: int):
    """
    Implementation of the difference of Gaussians.

    Args:
        im : input image
        sigma : standard deviation of the Gaussian kernel
        n : number of scales

    Returns:
        DoG : list of scale space DoGs of im
        scales : list of scales used in the pyramid
    """
    print("Creating DoG...")
    im_scales, scales = scale_spaced(im, sigma, n)
    DoG = []
    for i in range(1, n):
        diff = im_scales[i] - im_scales[i - 1]
        DoG.append(diff)
    print("Done with DoG.")
    return DoG, scales


# Ex 8.3
def detect_blobs(im, sigma: float, n: int, tau: float):
    """
    Detect blobs using thresholding and non-max suppression.

    Args:
        im : input image
        sigma : standard deviation of the Gaussian kernel
        n : number of scales
        tau : threshold for blob detection

    Returns:
        blobs : list of detected blobs
    """
    print("Detecting blobs...")
    DoG, scales = difference_of_gaussians(im, sigma, n)

    # non-max suppression using a 3x3 max filter
    MaxDoG = [cv2.dilate(dog, np.ones((3, 3))) for dog in DoG]

    blobs = []
    for i in range(1, len(DoG) - 1):  # for each DoG, skip first & last
        for j in range(1, im.shape[0] - 1):  # for each row
            for k in range(1, im.shape[1] - 1):  # for each column
                if (
                    MaxDoG[i][j, k] > tau  # thresholding
                    and MaxDoG[i][j, k] > MaxDoG[i - 1][j, k]  # previous DoG
                    and MaxDoG[i][j, k] > MaxDoG[i + 1][j, k]
                ):  # next DoG
                    blobs.append((j, k, scales[i]))
    print("Done with blob detection.")
    return blobs


def visualize_blobs(blobs, im):
    print("Visualizing blobs...")
    for x, y, scale in blobs:
        cv2.circle(im, (int(y), int(x)), int(scale), (10, 255, 255), 1)

    # Show a smaller image
    im_small = cv2.resize(im, (im.shape[1] // 4, im.shape[0] // 4))
    cv2.imshow("Blob Detection", im_small)
    print("Blob visualized. Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    im = "media/sunflowers.jpg"
    im = cv2.imread(im).astype(float).mean(2) / 255
    sigma = 2
    n = 7
    tau = 0.1
    blobs = detect_blobs(im, sigma, n, tau)
    visualize_blobs(blobs, im)
