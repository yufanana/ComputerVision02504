# ComputerVision02504

This repository contains my personal material used as part of the course 02504 Computer Vision, at DTU in Spring 2024.

Please submit a pull request if you find any mistakes or you would like to contribute. :)

## Topics Covered

- Week 1: Homogenous coordinates, pinhole model, projection
- Week 2: Camera model, lens distortion, homography, point normalization
- Week 3: Multiview geometry, epipolar, triangulation
- Week 4: Linear camera calibration
- Week 5: Nonlinear optimization, camera calibration
  - Levenberg-Marquardt: least squares problem with 2nd order approximation using only 1st order derivatives
  - Gradients: analytical or finite differences (Taylor series)
  - Rotations in Optimization (euler angles, axis angles, quaternions)
- Week 6: Simple features, Harris corner, Gaussian filtering, Gaussian derivative,
- Week 7: Robust model fitting, RANSAC, Hough transform
- Week 8: SIFT features, difference of Gaussians, scale space pyramid
- Week 9: Esimate Fundamental matrix using RANSAC

<!--toc-->

## Installation

Clone the repository.

```bash
git clone https://github.com/yufanana/ComputerVision02504.git
cd ComputerVision02504
```

Create a Python virtual environment with `virtualenv`.

```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# On Windows
.venv/Scripts activate
pip install -r requirements.txt
```

Or, create an environment with `Conda`.

```bash
conda create --name cv
pip install -r requirements.txt
```

When running the Jupyter notebooks, select the environment previously created.

## Development

Set up pre-commit for code checking. Hooks can be found in `.pre-commit-config.yaml`

```bash
conda activate cv
pre-commit install
pre-commit run --all-files
pre-commit run --file my_file.ipynb
```

Now, `pre-commit` will run every time before a commit. Remember to do `git add .` to add the changes made by the pre-commit (if any). Hooks can be temporarily disabled using:

```bash
git commit -m "<message>" --no-verify
```

## Week 8: BLOBs and SIFT features

See examples in [ex8.ipynb](notebooks/ex8.ipynb)

SIFT: features localized at interest points, adapted to scale, inavariance to appearance changes

- scale-space blob detection using difference of Gaussians (DoG)
- interest point localization
- orientation assignment
- interest point descriptor

BLOB: binary large object

Hessian matrix

$$
H =
\begin{bmatrix}
I_{xx}(x,y) & I_{xy}(x,y) \\[0.3em]
I_{xy}(x,y) & I_{yy}(x,y) \\[0.3em]
\end{bmatrix}
$$

- contains 2nd order derivatives of images
- measures curvature
- eigenvalues and eigenvectors are used to measure the direction of most change
- the Laplacian is used to estimate the eigenvalues (?)
- the Laplacian is approximated with DoG

Difference of Gaussians (DoG)

- iteratively blurring already blurred images (efficient)
- scale invariance: allows features to be detected at different scales
- kernel size increase with each iteration
- $DoG = L(x,y,k\sigma) - L(x,y,\sigma)$
- the same threshold can be applied for all scale spaces
- find local extrama of DoGs in scale space

Orientation assignment

- compute orientation of gradient around BLOB
- compute circular histogram of gradient orientations
- use histogram peak to assign orientation of point

Matching descriptors

- Use Euclidean distance between normalized vectors
- Cross checking: keep matches that are closest to each other
- Ratio test: compute ratio betwen closest and 2nd closest match, keep if it is below threshold e.g. 0.7

Variations

- RootSIFT: Hellinger kernel
- SURF, ORB, BRIEF, BRISK

## Week 9: Geometry Constrainde Feature Matching

Fundamentral and Essential Matrix

$$
\begin{align*}
E &= [t]_xR \\
F &= K_2^{-T}EK_1^{-1} \\
0 &= q_2^TFq_1
\end{align*}
$$

Estimate F by solving $0 = B^{(i)} flatten(F) $ using SVD, where:

$$
\begin{align*}
B^{(i)} &= [x_{1i} x_{2i} \ \ \
y_{1i}x_{2i} \ \ \
x_{2i} \ \ \
x_{1i}y_{2i} \ \ \
y_{1i}y_{2i} \ \ \
y_{2i}  \ \ \
x_{1i} \ \ \
y_{1i} \ \ \
1] \\

flatten(F) &= [F_{11} \ \ F_{12} \ \ F_{13}
            \ \ F_{21} \ \ F_{22} \ \ F_{23}
            \ \ F_{31} \ \ F_{32} \ \ F_{33}]^T

\end{align*}
$$

F has 9 DoF, scale invariant <br>
$\Rightarrow$ 8 data points is sufficient <br>
$\Rightarrow$ 8 pairs of corresponding points

It is also possible to estimate using 7 points.

$ q_2i^TFq_1i $ is the distance from the epipolar lines.

Use **Sampson's distance** to measure distance from model.

$$
d_{Samp} (F, q_{1i}, q_{2i}) =
\dfrac{(q_{2i}^T F q_{1i})^2}
{(q_{2i}^T F)_1^2 +
(q_{2i}^T F)_2^2 +
(F q_{1i})_1^2 +
(F q_{1i})_2^2}
$$

Threshold for RANSAC

- Assume each sample has error with m-dimensional normal distribution
- Choose a confidence level e.g. 95%
- Look up CDF for $\chi_m^2$ distribution

RANSAC Workflow

1. Find features in both images using SIFT
2. Match features using brute force matcher (e.g. 1000 matches)
3. Sample 8 of these matching features (8 points from image 1, 8 points from image 2)
4. Estimate fundamental matrix using SVD
5. Compute sampson distance to estimated F for all matches
6. Classify matches as inliers if distance < threshold
7. Repeat for fixed number of iterations
8. Refit fundamental matrix on set of best inliers

![Chi-square Distribution Table](assets/chi_square_distribution.png)