# ComputerVision02504

This repository contains my personal material used as part of the course 02504 Computer Vision, at DTU in Spring 2024.

Please submit a pull request if you find any mistakes or you would like to contribute. :)

## Topics Covered

- Week 1: Homogenous coordinates, pinhole model, projection
- Week 2: Camera model, lens distortion, homography, point normalization
- Week 3: Multiview geometry, epipolar, triangulation
- Week 4: Linear camera calibration
- Week 5: Nonlinear optimization, camera calibration, triangulation

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
```

Now, `pre-commit` will run every time before a commit. Remember to do `git add .` to add the changes made by the pre-commit (if any). You can temporarily disable hooks using `git commit -m "<message>" --no-verify`
