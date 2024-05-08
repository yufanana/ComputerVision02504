# ComputerVision02504

This repository contains my personal notes and notebooks used as part of the course 02504 Computer Vision, at DTU in Spring 2024.

Notes with key concepts and equations can be found in [cv_notes.md](cv_notes.md) and [cv_notes.html](cv_notes.html)

## Installation

Clone the repository.

```bash
git clone https://github.com/yufanana/ComputerVision02504.git
cd ComputerVision02504
```

Create an environment with `Conda`.

```bash
conda create --name cv python=3.11
pip install -r requirements.txt
```

Or, create a Python virtual environment with `virtualenv`.

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

When running the Jupyter notebooks, select kernel of the environment previously created.

## Development

The markdown file can be exported into HTML using the [Markdown+Math](https://marketplace.visualstudio.com/items?itemName=goessner.mdmath) extension by goessner on VSCode.

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
