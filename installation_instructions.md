# Deep Learning class - installation instructions

## Option 1: uv (recommended)

[uv](https://docs.astral.sh/uv/) is a fast, modern Python package manager. Follow the
[installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
for your platform.

Then clone the repository and set up the environment:

```bash
git clone https://github.com/rth/dl-lectures-labs
cd dl-lectures-labs
uv venv --python 3.13

# macOS/Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

uv pip install -r requirements.txt
```

## Option 2: conda/miniforge

Download the miniforge3 distribution for your Operating System
(Windows, macOS or Linux):

   https://github.com/conda-forge/miniforge#miniforge3

miniforge is a conda installation that uses the packages from the conda-forge
channel by default.

Create a dedicated conda environment for this class:

```bash
conda create -n dlclass python=3.13
conda activate dlclass
pip install -r requirements.txt
```

## Keras backend configuration

This course uses Keras 3 with the PyTorch backend. Set the backend before running
any notebooks:

```bash
# macOS/Linux
export KERAS_BACKEND=torch

# Windows (Command Prompt)
set KERAS_BACKEND=torch

# Windows (PowerShell)
$env:KERAS_BACKEND = "torch"
```

To make this permanent:
- **macOS/Linux**: Add the export line to your `~/.bashrc` or `~/.zshrc`
- **Windows**: Set via System Properties → Environment Variables, or add to your PowerShell profile

## Verification

Check that Keras 3 and PyTorch are installed correctly:

```bash
python -c "import keras; print(keras.__version__)"
python -c "import torch; print(torch.__version__)"
```

Keras should show version 3.x and PyTorch should show version 2.x.

## Jupyter kernel setup

If you have several installations of Python on your system (virtualenv, conda
environments...), it can be confusing to select the correct Python environment
from the Jupyter interface. You can register this environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name dlclass --display-name dlclass
```

Then create a new Jupyter notebook and check that you can import
numpy, matplotlib, keras, and torch modules.

## Webcam support (optional)

To take pictures with the webcam we will also need opencv-python:

```bash
pip install opencv-python
```

If your laptop does not have a webcam or if opencv does not work, don't worry
this is not mandatory.

# Troubleshooting

## Check your Python environment

Verify which Python is being used:

```bash
# macOS/Linux
which python

# Windows
where python

# All platforms
python --version
```

For conda users, check the installation location:

```bash
conda info
```

Read the output to verify that your conda command is installed where you expect
it to be. If it's not the case, you might need to adjust your PATH:
- **macOS/Linux**: Edit `~/.bashrc` or `~/.zshrc` to change the PATH order
- **Windows**: Modify PATH via System Properties → Environment Variables

## Check pip location

Check that the pip command in your PATH matches your Python environment:

```bash
pip show pip
python -m pip show pip
```

The "Location:" line should be a subfolder of your environment root.
