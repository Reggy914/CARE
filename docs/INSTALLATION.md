# Installation

## Supported environment

CARE targets Linux with Python 3.10 or 3.11. Python 3.12 may work with compatible third-party packages, but it is not the recommended reproducibility target. PyTorch 2.2 or newer is required.

CUDA is optional. Choose a PyTorch wheel or Conda package that matches the installed NVIDIA driver and CUDA runtime. The official selector is at https://pytorch.org/get-started/locally/.

## Conda installation

~~~bash
conda create -n care python=3.11 -y
conda activate care

# Install PyTorch using the command supplied by the official selector.
pip install -r requirements.txt
pip install -e .
~~~

## Virtual-environment installation

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install PyTorch using the command supplied by the official selector.
pip install -r requirements.txt
pip install -e .
~~~

The editable installation provides both python main.py and care command-line entry points.

## Verify the installation

~~~bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import numpy, pandas, sklearn, yaml, transformers"
python main.py --help
python main.py train --help
~~~

These checks do not download a model or start training.

## Install the external Timer checkpoint

The repository does not include pretrained model weights. Download the public Timer checkpoint separately:

~~~bash
huggingface-cli download thuml/timer-base-84m \
  --local-dir pretrained/timer-base-84m
~~~

The expected directory must contain a complete Transformers checkpoint, including its custom model implementation. You can store it elsewhere and pass --timer-model-dir /absolute/path/to/timer-base-84m.

The Timer loader is configured for local files, so routine runs do not silently download missing checkpoint files. Transformers uses trust_remote_code for this architecture. Review external model code and use only trusted revisions.

## Common installation problems

### ModuleNotFoundError

Confirm that the intended environment is active, then run:

~~~bash
python -m pip install -r requirements.txt
python -m pip install -e .
~~~

Using python -m pip avoids installing packages into a different interpreter.

### CUDA is unavailable

Check:

~~~bash
nvidia-smi
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"
~~~

If CUDA remains unavailable, use --device cpu. Do not use a CUDA device string unless torch.cuda.is_available() returns True.

### Timer cannot be loaded

Check that --timer-model-dir points to the checkpoint directory itself, not its parent. CARE deliberately does not fall back to an implicit network download.

### Out-of-memory errors

Reduce --batch-size. For quick interface checks, also limit the number of windows with --max-train-windows, --max-val-windows, and --max-test-windows. These options change the training workload and should be recorded when reporting results.
