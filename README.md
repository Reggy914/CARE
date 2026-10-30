# CARE

CARE is a research implementation for univariate electrical-current time-series forecasting. It combines a frozen Timer backbone, a target-domain causal anchor, an independent LSTM expert, and validation-only reconciliation. The repository also includes the baseline and teacher-model implementations used by the experiment controller.

This release contains source code and configuration only. It intentionally excludes private datasets, pretrained weights, generated results, analysis workspaces, and paper sources.

## Overview

CARE is organized into three forecasting stages:

1. The Anchor adapts frozen Timer features with a causal target-domain branch.
2. The Expert learns an independent forecast from historical target values.
3. Reconciliation selects a horizon-wise interpolation profile using validation data only.

The public controller preserves chronological train/validation/test splits. Fitted preprocessing statistics come from the training split and are reused for validation and test data.

## Repository structure

~~~text
.
├── configs/             Experiment and model configuration
├── data_provider/       ELE preprocessing and chronological data loading
├── docs/                Installation, dataset, configuration, and usage guides
├── examples/            Small read-only utility examples
├── exp/                 Training, evaluation, inference, and runtime helpers
├── layers/              CARE building blocks
├── models/              CARE, baselines, and teacher models
├── scripts/             Shell wrappers for common workflows
├── utils/               Metrics, prediction, shapes, and training control
├── main.py              Unified command-line entry point
├── pyproject.toml       Package metadata
└── requirements.txt     Reproducible Python dependency set
~~~

## Requirements

- Linux is the primary supported platform.
- Python 3.10 or 3.11 is recommended.
- PyTorch 2.2 or newer is required.
- A CUDA-capable GPU is optional; CPU execution is supported but slower.
- The Timer checkpoint is an external dependency and is not included.

The release was checked with Python 3.10.20, PyTorch 2.11.0, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, PyYAML 6.0.3, and Transformers 4.40.1. Install the PyTorch build appropriate for your operating system and CUDA version before installing the remaining requirements.

## Installation

Clone the repository and create an isolated environment:

~~~bash
git clone https://github.com/Reggy914/CARE.git
cd CARE

conda create -n care python=3.11 -y
conda activate care

# Install a platform-appropriate PyTorch build first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
pip install -e .
python main.py --help
~~~

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for environment and troubleshooting details.

## Dataset preparation

The ELE preprocessor expects a CSV whose first column is a timestamp and whose remaining columns are numeric current measurements:

~~~bash
python main.py preprocess \
  --raw-csv /path/to/raw/electricity.csv \
  --output-dir data/processed/ele
~~~

It writes:

- ele_processed.csv
- data_profile.json
- columns.json
- split_info.json

No dataset is distributed with this repository. See [docs/DATASET.md](docs/DATASET.md) for the schema, split semantics, and leakage controls.

## Configuration

General experiment defaults live in configs/experiments/default.yaml. Model definitions live in configs/models. Command-line values override configuration defaults. Important path options are always explicit:

- --data-dir or --ele-data-dir selects processed data.
- --timer-model-dir selects the external Timer checkpoint.
- --output-dir selects generated artifacts.
- --checkpoint selects a trained model for evaluation or inference.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for configuration precedence and model names.

## Usage

### Train CARE

~~~bash
python main.py train \
  --ele-data-dir data/processed/ele \
  --timer-model-dir pretrained/timer-base-84m \
  --output-dir outputs \
  --target-columns MT_001 \
  --models care \
  --device cuda
~~~

Training writes a run directory under outputs/ele. Each successful model artifact contains its checkpoint, resolved configuration, training history, metrics, predictions, and run manifest.

Use --dry-run to validate target selection, splits, configuration, and output paths without training:

~~~bash
python main.py train \
  --ele-data-dir data/processed/ele \
  --target-columns MT_001 \
  --models care \
  --dry-run
~~~

### Evaluate a checkpoint

~~~bash
python main.py evaluate \
  --data-dir data/processed/ele \
  --target-column MT_001 \
  --model care \
  --checkpoint outputs/ele/<run-id>/seed_2024/target_MT_001/care/checkpoints/best_model.pt \
  --timer-model-dir pretrained/timer-base-84m \
  --output-dir outputs/evaluation/MT_001 \
  --device cuda
~~~

### Run inference

~~~bash
python main.py inference \
  --data-dir data/processed/ele \
  --target-column MT_001 \
  --model care \
  --checkpoint outputs/ele/<run-id>/seed_2024/target_MT_001/care/checkpoints/best_model.pt \
  --timer-model-dir pretrained/timer-base-84m \
  --split test \
  --output-dir outputs/inference/MT_001 \
  --device cuda
~~~

Equivalent wrappers are available in scripts. See [docs/USAGE.md](docs/USAGE.md) for complete examples and artifact descriptions.

## Checkpoints

The external Timer checkpoint is not bundled. The default location is:

~~~text
pretrained/timer-base-84m/
~~~

Download the public checkpoint separately:

~~~bash
huggingface-cli download thuml/timer-base-84m --local-dir pretrained/timer-base-84m
~~~

You may pass another local directory with --timer-model-dir or CARE_TIMER_MODEL_DIR. Training stores model checkpoints under outputs/ele/<run-id>/seed_<seed>/target_<target>/<model>/checkpoints/. Evaluation and inference load a specific checkpoint through --checkpoint. No trained CARE checkpoint is distributed with this release.

The Timer loader uses local files and its custom Transformers implementation. Inspect third-party model code before trusting a checkpoint. Load only trusted PyTorch checkpoints.

The Timer repository declares the checkpoint and its custom model files under Apache-2.0. Timer is an external resource and is not covered by CARE's MIT License. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Reproducibility notes

- Data splits are chronological.
- Scaling is fitted on training data only.
- Validation data is used for checkpoint selection and CARE reconciliation.
- Test data is reserved for final evaluation.
- The default experiment runs seeds 2024, 2025, and 2026.
- Use a stable --run-id together with --resume to continue a partially completed multi-model run.
- Exact floating-point results can vary by hardware, CUDA, and PyTorch version.

Saved checkpoints are Python/PyTorch artifacts. Load only checkpoints from a trusted source.

## Citation

Formal publication metadata was not inferred from the private source tree. A CITATION.cff file should be added after the authors, title, year, venue, DOI, and canonical repository URL are confirmed.

## License

CARE is released under the [MIT License](LICENSE).

Copyright (c) 2026 Jiaji Ren.

Third-party dependencies and the external Timer resource retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the audited license inventory and redistribution notes.

## Support

Please use the [repository issue tracker](https://github.com/Reggy914/CARE/issues) for reproducible bug reports. Include the command, resolved configuration, Python/PyTorch versions, device, and the complete error message. Do not attach private datasets, credentials, or model weights unless you are authorized to share them.
