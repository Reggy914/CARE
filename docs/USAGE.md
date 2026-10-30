# Usage

All commands run from the repository root. Paths below are examples; replace target names and checkpoint paths with values from your environment.

## 1. Prepare data

~~~bash
python main.py preprocess \
  --raw-csv /path/to/electricity.csv \
  --output-dir data/processed/ele
~~~

Inspect data/processed/ele/data_profile.json and split_info.json before training.

## 2. Discover targets

~~~bash
python examples/list_targets.py data/processed/ele
~~~

The target names come from columns.json.

## 3. Validate a run without training

~~~bash
python main.py train \
  --ele-data-dir data/processed/ele \
  --target-columns MT_001 \
  --models care \
  --timer-model-dir pretrained/timer-base-84m \
  --output-dir outputs \
  --dry-run
~~~

Dry-run mode validates the processed dataset and resolves the requested configuration. It does not instantiate or train the selected model.

## 4. Train

~~~bash
python main.py train \
  --ele-data-dir data/processed/ele \
  --target-columns MT_001 \
  --models care \
  --timer-model-dir pretrained/timer-base-84m \
  --output-dir outputs \
  --run-id care-main \
  --device cuda
~~~

To run every valid target column, replace --target-columns MT_001 with --all-columns. To run several models, pass their identifiers after --models.

A run uses this structure:

~~~text
outputs/ele/<run-id>/
├── run_config.json
├── run_summary.json
├── aggregate/
│   └── tables/
└── seed_<seed>/
    ├── run_summary.json
    └── target_<target>/
        └── <model>/
            ├── checkpoints/best_model.pt
            ├── logs/
            ├── metrics/test_metrics.json
            └── predictions/
~~~

The exact files vary slightly by model. Treat checkpoint.pt as trusted executable data: PyTorch checkpoints can contain Python-serialized objects.

To resume an interrupted run with the same identity:

~~~bash
python main.py train \
  --ele-data-dir data/processed/ele \
  --target-columns MT_001 \
  --models care \
  --timer-model-dir pretrained/timer-base-84m \
  --output-dir outputs \
  --run-id care-main \
  --resume
~~~

Do not change data, model settings, or window limits when resuming a run unless you intend to create a different experiment.

## 5. Evaluate

~~~bash
python main.py evaluate \
  --data-dir data/processed/ele \
  --target-column MT_001 \
  --model care \
  --checkpoint outputs/ele/care-main/seed_2024/target_MT_001/care/checkpoints/best_model.pt \
  --timer-model-dir pretrained/timer-base-84m \
  --output-dir outputs/evaluation/care-main-MT_001 \
  --device cuda
~~~

Evaluation uses the chronological test split and writes metrics.json, evaluation.json, and prediction arrays. The data window settings must match the checkpoint's training setup.

## 6. Inference

~~~bash
python main.py inference \
  --data-dir data/processed/ele \
  --target-column MT_001 \
  --model care \
  --checkpoint outputs/ele/care-main/seed_2024/target_MT_001/care/checkpoints/best_model.pt \
  --timer-model-dir pretrained/timer-base-84m \
  --split test \
  --output-dir outputs/inference/care-main-MT_001 \
  --device cuda
~~~

Inference writes predictions.npy, sample_id.npy, and inference.json. Predictions are in original target units.

## Shell wrappers

The scripts directory provides wrappers driven by environment variables:

~~~bash
CARE_RAW_CSV=/path/to/electricity.csv scripts/run_preprocess.sh

CARE_DATA_DIR=data/processed/ele \
CARE_TARGET=MT_001 \
CARE_TIMER_MODEL_DIR=pretrained/timer-base-84m \
scripts/run_train.sh
~~~

Open each script before use to see its required and optional variables.

## Baselines

Select another model with --models during training and --model during evaluation or inference. Non-Timer checkpoints are required for evaluation and inference. The frozen timer baseline can run directly from a Timer checkpoint without a CARE checkpoint.

Some models require a Timer checkpoint or teacher artifacts even when their own checkpoint is supplied. Pass those external paths explicitly.
