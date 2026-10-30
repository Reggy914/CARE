# Configuration

## Configuration layers

CARE resolves settings from:

1. configs/models/<model>.yaml;
2. configs/experiments/default.yaml;
3. saved checkpoint configuration for evaluation or inference;
4. explicit command-line overrides.

Later layers override earlier layers. Runtime-only paths saved in an old checkpoint are discarded before current command-line paths are applied.

## Experiment defaults

configs/experiments/default.yaml defines window lengths, optimization defaults, seeds, and the default model list. The release defaults are:

~~~yaml
input_len: 96
pred_len: 96
stride: 96
epochs: 20
batch_size: 128
learning_rate: 0.001
seeds: [2024, 2025, 2026]
~~~

Use command-line options for a specific run instead of editing code. Keep a copy of the resolved configuration generated with each artifact.

## Available model identifiers

- care: full CARE model;
- care_anchor: Anchor stage without the independent Expert;
- care_expert: standalone Expert;
- timer: frozen Timer inference baseline;
- timer_finetune: Timer fine-tuning baseline;
- patchtst: PatchTST baseline;
- itransformer: iTransformer baseline;
- timemixer: TimeMixer baseline;
- gtr: GTR teacher model;
- normal_head: normal-head teacher model;
- output_correction: output-correction teacher model;
- distill: distilled student using a selected teacher.

Additional ablation configuration files are present in configs/models and can be selected by their filename stem.

## External paths

No machine-specific absolute path is embedded in the release.

- --ele-data-dir is used by training.
- --data-dir is used by evaluation and inference.
- --timer-model-dir points to the external Timer checkpoint.
- --teacher-dir points to an explicit teacher artifact where required.
- --output-dir contains all generated files.

The training controller also accepts CARE_TIMER_MODEL_DIR and CARE_PRETRAINED_ROOT. Explicit command-line values are preferred because they are visible in shell history and experiment records.

## Device selection

Use one of:

~~~text
--device cpu
--device cuda
--device cuda:0
~~~

Evaluation and inference default to CUDA only when PyTorch reports that CUDA is available. Record the device and software versions with published results.

## Reproducibility controls

- --seed runs one seed.
- --seeds accepts one or more seeds and overrides the default list.
- --run-id chooses a stable output-directory name.
- --resume skips model artifacts that the controller verifies as successful.
- --dry-run resolves data, targets, models, and paths without fitting a model.

Window-limit flags are intended for debugging and resource control. They alter the effective sample set and must not be mixed silently with full-data results.
