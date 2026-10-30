# CARE Release Report

## 1. Project Overview

This report describes the repository root. It is a standalone, source-only CARE codebase for univariate electrical-current forecasting. It contains data preprocessing and loading, model training and validation, checkpoint evaluation, inference, metrics, configuration, command wrappers, documentation, packaging metadata, and static GitHub checks.

The source project was treated as read-only. Its pre-migration and post-migration metadata fingerprints are identical:
The source path is not inside a Git working tree, so no Git status was available; metadata fingerprinting was used to detect any change.

~~~text
deea6fe0166c2d885520ad75c1a190f49fb05a37b6ed89955f06c352ea09a60b
~~~

The release directory was initialized as a local Git repository on the main branch. No remote repository was created and no push was attempted.

## 2. Repository Structure

~~~text
CARE_GitHub_Release/
├── .github/
│   ├── ISSUE_TEMPLATE/bug_report.yml
│   └── workflows/static-checks.yml
├── configs/
│   ├── experiments/default.yaml
│   └── models/*.yaml
├── data_provider/
│   ├── ele.py
│   └── prepare.py
├── docs/
│   ├── CONFIGURATION.md
│   ├── DATASET.md
│   ├── INSTALLATION.md
│   └── USAGE.md
├── examples/list_targets.py
├── exp/
│   ├── evaluate.py
│   ├── inference.py
│   ├── runner.py
│   ├── runtime.py
│   └── teacher_selection.py
├── layers/
│   ├── anchor.py
│   ├── expert.py
│   ├── reconciliation.py
│   └── timer_backbone.py
├── models/
│   ├── care.py
│   ├── care_anchor.py
│   ├── care_expert.py
│   ├── distill.py
│   ├── gtr.py
│   ├── itransformer.py
│   ├── normal_head.py
│   ├── output_correction.py
│   ├── patchtst.py
│   ├── timemixer.py
│   ├── timer.py
│   └── timer_finetune.py
├── scripts/
│   ├── run_evaluate.sh
│   ├── run_inference.sh
│   ├── run_preprocess.sh
│   └── run_train.sh
├── utils/
│   ├── evaluator.py
│   ├── predictor.py
│   ├── tensor_shapes.py
│   └── training_control.py
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── .gitattributes
├── .gitignore
├── main.py
├── pyproject.toml
├── README.md
├── RELEASE_REPORT.md
└── requirements.txt
~~~

Package initializers are present where required but omitted from this compact tree.

## 3. Code Migration

Forty-two source files were migrated or refactored:

- 5 care_core files moved to layers;
- 13 model files retained under models;
- 6 pipeline files moved to data_provider, exp, or utils according to responsibility;
- controllers/run.py became exp/runner.py;
- 17 YAML files were retained, with configs/experiments/current.yaml renamed to configs/experiments/default.yaml.

The release adds real public entry points for preprocessing, evaluation, and inference around the existing model and metric interfaces. It also adds packaging metadata, shell wrappers, English documentation, one minimal target-discovery example, and static-only GitHub support files.

Model and layer diffs were audited against the source. CARE forward propagation, network construction, loss functions, optimizers, training stages, and metrics were not changed. Model-tree changes are limited to:

- internal import relocation;
- replacement of one private absolute Timer path with a repository-relative configurable default;
- removal of historical paper-version wording while preserving validation constraints;
- removal of three confirmed stale imported symbols.

One stale import was also removed from the migrated teacher-selection module. Release controller changes concern public paths, output placement, CLI handling, and failure exit status.

## 4. Dependency Audit

AST import analysis found these third-party runtime import roots:

- numpy;
- pandas;
- scikit-learn;
- torch;
- transformers;
- PyYAML.

huggingface-hub is declared for the documented checkpoint download workflow. tabulate is declared so pandas can render Markdown metric tables; the controller has a CSV fallback if it is unavailable.

requirements.txt and pyproject.toml declare compatible, bounded dependencies and Python 3.10 through 3.12. The isolated validation environment used:

- Python 3.10.20;
- PyTorch 2.11.0;
- NumPy 2.2.6;
- pandas 2.3.3;
- scikit-learn 1.7.2;
- PyYAML 6.0.3;
- Transformers 4.40.1;
- huggingface-hub 0.36.2.

CUDA is optional and was not available during validation. Timer weights are not bundled. The documented public external checkpoint is thuml/timer-base-84m.

All local import roots resolve within the release: data_provider, exp, layers, models, and utils. No source-project import remains.

The package-discovery configuration was corrected to include data_provider and the YAML configuration package. A local wheel built successfully and contained main.py, all runtime packages, and both experiment and model YAML files.

## 5. Removed and Excluded Files

No source file was deleted or modified. The following 10,632 source files were intentionally not migrated:

| Category | Files | Reason |
|---|---:|---|
| results | 10,555 | Generated metrics, predictions, and experiment artifacts are prohibited |
| analysis | 46 | Analysis outputs and reports are outside the release scope |
| tests | 12 | Independent test files and test directories are explicitly excluded |
| plot | 18 | Paper figures and plotting material are explicitly excluded |
| controllers/scheduler.py | 1 | Machine-specific scheduling wrapper with a hard-coded interpreter; not required by the independent controller |

No paper, LaTeX, BibTeX, PDF, notebook, dataset, checkpoint, prediction, result, log, cache, or large binary file is present. Temporary synthetic validation data, Python caches, wheel output, build output, and package metadata were removed after inspection.

## 6. Path Migration

Private and machine-specific paths were removed from the release.

- The Timer checkpoint defaults to pretrained/timer-base-84m and can be overridden with --timer-model-dir or CARE_TIMER_MODEL_DIR.
- Processed data defaults to data/processed/ele and can be overridden through the CLI.
- Training artifacts default to outputs and can be redirected with --output-dir.
- Evaluation and inference accept explicit data, checkpoint, teacher, Timer, device, and output paths.
- Distillation cache output is placed under the selected output root.
- No reference to /root, /home, /mnt, the source project, or an author server remains in release content.

## 7. Runtime Verification

The following checks were actually completed without training a model:

- all final Python files were parsed and compiled in memory;
- all YAML files and pyproject.toml were parsed;
- all 16 configured model identifiers resolved to existing classes;
- core local modules imported successfully in the isolated environment;
- the root CLI and preprocess, train, evaluate, and inference help interfaces started successfully;
- all shell scripts passed bash syntax checking;
- all four shell wrappers reached their intended entry points;
- preprocessing completed on a temporary 1,200-row, two-target CSV;
- preprocessing produced ele_processed.csv, data_profile.json, columns.json, and split_info.json;
- target discovery returned both temporary targets;
- a training dry run built chronological train, validation, and test loaders without fitting a model;
- train-only scaler metadata and globally aligned sample IDs were produced;
- PatchTST, iTransformer, TimeMixer, and the fixed 128-dimensional two-layer CARE Expert instantiated on CPU;
- each timer-free forward check returned a finite tensor with shape (2, 12, 1);
- a local wheel built successfully and contained runtime code and YAML configuration;
- a conservative AST unused-import scan passed after confirmed stale imports were removed;
- the source metadata fingerprint matched its baseline after all release work.

The GitHub workflow performs only Python bytecode compilation and shell syntax checking. It does not claim end-to-end model validation.

## 8. Known Limitations

- A complete CARE/Anchor/Timer run was not performed because the external Timer checkpoint is not bundled.
- Evaluation and inference were startup-checked but not run against a trained CARE checkpoint, because no trained checkpoint or real public dataset was supplied and fabricating one is prohibited.
- No model was trained or fine-tuned.
- The dependency set was validated in an existing isolated environment, not installed from scratch into a newly created environment. tabulate was not present in that validation environment; its call site used a guarded fallback and it is declared in the release requirements.
- Exact real-data reproducibility remains dependent on the user's dataset, Timer revision, hardware, and trained checkpoint.
- CITATION.cff is intentionally absent until the publication title, year, venue, and DOI are confirmed. This optional metadata does not block the source-code release.

## 9. GitHub Readiness

The code, paths, documentation, package metadata, ignore rules, static workflow, and issue template are suitable for a clean GitHub repository. The release contains no generated results, tests, analysis, paper files, private paths, secrets, weights, or large files.

The repository is licensed under MIT with Copyright (c) 2026 Jiaji Ren. Direct dependencies and the external Timer resource are documented in THIRD_PARTY_NOTICES.md and retain their upstream licenses.

The source-code repository is ready for an initial public release. Verified publication metadata can be added later in CITATION.cff without changing the released implementation.
