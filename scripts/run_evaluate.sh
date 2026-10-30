#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${CARE_DATA_DIR:?Set CARE_DATA_DIR to the processed ELE directory.}"
: "${CARE_TARGET:?Set CARE_TARGET to a target column name.}"
: "${CARE_CHECKPOINT:?Set CARE_CHECKPOINT to a trained checkpoint.}"
CARE_MODEL="${CARE_MODEL:-care}"
CARE_TIMER_MODEL_DIR="${CARE_TIMER_MODEL_DIR:-${PROJECT_ROOT}/pretrained/timer-base-84m}"
CARE_OUTPUT_DIR="${CARE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/evaluation}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py" evaluate \
  --data-dir "${CARE_DATA_DIR}" \
  --target-column "${CARE_TARGET}" \
  --model "${CARE_MODEL}" \
  --checkpoint "${CARE_CHECKPOINT}" \
  --timer-model-dir "${CARE_TIMER_MODEL_DIR}" \
  --output-dir "${CARE_OUTPUT_DIR}" \
  "$@"
