#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

: "${CARE_RAW_CSV:?Set CARE_RAW_CSV to the raw ELE CSV path.}"
CARE_DATA_DIR="${CARE_DATA_DIR:-${PROJECT_ROOT}/data/processed/ele}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/main.py" preprocess \
  --raw-csv "${CARE_RAW_CSV}" \
  --output-dir "${CARE_DATA_DIR}" \
  "$@"
