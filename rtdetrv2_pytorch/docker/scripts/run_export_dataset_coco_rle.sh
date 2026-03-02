#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
PYTHON="${PYTHON_BIN:-/opt/venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/datasets}"

cd "${ROOT}/nn-utilities/helpers"
exec "${PYTHON}" "${ROOT}/nn-utilities/helpers/supervisely_to_coco_rle.py" \
  --dataset_root "${DATASET_ROOT}" \
  --output_dir "${OUTPUT_DIR}"
