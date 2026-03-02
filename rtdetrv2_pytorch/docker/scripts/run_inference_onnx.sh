#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
PYTHON="${PYTHON_BIN:-/opt/venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

CONFIG="${CONFIG:-${ROOT}/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yml}"
ONNX_MODEL="${ONNX_MODEL:-${ROOT}/exported_models/model.onnx}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/docker/input_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/docker/output_data}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"

cd "${ROOT}/tools"
exec "${PYTHON}" "${ROOT}/tools/inference.py" \
  -c "${CONFIG}" \
  --onnx_model "${ONNX_MODEL}" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-json
