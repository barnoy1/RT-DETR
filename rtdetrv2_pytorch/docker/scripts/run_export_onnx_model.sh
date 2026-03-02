#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
PYTHON="${PYTHON_BIN:-/opt/venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

CONFIG="${CONFIG:-${ROOT}/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yml}"
WEIGHTS="${WEIGHTS:-${ROOT}/weights/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth}"
OUTPUT_ONNX="${OUTPUT_ONNX:-${ROOT}/exported_models/model.onnx}"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "Weights file not found: ${WEIGHTS}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ONNX}")"

cd "${ROOT}/tools"
exec "${PYTHON}" "${ROOT}/tools/export_onnx.py" \
  -c "${CONFIG}" \
  -r "${WEIGHTS}" \
  -u "PResNet.pretrained=False" \
  -o "${OUTPUT_ONNX}" \
  --check \
  --simplify
