#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace"
PYTHON="${PYTHON_BIN:-/opt/venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

CONFIG="${CONFIG:-${ROOT}/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco_instance_seg_rle.yml}"
DEVICE="${DEVICE:-cuda}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/datasets}"
TRAIN_IMG_FOLDER="${TRAIN_IMG_FOLDER:-${DATASET_ROOT}/train/img}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-${DATASET_ROOT}/instances_train.json}"
VAL_IMG_FOLDER="${VAL_IMG_FOLDER:-${DATASET_ROOT}/valid/img}"
VAL_ANN_FILE="${VAL_ANN_FILE:-${DATASET_ROOT}/instances_valid.json}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"

cd "${ROOT}/tools"
exec "${PYTHON}" "${ROOT}/tools/train.py" \
  -c "${CONFIG}" \
  -d "${DEVICE}" \
  -u "train_dataloader.total_batch_size=${TRAIN_BATCH_SIZE}" \
     "val_dataloader.total_batch_size=${VAL_BATCH_SIZE}" \
     "train_dataloader.dataset.img_folder=${TRAIN_IMG_FOLDER}" \
     "train_dataloader.dataset.ann_file=${TRAIN_ANN_FILE}" \
     "val_dataloader.dataset.img_folder=${VAL_IMG_FOLDER}" \
     "val_dataloader.dataset.ann_file=${VAL_ANN_FILE}"
