#!/usr/bin/env bash
# MemoryWAM RoboMME training on 4 GPUs with FSDP FULL_SHARD.
#
# Usage: bash scripts/train_fsdp4.sh [hydra_overrides...]
#   GPUS=4,5,3,2 bash scripts/train_fsdp4.sh task=memorywam_robomme
#
# The paper trains this model on 8 GPUs. Four is the approved local scale; the
# measured per-GPU budget is in scripts/accelerate_configs/accelerate_fsdp4_memorywam.yaml.
set -euo pipefail

GPUS="${GPUS:-2,3,4,5}"
NUM_GPUS="$(awk -F',' '{print NF}' <<< "${GPUS}")"
CONFIG=scripts/accelerate_configs/accelerate_fsdp4_memorywam.yaml

if [[ "${NUM_GPUS}" -ne 4 ]]; then
  echo "Error: this launcher is configured for 4 GPUs, got ${NUM_GPUS} (GPUS=${GPUS})." >&2
  echo "Adjust num_processes in ${CONFIG} and re-check the memory budget first." >&2
  exit 1
fi

echo "MemoryWAM FSDP training on GPUs ${GPUS}"
CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch \
  --config_file "${CONFIG}" \
  --num_processes "${NUM_GPUS}" \
  --main_process_port "${MASTER_PORT:-29500}" \
  scripts/train.py "$@"
