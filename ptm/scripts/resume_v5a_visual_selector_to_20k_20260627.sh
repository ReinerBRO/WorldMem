#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

export RUN_NAME="${RUN_NAME:-ptm_v5a_visual_selector_summary_visual_oasis_10k_20260627_015731}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
export PTM_LOCAL_SAVE_DIR="${PTM_LOCAL_SAVE_DIR:-/gfs/space/private/zjc/ptm/outputs/${RUN_NAME}}"
export PTM_RESUME_CKPT="${PTM_RESUME_CKPT:-/gfs/space/private/zjc/ptm/outputs/${RUN_NAME}/checkpoints/epoch1_step10000.ckpt}"
export PTM_WANDB_RESUME_ID="${PTM_WANDB_RESUME_ID:-cn3r8uog}"
export PTM_MAX_STEPS="${PTM_MAX_STEPS:-20000}"
export PTM_CKPT_EVERY="${PTM_CKPT_EVERY:-2500}"
export PTM_VAL_EVERY_N_STEP="${PTM_VAL_EVERY_N_STEP:-2500}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-jinczhu12-hkust}"
export WANDB_PROJECT="${WANDB_PROJECT:-ptm}"
export WORLDMEM_LOCAL_LOSS_LOG="${WORLDMEM_LOCAL_LOSS_LOG:-/gfs/space/private/zjc/logs/${RUN_NAME}_local_loss.log}"

if [[ ! -f "${PTM_RESUME_CKPT}" ]]; then
  echo "missing PTM_RESUME_CKPT=${PTM_RESUME_CKPT}" >&2
  exit 2
fi

exec bash ptm/scripts/train_ptm_v5_visual_selector.sh
