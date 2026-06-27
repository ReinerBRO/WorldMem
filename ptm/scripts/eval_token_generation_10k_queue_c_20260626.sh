#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PTM_NUM_SHARDS=4
export PTM_GENERATION_BATCH_SIZE=2
export PTM_GENERATION_LIMIT_BATCH=1
export PTM_GENERATION_NUM_WORKERS=0
export PTM_ABLATIONS="normal zero_token shuffle_token"
export PTM_VAL_ABLATION_MODES="normal zero_token shuffle_token"
export PTM_NPZ_CACHE_SPLIT=test
export PTM_MAX_HISTORY=16
export PTM_MAX_HISTORY_CANDIDATES=16
export PTM_USE_PTM_CROSS_ATTENTION=true
export PTM_USE_PTM_REFERENCE_ADAPTER=false
export WANDB_MODE=disabled

run_eval() {
  local label="$1"
  local ckpt="$2"
  local memattn="$3"

  export PTM_EVAL_LABEL="${label}"
  export PTM_EVAL_ROOT="/gfs/space/private/zjc/ptm/outputs/${label}"
  export PTM_CKPT="${ckpt}"
  export PTM_USE_MEMORY_ATTENTION="${memattn}"

  echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] start label=${label} memattn=${memattn} ckpt=${ckpt} cuda=${CUDA_VISIBLE_DEVICES}"
  if [ -f "${PTM_EVAL_ROOT}/generation_summary.json" ]; then
    echo "summary already exists: ${PTM_EVAL_ROOT}/generation_summary.json"
    return 0
  fi
  bash /gfs/space/private/zjc/ptm/ptm/scripts/run_generation_ablation_clean.sh
  echo "[$(date +%Y-%m-%dT%H:%M:%S%z)] done label=${label}"
}

run_eval killswitch1_token_v1_10k_memattn_false /gfs/space/private/zjc/ptm/outputs/ptm_v1_full_contrast_detach_15k/checkpoints/epoch0_step10000.ckpt false
run_eval killswitch1_token_v2_10k_memattn_false /gfs/space/private/zjc/ptm/outputs/ptm_v2_no_contrast_15k/checkpoints/epoch0_step10000.ckpt false
