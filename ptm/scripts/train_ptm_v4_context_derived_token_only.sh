#!/usr/bin/env bash
set -euo pipefail

export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /gfs/space/private/zjc/.secrets/wandb_api_key 2>/dev/null || true)}"
if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "WANDB_API_KEY is not set and /gfs/space/private/zjc/.secrets/wandb_api_key is missing" >&2
  exit 2
fi

cd /gfs/space/private/zjc/ptm

RUN_NAME="${RUN_NAME:-ptm_v4b_context_derived_token_only_10k_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-/gfs/space/private/zjc/logs}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${RUN_NAME}}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
case "${OUTPUT_DIR}" in
  /*) LOCAL_OUTPUT_DIR="${OUTPUT_DIR}" ;;
  *) LOCAL_OUTPUT_DIR="/gfs/space/private/zjc/ptm/${OUTPUT_DIR}" ;;
esac

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-jinczhu12-hkust}"
export WANDB_PROJECT="${WANDB_PROJECT:-ptm}"
export WORLDMEM_LOCAL_LOSS_LOG="${WORLDMEM_LOCAL_LOSS_LOG:-${LOG_DIR}/${RUN_NAME}_local_loss.log}"

export PTM_RUN_NAME="${RUN_NAME}"
export PTM_OUTPUT_DIR="${OUTPUT_DIR}"
export PTM_LOCAL_SAVE_DIR="${PTM_LOCAL_SAVE_DIR:-${LOCAL_OUTPUT_DIR}}"
export PTM_INIT_MODE="${PTM_INIT_MODE:-oasis}"
export PTM_BASE_CKPT="${PTM_BASE_CKPT:-}"
export PTM_REQUIRE_BASE_CKPT="${PTM_REQUIRE_BASE_CKPT:-false}"

export PTM_DATA_ROOT="${PTM_DATA_ROOT:-ptm_minedojo_data/long_1500_360x640}"
export PTM_NPZ_CACHE_DIR="${PTM_NPZ_CACHE_DIR:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache}"
export PTM_NPZ_CACHE_DIR_VAL="${PTM_NPZ_CACHE_DIR_VAL:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/gen600x100_npz_cache}"
export PTM_NPZ_CACHE_SPLITS="${PTM_NPZ_CACHE_SPLITS:-train,val}"
export PTM_VIDEO_CACHE_SIZE="${PTM_VIDEO_CACHE_SIZE:-0}"

export PTM_CONTEXT_LENGTH="${PTM_CONTEXT_LENGTH:-4}"
export PTM_FUTURE_LENGTH="${PTM_FUTURE_LENGTH:-4}"
export PTM_VAL_CONTEXT_LENGTH="${PTM_VAL_CONTEXT_LENGTH:-600}"
export PTM_VAL_FUTURE_LENGTH="${PTM_VAL_FUTURE_LENGTH:-100}"
export PTM_MEMORY_CONDITION_LENGTH="${PTM_MEMORY_CONDITION_LENGTH:-8}"
export PTM_RAW_REFERENCE_LENGTH=0

export PTM_TRAIN_CONTEXT_MEMORY_ONLY=true
export PTM_TRAIN_CONTEXT_TOKEN_SOURCE=context
export PTM_CONTEXT_MEMORY_ONLY=true
export PTM_CONTEXT_MEMORY_STRATEGY="${PTM_CONTEXT_MEMORY_STRATEGY:-strided}"
export PTM_USE_MEMORY_ATTENTION=false
export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
export PTM_USE_PTM_REFERENCE_ADAPTER=false
export PTM_USE_PTM_CROSS_ATTENTION=true
export PTM_TRAIN_CONSUMER_ONLY="${PTM_TRAIN_CONSUMER_ONLY:-true}"
export PTM_DETACH_FOR_GENERATION=false

export PTM_LOSS_WEIGHT="${PTM_LOSS_WEIGHT:-0.1}"
export PTM_BOTTLENECK_WEIGHT="${PTM_BOTTLENECK_WEIGHT:-0.001}"
export PTM_CONTRAST_WEIGHT="${PTM_CONTRAST_WEIGHT:-0.1}"
export PTM_CONTRAST_MARGIN="${PTM_CONTRAST_MARGIN:-0.02}"
export PTM_GENERATION_TARGET_LOSS_WEIGHT="${PTM_GENERATION_TARGET_LOSS_WEIGHT:-1.0}"
export PTM_GENERATION_LATE_LOSS_WEIGHT="${PTM_GENERATION_LATE_LOSS_WEIGHT:-0.5}"
export PTM_GENERATION_TARGET_WINDOW_RADIUS="${PTM_GENERATION_TARGET_WINDOW_RADIUS:-1}"
export PTM_GENERATION_LATE_HORIZON_START="${PTM_GENERATION_LATE_HORIZON_START:-50}"

export PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-16}"
export PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-16}"
export PTM_VALIDATION_ABLATION_MODES="${PTM_VALIDATION_ABLATION_MODES:-normal,zero_token,shuffle_token}"
export PTM_VALIDATION_VIDEO_MODE="${PTM_VALIDATION_VIDEO_MODE:-normal}"

export PTM_MAX_STEPS="${PTM_MAX_STEPS:-10000}"
export PTM_BATCH_SIZE="${PTM_BATCH_SIZE:-8}"
export PTM_NUM_WORKERS="${PTM_NUM_WORKERS:-8}"
export PTM_VAL_BATCH_SIZE="${PTM_VAL_BATCH_SIZE:-2}"
export PTM_VAL_NUM_WORKERS="${PTM_VAL_NUM_WORKERS:-0}"
export PTM_LIMIT_VAL_BATCH="${PTM_LIMIT_VAL_BATCH:-1}"
export PTM_VAL_EVERY_N_STEP="${PTM_VAL_EVERY_N_STEP:-2500}"
export PTM_CKPT_EVERY="${PTM_CKPT_EVERY:-2500}"
export PTM_PRECISION="${PTM_PRECISION:-16-mixed}"
export PTM_LOG_VIDEO="${PTM_LOG_VIDEO:-true}"
export PTM_MAX_LOG_VIDEOS="${PTM_MAX_LOG_VIDEOS:-1}"

echo "[v4b] run=${RUN_NAME}"
echo "[v4b] base=${PTM_BASE_CKPT}"
echo "[v4b] train_cache=${PTM_NPZ_CACHE_DIR}"
echo "[v4b] val_cache=${PTM_NPZ_CACHE_DIR_VAL}"
echo "[v4b] context_derived_token_only=true raw_reference_length=${PTM_RAW_REFERENCE_LENGTH} memory_attention_runtime=${PTM_USE_MEMORY_ATTENTION_RUNTIME}"

exec bash ptm/scripts/train_ptm_main.sh
