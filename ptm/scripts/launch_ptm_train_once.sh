#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
LOG_DIR="${LOG_DIR:-/gfs/space/private/zjc/logs}"
GPU_MEM_THRESHOLD_MB="${GPU_MEM_THRESHOLD_MB:-1000}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}"

gpu_busy_count() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' \
    | wc -l \
    | tr -d ' '
}

gpu_max_mem() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk 'BEGIN { max = 0 } { if ($1 > max) max = $1 } END { print max + 0 }'
}

if [[ "${ALLOW_BUSY_GPU}" != "1" ]]; then
  busy="$(gpu_busy_count)"
  max_mem="$(gpu_max_mem)"
  if [[ "${busy}" -ne 0 || "${max_mem}" -gt "${GPU_MEM_THRESHOLD_MB}" ]]; then
    echo "GPUs are busy: compute_apps=${busy} max_mem_mb=${max_mem}" >&2
    exit 3
  fi
fi

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_ENTITY="${WANDB_ENTITY:-jinczhu12-hkust}"
export WANDB_PROJECT="${WANDB_PROJECT:-PTM}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PTM_DATA_ROOT="${PTM_DATA_ROOT:-ptm_minedojo_data/stage1_360x640}"
export PTM_MAX_STEPS="${PTM_MAX_STEPS:-6000}"
export PTM_BATCH_SIZE="${PTM_BATCH_SIZE:-1}"
export PTM_NUM_WORKERS="${PTM_NUM_WORKERS:-8}"
export PTM_VAL_BATCH_SIZE="${PTM_VAL_BATCH_SIZE:-1}"
export PTM_VAL_NUM_WORKERS="${PTM_VAL_NUM_WORKERS:-4}"
export PTM_LIMIT_VAL_BATCH="${PTM_LIMIT_VAL_BATCH:-2}"
export PTM_VAL_EVERY_N_STEP="${PTM_VAL_EVERY_N_STEP:-1500}"
export PTM_VAL_CONTEXT_LENGTH="${PTM_VAL_CONTEXT_LENGTH:-600}"
export PTM_VAL_FUTURE_LENGTH="${PTM_VAL_FUTURE_LENGTH:-100}"
export PTM_N_FRAMES_VALID="${PTM_N_FRAMES_VALID:-$((PTM_VAL_CONTEXT_LENGTH + PTM_VAL_FUTURE_LENGTH))}"
export PTM_MEMORY_CONDITION_LENGTH="${PTM_MEMORY_CONDITION_LENGTH:-8}"
export PTM_CKPT_EVERY="${PTM_CKPT_EVERY:-1500}"
export PTM_CKPT_SAVE_LAST="${PTM_CKPT_SAVE_LAST:-true}"
export PTM_LOG_VIDEO="${PTM_LOG_VIDEO:-true}"
export PTM_MAX_LOG_VIDEOS="${PTM_MAX_LOG_VIDEOS:-1}"
export PTM_VIDEO_LOG_STAGE="${PTM_VIDEO_LOG_STAGE:-ptm}"
export PTM_USE_MEMORY_ATTENTION="${PTM_USE_MEMORY_ATTENTION:-false}"
export PTM_USE_PTM_REFERENCE_ADAPTER="${PTM_USE_PTM_REFERENCE_ADAPTER:-true}"
export PTM_VIDEO_CACHE_SIZE="${PTM_VIDEO_CACHE_SIZE:-0}"
PTM_FORMAL_NPZ_CACHE_DIR="/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache"
export PTM_NPZ_CACHE_DIR="${PTM_NPZ_CACHE_DIR:-${PTM_FORMAL_NPZ_CACHE_DIR}}"
if [[ "${PTM_NPZ_CACHE_DIR}" != "${PTM_FORMAL_NPZ_CACHE_DIR}" ]]; then
  echo "refusing non-formal PTM_NPZ_CACHE_DIR=${PTM_NPZ_CACHE_DIR}; expected ${PTM_FORMAL_NPZ_CACHE_DIR}" >&2
  exit 2
fi
export PTM_NPZ_CACHE_SPLITS="${PTM_NPZ_CACHE_SPLITS:-train}"
export PTM_NPZ_VAL_INDICES_FILE="${PTM_NPZ_VAL_INDICES_FILE:-}"
export PTM_VAL_INDICES_FILE="${PTM_VAL_INDICES_FILE:-}"
export PTM_REQUIRED_MEMORY_STRATEGY="${PTM_REQUIRED_MEMORY_STRATEGY:-causal_slots}"
export PTM_VERIFY_DATASET="${PTM_VERIFY_DATASET:-false}"
export PTM_VALIDATION_ABLATION_MODES="${PTM_VALIDATION_ABLATION_MODES:-normal}"
export PTM_VALIDATION_VIDEO_MODE="${PTM_VALIDATION_VIDEO_MODE:-normal}"
export PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-$(( PTM_N_FRAMES_VALID + PTM_MEMORY_CONDITION_LENGTH ))}"
export PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-${PTM_MAX_HISTORY}}"
export PTM_INIT_MODE="${PTM_INIT_MODE:-oasis}"
export PTM_DIFFUSION_CKPT="${PTM_DIFFUSION_CKPT:-/gfs/space/private/zjc/models/oasis-500m/oasis500m.safetensors}"
export PTM_VAE_CKPT="${PTM_VAE_CKPT:-/gfs/space/private/zjc/models/oasis-500m/vit-l-20.safetensors}"
export PTM_ZERO_INIT_GATE="${PTM_ZERO_INIT_GATE:-true}"
export PTM_RUN_NAME="${PTM_RUN_NAME:-ptm_oasis_stage1_6k_audit2994_$(date +%Y%m%d_%H%M%S)}"
export PTM_OUTPUT_DIR="${PTM_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/${PTM_RUN_NAME}}"
export WORLDMEM_LOCAL_LOSS_LOG="${WORLDMEM_LOCAL_LOSS_LOG:-${LOG_DIR}/${PTM_RUN_NAME}_local_loss.log}"

train_log="${LOG_DIR}/${PTM_RUN_NAME}.log"
env_file="${LOG_DIR}/${PTM_RUN_NAME}.env"
pid_file="${LOG_DIR}/${PTM_RUN_NAME}.pid"

if [[ "${PTM_LOG_VIDEO}" != "true" ]]; then
  echo "refusing PTM_LOG_VIDEO=${PTM_LOG_VIDEO}; PTM training requires validation video logging." >&2
  exit 2
fi

if [[ "${PTM_VAL_EVERY_N_STEP}" =~ ^[0-9]+$ && "${PTM_MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  if (( PTM_VAL_EVERY_N_STEP > PTM_MAX_STEPS )); then
    echo "refusing PTM_VAL_EVERY_N_STEP=${PTM_VAL_EVERY_N_STEP} > PTM_MAX_STEPS=${PTM_MAX_STEPS}; this would skip validation for the whole run." >&2
    exit 2
  fi
fi

{
  echo "PTM_RUN_NAME=${PTM_RUN_NAME}"
  echo "PTM_DATA_ROOT=${PTM_DATA_ROOT}"
  echo "PTM_OUTPUT_DIR=${PTM_OUTPUT_DIR}"
  echo "PTM_MAX_STEPS=${PTM_MAX_STEPS}"
  echo "PTM_BATCH_SIZE=${PTM_BATCH_SIZE}"
  echo "PTM_VAL_BATCH_SIZE=${PTM_VAL_BATCH_SIZE}"
  echo "PTM_LIMIT_VAL_BATCH=${PTM_LIMIT_VAL_BATCH}"
  echo "PTM_VAL_EVERY_N_STEP=${PTM_VAL_EVERY_N_STEP}"
  echo "PTM_VAL_CONTEXT_LENGTH=${PTM_VAL_CONTEXT_LENGTH}"
  echo "PTM_VAL_FUTURE_LENGTH=${PTM_VAL_FUTURE_LENGTH}"
  echo "PTM_N_FRAMES_VALID=${PTM_N_FRAMES_VALID}"
  echo "PTM_MEMORY_CONDITION_LENGTH=${PTM_MEMORY_CONDITION_LENGTH}"
  echo "PTM_CKPT_EVERY=${PTM_CKPT_EVERY}"
  echo "PTM_CKPT_SAVE_LAST=${PTM_CKPT_SAVE_LAST}"
  echo "PTM_LOG_VIDEO=${PTM_LOG_VIDEO}"
  echo "PTM_MAX_LOG_VIDEOS=${PTM_MAX_LOG_VIDEOS}"
  echo "PTM_VIDEO_LOG_STAGE=${PTM_VIDEO_LOG_STAGE}"
  echo "PTM_USE_MEMORY_ATTENTION=${PTM_USE_MEMORY_ATTENTION}"
  echo "PTM_USE_PTM_REFERENCE_ADAPTER=${PTM_USE_PTM_REFERENCE_ADAPTER}"
  echo "PTM_VIDEO_CACHE_SIZE=${PTM_VIDEO_CACHE_SIZE}"
  echo "PTM_NPZ_CACHE_DIR=${PTM_NPZ_CACHE_DIR}"
  echo "PTM_NPZ_CACHE_SPLITS=${PTM_NPZ_CACHE_SPLITS}"
  echo "PTM_NPZ_VAL_INDICES_FILE=${PTM_NPZ_VAL_INDICES_FILE}"
  echo "PTM_VAL_INDICES_FILE=${PTM_VAL_INDICES_FILE}"
  echo "PTM_REQUIRED_MEMORY_STRATEGY=${PTM_REQUIRED_MEMORY_STRATEGY}"
  echo "PTM_VERIFY_DATASET=${PTM_VERIFY_DATASET}"
  echo "PTM_VALIDATION_ABLATION_MODES=${PTM_VALIDATION_ABLATION_MODES}"
  echo "PTM_VALIDATION_VIDEO_MODE=${PTM_VALIDATION_VIDEO_MODE}"
  echo "PTM_MAX_HISTORY=${PTM_MAX_HISTORY}"
  echo "PTM_MAX_HISTORY_CANDIDATES=${PTM_MAX_HISTORY_CANDIDATES}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "WANDB_MODE=${WANDB_MODE}"
  echo "WANDB_ENTITY=${WANDB_ENTITY}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "HF_ENDPOINT=${HF_ENDPOINT}"
  echo "PTM_INIT_MODE=${PTM_INIT_MODE}"
  echo "PTM_DIFFUSION_CKPT=${PTM_DIFFUSION_CKPT}"
  echo "PTM_VAE_CKPT=${PTM_VAE_CKPT}"
  echo "PTM_ZERO_INIT_GATE=${PTM_ZERO_INIT_GATE}"
  echo "PTM_BASE_CKPT=${PTM_BASE_CKPT:-}"
  echo "WORLDMEM_LOCAL_LOSS_LOG=${WORLDMEM_LOCAL_LOSS_LOG}"
  echo "train_log=${train_log}"
} > "${env_file}"

nohup bash ptm/scripts/train_ptm_main.sh > "${train_log}" 2>&1 < /dev/null &
pid="$!"
echo "${pid}" > "${pid_file}"

echo "TRAIN_PID=${pid}"
echo "RUN_NAME=${PTM_RUN_NAME}"
echo "LOG=${train_log}"
echo "OUT=${PTM_OUTPUT_DIR}"
echo "ENV=${env_file}"
echo "PIDFILE=${pid_file}"
