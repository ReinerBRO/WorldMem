#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

TS="${TS:-20260627_011523}"
RUN_TAG="${RUN_TAG:-ptm_v4_full_dit_oasis_serial_${TS}}"
LOG_DIR="${LOG_DIR:-/gfs/space/private/zjc/logs}"
MARKER_PREFIX="${LOG_DIR}/${RUN_TAG}"
WORLD_MEM_CACHE_DIR="${WORLD_MEM_CACHE_DIR:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/worldmem_minecraft_test_gen600x100_npz_cache_}"
PTMFREE_SHARD_DIR="${PTMFREE_SHARD_DIR:-/gfs/space/private/zjc/ptm/outputs/external_worldmem_ptmfree_10k_normal_20260626_worldmem16_eval/shard_indices}"
TRAIN_NPZ_CACHE_DIR="${TRAIN_NPZ_CACHE_DIR:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache}"
TRAIN_NPZ_CACHE_DIR_VAL="${TRAIN_NPZ_CACHE_DIR_VAL:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/gen600x100_npz_cache}"
V4A_RUN_NAME="${V4A_RUN_NAME:-ptm_v4a_causal_slot_token_only_full_dit_oasis_10k_${TS}}"
V4B_RUN_NAME="${V4B_RUN_NAME:-ptm_v4b_context_derived_token_only_full_dit_oasis_10k_${TS}}"

mkdir -p "${LOG_DIR}" /gfs/space/private/zjc/.cache /gfs/space/private/zjc/.cache/torch /gfs/space/private/zjc/.cache/matplotlib

echo "${RUN_TAG}" > "${MARKER_PREFIX}.runtag"
echo "$$" > "${MARKER_PREFIX}.runner"
echo "${LOG_DIR}/${RUN_TAG}.log" > "${MARKER_PREFIX}.logpath"
echo "resume_after_v4a" > "${MARKER_PREFIX}.current_phase"
touch "${MARKER_PREFIX}.runs"
grep -qxF "${V4A_RUN_NAME}" "${MARKER_PREFIX}.runs" || echo "${V4A_RUN_NAME}" >> "${MARKER_PREFIX}.runs"

if [[ ! -d "${WORLD_MEM_CACHE_DIR}" ]]; then
  echo "WORLD_MEM_CACHE_DIR not found: ${WORLD_MEM_CACHE_DIR}" >&2
  exit 2
fi
if [[ ! -d "${PTMFREE_SHARD_DIR}" ]]; then
  echo "PTMFREE_SHARD_DIR not found: ${PTMFREE_SHARD_DIR}" >&2
  exit 2
fi

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="/gfs/space/private/zjc/.cache"
export TORCH_HOME="/gfs/space/private/zjc/.cache/torch"
export MPLCONFIGDIR="/gfs/space/private/zjc/.cache/matplotlib"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

latest_step_ckpt() {
  local run_name="$1"
  local ckpt_dir="/gfs/space/private/zjc/ptm/outputs/${run_name}/checkpoints"
  local ckpt
  ckpt="$(find "${ckpt_dir}" -maxdepth 1 -type f -name '*step10000.ckpt' | sort | tail -n 1 || true)"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.ckpt' | sort | tail -n 1 || true)"
  fi
  if [[ -z "${ckpt}" ]]; then
    echo "no checkpoint found in ${ckpt_dir}" >&2
    exit 3
  fi
  printf '%s\n' "${ckpt}"
}

run_eval() {
  local train_name="$1"
  local eval_label="$2"
  local ckpt
  ckpt="$(latest_step_ckpt "${train_name}")"
  echo "[serial-continue] eval ${eval_label} ckpt=${ckpt}"
  echo "eval ${eval_label}" > "${MARKER_PREFIX}.current_phase"

  export WANDB_MODE=disabled
  export PTM_NPZ_CACHE_DIR="${WORLD_MEM_CACHE_DIR}"
  export PTM_ALLOW_NONFORMAL_NPZ_CACHE=true
  export PTM_REQUIRED_MEMORY_STRATEGY=causal_slots
  export PTM_NUM_SHARDS=8
  export PTM_GENERATION_BATCH_SIZE=2
  export PTM_GENERATION_LIMIT_BATCH=1
  export PTM_GENERATION_NUM_WORKERS=0
  export PTM_NPZ_CACHE_SPLIT=test
  export PTM_EVAL_LABEL="${eval_label}"
  export PTM_EVAL_ROOT="/gfs/space/private/zjc/ptm/outputs/${eval_label}"
  export PTM_CKPT="${ckpt}"
  export PTM_MEMORY_CONDITION_LENGTH=8
  export PTM_RAW_REFERENCE_LENGTH=0
  export PTM_CONTEXT_MEMORY_ONLY=true
  export PTM_CONTEXT_MEMORY_STRATEGY=strided
  export PTM_MAX_HISTORY=16
  export PTM_MAX_HISTORY_CANDIDATES=16
  export PTM_USE_MEMORY_ATTENTION=false
  export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
  export PTM_USE_PTM_REFERENCE_ADAPTER=false
  export PTM_USE_PTM_MEMORY=true
  export PTM_USE_PTM_CROSS_ATTENTION=true
  export PTM_ABLATIONS="normal zero_token shuffle_token"
  export PTM_VAL_ABLATION_MODES="${PTM_ABLATIONS}"
  export PTM_SHARD_INDICES_DIR="${PTMFREE_SHARD_DIR}"

  bash ptm/scripts/run_generation_context_memory_p0.sh
  if [[ -f "${PTM_EVAL_ROOT}/generation_summary.json" ]]; then
    echo "[serial-continue] eval summary ${PTM_EVAL_ROOT}/generation_summary.json"
    cat "${PTM_EVAL_ROOT}/generation_summary.json"
  else
    echo "missing eval summary: ${PTM_EVAL_ROOT}/generation_summary.json" >&2
    exit 4
  fi
}

prepare_train_env() {
  unset PTM_EVAL_LABEL PTM_EVAL_ROOT PTM_CKPT PTM_ABLATIONS PTM_VAL_ABLATION_MODES PTM_SHARD_INDICES_DIR
  unset PTM_REQUIRED_MEMORY_STRATEGY PTM_NUM_SHARDS PTM_GENERATION_BATCH_SIZE PTM_GENERATION_LIMIT_BATCH
  unset PTM_GENERATION_NUM_WORKERS PTM_USE_PTM_MEMORY

  export PTM_DATA_ROOT=ptm_minedojo_data/long_1500_360x640
  export PTM_NPZ_CACHE_DIR="${TRAIN_NPZ_CACHE_DIR}"
  export PTM_NPZ_CACHE_DIR_VAL="${TRAIN_NPZ_CACHE_DIR_VAL}"
  export PTM_NPZ_CACHE_SPLITS=train,val
  export PTM_NPZ_CACHE_SPLIT=train
  export PTM_ALLOW_NONFORMAL_NPZ_CACHE=false
  export PTM_MEMORY_CONDITION_LENGTH=8
  export PTM_RAW_REFERENCE_LENGTH=0
  export PTM_CONTEXT_MEMORY_ONLY=true
  export PTM_CONTEXT_MEMORY_STRATEGY=strided
  export PTM_USE_MEMORY_ATTENTION=false
  export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
  export PTM_USE_PTM_REFERENCE_ADAPTER=false
  export PTM_USE_PTM_CROSS_ATTENTION=true
}

run_train_v4b() {
  local run_name="${V4B_RUN_NAME}"
  grep -qxF "${run_name}" "${MARKER_PREFIX}.runs" || echo "${run_name}" >> "${MARKER_PREFIX}.runs"
  echo "train ${run_name}" > "${MARKER_PREFIX}.current_phase"
  echo "[serial-continue] train v4b full-DiT run=${run_name}"

  prepare_train_env
  export WANDB_MODE="${TRAIN_WANDB_MODE:-online}"
  export RUN_NAME="${run_name}"
  export OUTPUT_DIR="outputs/${run_name}"
  export PTM_INIT_MODE=oasis
  export PTM_BASE_CKPT=""
  export PTM_TRAIN_CONSUMER_ONLY=false
  export PTM_MAX_STEPS=10000
  export PTM_VAL_EVERY_N_STEP=2500
  export PTM_CKPT_EVERY=2500
  export PTM_BATCH_SIZE=8
  export PTM_VAL_BATCH_SIZE=2
  export PTM_VAL_NUM_WORKERS=0
  export PTM_LIMIT_VAL_BATCH=1
  export PTM_LOG_VIDEO=true
  export PTM_MAX_LOG_VIDEOS=1
  export WORLDMEM_LOCAL_LOSS_LOG="${LOG_DIR}/${run_name}_local_loss.log"
  bash ptm/scripts/train_ptm_v4_context_derived_token_only.sh
  run_eval "${run_name}" "external_worldmem_${run_name}_eval"
}

echo "[serial-continue] start $(date -u +%Y-%m-%dT%H:%M:%SZ) tag=${RUN_TAG}"
echo "[serial-continue] worldmem_cache=${WORLD_MEM_CACHE_DIR}"
echo "[serial-continue] shard_dir=${PTMFREE_SHARD_DIR}"
echo "[serial-continue] train_cache=${TRAIN_NPZ_CACHE_DIR}"
echo "[serial-continue] train_val_cache=${TRAIN_NPZ_CACHE_DIR_VAL}"
echo "[serial-continue] cuda=${CUDA_VISIBLE_DEVICES}"

run_eval "${V4A_RUN_NAME}" "external_worldmem_${V4A_RUN_NAME}_eval"
run_train_v4b

echo "complete" > "${MARKER_PREFIX}.current_phase"
echo "[serial-continue] complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
