#!/usr/bin/env bash
set -euo pipefail

PTM_CONTEXT_LENGTH="${PTM_CONTEXT_LENGTH:-4}"
PTM_FUTURE_LENGTH="${PTM_FUTURE_LENGTH:-4}"
PTM_N_FRAMES="${PTM_N_FRAMES:-$((PTM_CONTEXT_LENGTH + PTM_FUTURE_LENGTH))}"
PTM_MEMORY_CONDITION_LENGTH="${PTM_MEMORY_CONDITION_LENGTH:-8}"
PTM_FRAME_HEIGHT="${PTM_FRAME_HEIGHT:-360}"
PTM_FRAME_WIDTH="${PTM_FRAME_WIDTH:-640}"
PTM_VIDEO_CACHE_SIZE="${PTM_VIDEO_CACHE_SIZE:-2}"
PTM_BASE_CKPT="${PTM_BASE_CKPT:-/gfs/space/private/zjc/WorldMem/outputs/repro_train_stage1/6k_8gpu_wandb_20260621_234906/checkpoints/epoch0_step6000.ckpt}"
PTM_CACHE_ROOT="${PTM_CACHE_ROOT:-/gfs/space/private/zjc/.cache}"

export TORCH_HOME="${TORCH_HOME:-${PTM_CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${PTM_CACHE_ROOT}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PTM_CACHE_ROOT}/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PTM_CACHE_ROOT}/matplotlib}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

ARGS=(
  +name=ptm_worldmem_smoke
  dataset=ptm_minedojo
  dataset.save_dir="${PTM_DATA_ROOT:-ptm_minedojo_data/stage0}"
  dataset.resolution="[${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]"
  dataset.observation_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]"
  dataset.n_frames="${PTM_N_FRAMES}"
  dataset.context_length="${PTM_CONTEXT_LENGTH}"
  dataset.future_length="${PTM_FUTURE_LENGTH}"
  dataset.ptm_context_length="${PTM_CONTEXT_LENGTH}"
  dataset.ptm_future_length="${PTM_FUTURE_LENGTH}"
  dataset.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}"
  +dataset.video_cache_size="${PTM_VIDEO_CACHE_SIZE}"
  algorithm.x_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]"
  algorithm.context_frames="${PTM_CONTEXT_LENGTH}"
  +algorithm.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}"
  algorithm.use_ptm_memory=true
  +algorithm.log_video="${PTM_LOG_VIDEO:-false}"
  algorithm.num_memory_tokens="${PTM_MEMORY_TOKENS:-8}"
  algorithm.ptm_loss_weight="${PTM_LOSS_WEIGHT:-0.25}"
  experiment.training.max_steps="${PTM_MAX_STEPS:-100}"
  experiment.training.batch_size="${PTM_BATCH_SIZE:-1}"
  experiment.training.data.num_workers="${PTM_NUM_WORKERS:-1}"
  experiment.validation.limit_batch="${PTM_LIMIT_VAL_BATCH:-0}"
  wandb.mode="${WANDB_MODE:-disabled}"
)

if [[ -n "${PTM_BASE_CKPT:-}" && -f "${PTM_BASE_CKPT}" ]]; then
  ARGS+=(load="${PTM_BASE_CKPT}" +customized_load=true)
fi

python -m ptm.data.verify_dataset --data_root "${PTM_DATA_ROOT:-ptm_minedojo_data/stage0}"
python main.py "${ARGS[@]}"
