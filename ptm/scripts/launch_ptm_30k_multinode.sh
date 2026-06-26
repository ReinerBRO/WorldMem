#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${NODE_RANK:?set NODE_RANK to 0 on the master node and 1 on the worker node}"
RUN="${RUN:?set RUN to the shared run name before launching both nodes}"
MASTER_ADDR="${MASTER_ADDR:-10.244.77.143}"
MASTER_PORT="${MASTER_PORT:-29631}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

cd /gfs/space/private/zjc/ptm
export PATH=/gfs/space/private/zjc/envs/worldmem/bin:${PATH}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-jinczhu12-hkust}"
export WANDB_PROJECT="${WANDB_PROJECT:-PTM}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /gfs/space/private/zjc/.secrets/wandb_api_key)}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

export PTM_RUN_NAME="${RUN}"
export PTM_OUTPUT_DIR="/gfs/space/private/zjc/ptm/outputs/${RUN}"
export PTM_DATA_ROOT="${PTM_DATA_ROOT:-ptm_minedojo_data/long_1500_360x640}"
PTM_FORMAL_NPZ_CACHE_DIR="/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache"
export PTM_NPZ_CACHE_DIR="${PTM_NPZ_CACHE_DIR:-${PTM_FORMAL_NPZ_CACHE_DIR}}"
if [[ "${PTM_NPZ_CACHE_DIR}" != "${PTM_FORMAL_NPZ_CACHE_DIR}" ]]; then
  echo "refusing non-formal PTM_NPZ_CACHE_DIR=${PTM_NPZ_CACHE_DIR}; expected ${PTM_FORMAL_NPZ_CACHE_DIR}" >&2
  exit 2
fi
export PTM_NPZ_CACHE_SPLITS="${PTM_NPZ_CACHE_SPLITS:-train}"
export PTM_REQUIRED_MEMORY_STRATEGY="${PTM_REQUIRED_MEMORY_STRATEGY:-causal_slots}"

export PTM_INIT_MODE="${PTM_INIT_MODE:-oasis}"
export PTM_MAX_STEPS="${PTM_MAX_STEPS:-30000}"
export PTM_BATCH_SIZE="${PTM_BATCH_SIZE:-8}"
export PTM_NUM_WORKERS="${PTM_NUM_WORKERS:-6}"
export PTM_VAL_BATCH_SIZE="${PTM_VAL_BATCH_SIZE:-1}"
export PTM_VAL_NUM_WORKERS="${PTM_VAL_NUM_WORKERS:-2}"
export PTM_LIMIT_VAL_BATCH="${PTM_LIMIT_VAL_BATCH:-2}"
export PTM_VAL_EVERY_N_STEP="${PTM_VAL_EVERY_N_STEP:-3000}"
export PTM_CKPT_EVERY="${PTM_CKPT_EVERY:-3000}"
export PTM_LOG_VIDEO="${PTM_LOG_VIDEO:-true}"
export PTM_MAX_LOG_VIDEOS="${PTM_MAX_LOG_VIDEOS:-1}"
export PTM_VIDEO_LOG_STAGE="${PTM_VIDEO_LOG_STAGE:-ptm}"
export PTM_VIDEO_CACHE_SIZE="${PTM_VIDEO_CACHE_SIZE:-0}"
export PTM_PRECISION="${PTM_PRECISION:-16-mixed}"
export PTM_NUM_NODES="${NNODES}"

export WORLDMEM_LOCAL_LOSS_LOG="/gfs/space/private/zjc/logs/${RUN}_node${NODE_RANK}_local_loss.log"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --no_python \
  ptm/scripts/train_ptm_main.sh
