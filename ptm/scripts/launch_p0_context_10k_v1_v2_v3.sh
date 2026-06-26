#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

stamp="${PTM_P0_STAMP:-$(date +%Y%m%d_%H%M)}"

run_eval_blocking() {
  local label="$1"
  local ckpt="$2"
  local gpus="$3"
  local log="/gfs/space/private/zjc/logs/${label}.log"

  {
    echo "[launcher] start label=${label} ckpt=${ckpt} gpus=${gpus} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    export PATH=/gfs/space/private/zjc/envs/worldmem/bin:${PATH}
    export CUDA_VISIBLE_DEVICES="${gpus}"
    export PTM_CKPT="${ckpt}"
    export PTM_EVAL_LABEL="${label}"
    export PTM_NUM_SHARDS=4
    export PTM_GENERATION_BATCH_SIZE=4
    export PTM_GENERATION_LIMIT_BATCH=1
    export PTM_ABLATIONS="normal zero_token shuffle_token"
    bash ptm/scripts/run_generation_context_memory_p0.sh
    echo "[launcher] done label=${label} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${log}" 2>&1
}

v1_label="p0_context_v1_full_10k_${stamp}"
v2_label="p0_context_v2_no_contrast_10k_${stamp}"
v3_label="p0_context_v3_encoder_coupled_10k_${stamp}"

echo "[launcher] clean P0 10k start stamp=${stamp}"
echo "[launcher] V1 log=/gfs/space/private/zjc/logs/${v1_label}.log out=/gfs/space/private/zjc/ptm/outputs/${v1_label}"
echo "[launcher] V2 log=/gfs/space/private/zjc/logs/${v2_label}.log out=/gfs/space/private/zjc/ptm/outputs/${v2_label}"
echo "[launcher] V3 log=/gfs/space/private/zjc/logs/${v3_label}.log out=/gfs/space/private/zjc/ptm/outputs/${v3_label}"

run_eval_blocking \
  "${v1_label}" \
  /gfs/space/private/zjc/ptm/outputs/ptm_v1_full_contrast_detach_15k/checkpoints/epoch0_step10000.ckpt \
  0,1,2,3 &
v1_pid=$!

run_eval_blocking \
  "${v2_label}" \
  /gfs/space/private/zjc/ptm/outputs/ptm_v2_no_contrast_15k/checkpoints/epoch0_step10000.ckpt \
  4,5,6,7 &
v2_pid=$!

wait "${v1_pid}"
wait "${v2_pid}"

run_eval_blocking \
  "${v3_label}" \
  /gfs/space/private/zjc/ptm/outputs/ptm_v3_encoder_coupled_15k/checkpoints/epoch0_step10000.ckpt \
  0,1,2,3

echo "[launcher] clean P0 10k all done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
