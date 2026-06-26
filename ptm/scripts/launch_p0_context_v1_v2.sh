#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

run_eval() {
  local label="$1"
  local ckpt="$2"
  local gpus="$3"
  local log="/gfs/space/private/zjc/logs/${label}.log"

  nohup bash -lc "
    cd /gfs/space/private/zjc/ptm
    export PATH=/gfs/space/private/zjc/envs/worldmem/bin:\$PATH
    export CUDA_VISIBLE_DEVICES=${gpus}
    export PTM_CKPT=${ckpt}
    export PTM_EVAL_LABEL=${label}
    export PTM_NUM_SHARDS=4
    export PTM_GENERATION_BATCH_SIZE=4
    export PTM_GENERATION_LIMIT_BATCH=1
    export PTM_ABLATIONS='normal zero_token shuffle_token'
    bash ptm/scripts/run_generation_context_memory_p0.sh
  " > "${log}" 2>&1 &

  echo "${label} pid=$! log=${log} out=/gfs/space/private/zjc/ptm/outputs/${label}"
}

run_eval \
  p0_context_v1_full_15k_20260626_1400 \
  /gfs/space/private/zjc/ptm/outputs/ptm_v1_full_contrast_detach_15k/checkpoints/epoch0_step15000.ckpt \
  0,1,2,3

run_eval \
  p0_context_v2_no_contrast_15k_20260626_1400 \
  /gfs/space/private/zjc/ptm/outputs/ptm_v2_no_contrast_15k/checkpoints/epoch0_step15000.ckpt \
  4,5,6,7
