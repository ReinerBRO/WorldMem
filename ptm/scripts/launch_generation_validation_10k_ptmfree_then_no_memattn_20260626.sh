#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

echo "[validation_queue] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[validation_queue] cuda=${CUDA_VISIBLE_DEVICES}"

(
  export PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_free_generation_baseline_15k_20260626_131030/checkpoints/epoch0_step10000.ckpt"
  export PTM_EVAL_LABEL="ptm_free_generation_baseline_10k_normal_20260626_1418"
  export PTM_NUM_SHARDS=8
  export PTM_GENERATION_BATCH_SIZE=2
  export PTM_GENERATION_LIMIT_BATCH=1
  export PTM_ABLATIONS="normal"
  export PTM_VAL_ABLATION_MODES="normal"
  export PTM_MEMORY_CONDITION_LENGTH=0
  export PTM_RAW_REFERENCE_LENGTH=0
  export PTM_CONTEXT_MEMORY_ONLY=false
  export PTM_USE_PTM_MEMORY=false
  export PTM_USE_PTM_CROSS_ATTENTION=false
  export PTM_USE_MEMORY_ATTENTION=false
  export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
  export PTM_USE_PTM_REFERENCE_ADAPTER=false
  echo "[validation_queue] run ptm-free 10k normal $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash ptm/scripts/run_generation_ablation_clean.sh
)

(
  export PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_v1_full_contrast_detach_no_memattn_15k_20260626_105120/checkpoints/epoch0_step10000.ckpt"
  export PTM_EVAL_LABEL="ptm_v1_no_memattn_10k_token_ablation_20260626_1418"
  export PTM_NUM_SHARDS=8
  export PTM_GENERATION_BATCH_SIZE=2
  export PTM_GENERATION_LIMIT_BATCH=1
  export PTM_ABLATIONS="normal zero_token shuffle_token"
  export PTM_VAL_ABLATION_MODES="normal zero_token shuffle_token"
  export PTM_MEMORY_CONDITION_LENGTH=8
  export PTM_CONTEXT_MEMORY_ONLY=true
  export PTM_RAW_REFERENCE_LENGTH=0
  export PTM_MAX_HISTORY=16
  export PTM_MAX_HISTORY_CANDIDATES=16
  export PTM_USE_PTM_MEMORY=true
  export PTM_USE_PTM_CROSS_ATTENTION=true
  export PTM_USE_MEMORY_ATTENTION=false
  export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
  export PTM_USE_PTM_REFERENCE_ADAPTER=false
  echo "[validation_queue] run no-memattn V1 10k token ablation $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  bash ptm/scripts/run_generation_context_memory_p0.sh
)

echo "[validation_queue] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
