#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_v1_full_contrast_detach_no_memattn_15k_20260626_105120/checkpoints/epoch0_step10000.ckpt"
export PTM_EVAL_LABEL="ptm_v1_no_memattn_10k_token_ablation_4x4_20260626_1428"
export PTM_NUM_SHARDS=4
export PTM_GENERATION_BATCH_SIZE=4
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

echo "[validation] start no-memattn V1 10k token ablation 4x4 $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[validation] ckpt=${PTM_CKPT}"
echo "[validation] cuda=${CUDA_VISIBLE_DEVICES}"
bash ptm/scripts/run_generation_context_memory_p0.sh
echo "[validation] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
