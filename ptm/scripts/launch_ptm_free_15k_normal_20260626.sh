#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_free_generation_baseline_15k_20260626_131030/checkpoints/epoch0_step15000.ckpt"
export PTM_EVAL_LABEL="ptm_free_generation_baseline_15k_normal_20260626_1438"
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

echo "[validation] start PTM-free 15k normal $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[validation] ckpt=${PTM_CKPT}"
echo "[validation] cuda=${CUDA_VISIBLE_DEVICES}"
bash ptm/scripts/run_generation_ablation_clean.sh
echo "[validation] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
