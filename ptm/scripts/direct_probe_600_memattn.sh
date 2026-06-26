#!/bin/bash
set -e
PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_gate_on_memattn_targetloss_10k/checkpoints/epoch0_step10000.ckpt"
PTM_EVAL_LABEL="direct_probe_600_memattn_10000"
PTM_NUM_SHARDS=4
PTM_GPU_LIST=0,1,2,3
PTM_NPZ_CACHE_SPLIT="test"
PTM_DIRECT_BATCH_SIZE=4
PTM_MAX_HISTORY=16
PTM_MAX_HISTORY_CANDIDATES=16
PTM_USE_PTM_CROSS_ATTENTION="true"
PTM_NPZ_CACHE_DIR="/gfs/space/private/zjc/ptm/ptm_minedojo_data/gen600x100_npz_cache"
PTM_CONTEXT_LENGTH=600
PTM_FUTURE_LENGTH=100
PTM_WINDOW_CENTERS="target"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PTM_CKPT
export PTM_EVAL_LABEL
export PTM_NUM_SHARDS
export PTM_GPU_LIST
export PTM_NPZ_CACHE_SPLIT
export PTM_DIRECT_BATCH_SIZE=4
export PTM_USE_PTM_CROSS_ATTENTION
export PTM_MAX_HISTORY
export PTM_MAX_HISTORY_CANDIDATES
export PTM_NPZ_CACHE_DIR
export PTM_CONTEXT_LENGTH
export PTM_FUTURE_LENGTH
export PTM_WINDOW_CENTERS
cd /gfs/space/private/zjc/ptm
bash /gfs/space/private/zjc/ptm/ptm/scripts/run_direct_ablation_clean.sh
