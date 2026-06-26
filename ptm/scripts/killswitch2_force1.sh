#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd /gfs/space/private/zjc/ptm
export PTM_CKPT=/gfs/space/private/zjc/ptm/outputs/ptm_gate_on_targetloss_10k/checkpoints/epoch0_step10000.ckpt
export PTM_EVAL_LABEL=killswitch2_force1_gate_on_10000
export PTM_NUM_SHARDS=4
export PTM_GPU_LIST=0,1,2,3
export PTM_GENERATION_BATCH_SIZE=2
export PTM_GENERATION_LIMIT_BATCH=1
export PTM_GENERATION_NUM_WORKERS=0
export PTM_NPZ_CACHE_SPLIT=test
export PTM_MEMORY_CONDITION_LENGTH=8
export PTM_MAX_HISTORY=16
export PTM_MAX_HISTORY_CANDIDATES=16
export PTM_ABLATIONS="normal zero hard_shuffle"
export PTM_FORCE_GATE=1.0
bash ptm/scripts/run_generation_ablation_clean.sh
