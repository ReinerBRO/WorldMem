#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XDG_CACHE_HOME="/gfs/space/private/zjc/.cache"
export TORCH_HOME="/gfs/space/private/zjc/.cache/torch"
export MPLCONFIGDIR="/gfs/space/private/zjc/.cache/matplotlib"
mkdir -p "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${MPLCONFIGDIR}"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"

export PTM_NUM_SHARDS=8
export PTM_GENERATION_BATCH_SIZE=2
export PTM_GENERATION_LIMIT_BATCH=1
export PTM_GENERATION_NUM_WORKERS=0
export PTM_NPZ_CACHE_SPLIT=test
export PTM_ABLATIONS=normal
export PTM_VAL_ABLATION_MODES=normal
export PTM_USE_MEMORY_ATTENTION=false
export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
export PTM_USE_PTM_REFERENCE_ADAPTER=false

echo "[external_matched] start $(date -u +%Y-%m-%dT%H:%M:%SZ) ts=${TS}"
echo "[external_matched] cuda=${CUDA_VISIBLE_DEVICES}"

export PTM_EVAL_LABEL="external_matched_ptmfree_10k_normal_${TS}"
export PTM_EVAL_ROOT="/gfs/space/private/zjc/ptm/outputs/${PTM_EVAL_LABEL}"
export PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_free_generation_baseline_15k_20260626_131030/checkpoints/epoch0_step10000.ckpt"
export PTM_MEMORY_CONDITION_LENGTH=0
export PTM_RAW_REFERENCE_LENGTH=0
export PTM_CONTEXT_MEMORY_ONLY=false
export PTM_USE_PTM_MEMORY=false
export PTM_USE_PTM_CROSS_ATTENTION=false

echo "[external_matched] run ptmfree label=${PTM_EVAL_LABEL}"
bash ptm/scripts/run_generation_ablation_clean.sh
PTMFREE_ROOT="${PTM_EVAL_ROOT}"

export PTM_EVAL_LABEL="external_matched_v4a_oasis_10k_normal_${TS}"
export PTM_EVAL_ROOT="/gfs/space/private/zjc/ptm/outputs/${PTM_EVAL_LABEL}"
export PTM_CKPT="/gfs/space/private/zjc/ptm/outputs/ptm_v4a_causal_slot_token_only_oasis_10k_20260626_203311/checkpoints/epoch1_step10000.ckpt"
export PTM_MEMORY_CONDITION_LENGTH=8
export PTM_RAW_REFERENCE_LENGTH=0
export PTM_CONTEXT_MEMORY_ONLY=true
export PTM_CONTEXT_MEMORY_STRATEGY=strided
export PTM_MAX_HISTORY=16
export PTM_MAX_HISTORY_CANDIDATES=16
export PTM_USE_PTM_MEMORY=true
export PTM_USE_PTM_CROSS_ATTENTION=true

echo "[external_matched] run v4a label=${PTM_EVAL_LABEL}"
bash ptm/scripts/run_generation_context_memory_p0.sh
V4A_ROOT="${PTM_EVAL_ROOT}"

echo "[external_matched] compare plans"
diff -q "${PTMFREE_ROOT}/shard_indices/plan.json" "${V4A_ROOT}/shard_indices/plan.json"

python3 - "${PTMFREE_ROOT}" "${V4A_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    payload = json.loads((root / "generation_summary.json").read_text(encoding="utf-8"))
    data = payload.get("normal", payload)
    print(str(root))
    print(json.dumps(data, indent=2, sort_keys=True)[:2000])
PY

echo "[external_matched] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
