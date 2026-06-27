#!/usr/bin/env bash
set -euo pipefail

cd /gfs/space/private/zjc/ptm

SUITE_TS="${PTM_V6_SUITE_TS:-$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${PTM_V6_MAX_STEPS:-15000}"
MAX_STEPS_K="$((MAX_STEPS / 1000))k"
SUITE_LABEL="${PTM_V6_SUITE_LABEL:-ptm_v6_generation_router_${MAX_STEPS_K}_serial_${SUITE_TS}}"
SUITE_ROOT="${PTM_V6_SUITE_ROOT:-/gfs/space/private/zjc/ptm/outputs/${SUITE_LABEL}}"
LOG_DIR="${PTM_V6_LOG_DIR:-/gfs/space/private/zjc/logs}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
SHARD_INDICES_DIR="${PTM_V6_EXTERNAL_SHARD_INDICES_DIR:-/gfs/space/private/zjc/ptm/outputs/external_worldmem_ptmfree_10k_normal_20260626_worldmem16_eval/shard_indices}"
WORLDMEM16_CACHE="${PTM_V6_EXTERNAL_NPZ_CACHE_DIR:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/worldmem_minecraft_test_gen600x100_npz_cache_}"

mkdir -p "${SUITE_ROOT}" "${LOG_DIR}"

summarize_eval() {
  local summary="$1"
  if [[ ! -f "${summary}" ]]; then
    echo "[suite] missing summary ${summary}"
    return
  fi
  PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}" python - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

summary = Path(sys.argv[1])
data = json.loads(summary.read_text(encoding="utf-8"))
print(f"[suite] summary={summary}")
for mode in ("normal", "zero_token", "shuffle_token"):
    item = data.get(mode, data if mode == "normal" else None)
    if not isinstance(item, dict):
        continue
    for scope in ("overall", "target_window", "late_horizon"):
        metrics = item.get(scope, item if scope == "overall" else None)
        if isinstance(metrics, dict) and "psnr" in metrics:
            print(
                f"[suite] {mode}/{scope}: "
                f"psnr={metrics.get('psnr'):.6f} "
                f"mse={metrics.get('mse'):.8f} "
                f"lpips={metrics.get('lpips'):.6f}"
            )
PY
}

run_external_eval() {
  local variant="$1"
  local run_name="$2"
  local ckpt="$3"
  local prior_alpha="$4"
  local eval_root="/gfs/space/private/zjc/ptm/outputs/external_worldmem_${run_name}_eval"
  local eval_log="${LOG_DIR}/${run_name}_external_worldmem16_eval.log"

  echo "[suite] external eval variant=${variant} ckpt=${ckpt} eval_root=${eval_root}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    export PTM_CKPT="${ckpt}"
    export PTM_EVAL_ROOT="${eval_root}"
    export PTM_NPZ_CACHE_DIR="${WORLDMEM16_CACHE}"
    export PTM_ALLOW_NONFORMAL_NPZ_CACHE=true
    export PTM_SHARD_INDICES_DIR="${SHARD_INDICES_DIR}"
    export PTM_CONTEXT_MEMORY_ONLY=true
    export PTM_RAW_REFERENCE_LENGTH=0
    export PTM_USE_MEMORY_ATTENTION=false
    export PTM_USE_MEMORY_ATTENTION_RUNTIME=false
    export PTM_USE_PTM_REFERENCE_ADAPTER=false
    export PTM_USE_PTM_CROSS_ATTENTION=true
    export PTM_VISUAL_MEMORY_SELECTION=true
    export PTM_VISUAL_ROUTING_MODE=slot_router
    export PTM_VISUAL_ROUTE_PRIOR_ALPHA="${prior_alpha}"
    export PTM_VISUAL_ROUTE_TOP_M="${PTM_VISUAL_ROUTE_TOP_M:-8}"
    export PTM_VISUAL_ROUTE_TAU="${PTM_VISUAL_ROUTE_TAU:-0.2}"
    export PTM_VISUAL_ROUTE_DIM="${PTM_VISUAL_ROUTE_DIM:-0}"
    export PTM_VISUAL_TOP_K="${PTM_VISUAL_TOP_K:-8}"
    export PTM_VISUAL_NUM_CANDIDATES="${PTM_VISUAL_NUM_CANDIDATES:-64}"
    export PTM_VISUAL_POOL="${PTM_VISUAL_POOL:-grid2x2}"
    export PTM_VISUAL_CANDIDATE_SOURCE="${PTM_VISUAL_CANDIDATE_SOURCE:-context_strided}"
    export PTM_VISUAL_INCLUDE_SUMMARY_TOKENS="${PTM_VISUAL_INCLUDE_SUMMARY_TOKENS:-true}"
    export PTM_VISUAL_REMAP_MATCH_LABELS=true
    export PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-64}"
    export PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-64}"
    export PTM_ABLATIONS="${PTM_ABLATIONS:-normal zero_token shuffle_token}"
    export PTM_NUM_SHARDS="${PTM_NUM_SHARDS:-8}"
    export PTM_GENERATION_BATCH_SIZE="${PTM_GENERATION_BATCH_SIZE:-2}"
    export PTM_GENERATION_LIMIT_BATCH="${PTM_GENERATION_LIMIT_BATCH:-1}"
    export PTM_GENERATION_NUM_WORKERS="${PTM_GENERATION_NUM_WORKERS:-0}"
    bash ptm/scripts/run_generation_ablation_clean.sh
  ) 2>&1 | tee "${eval_log}"
  summarize_eval "${eval_root}/generation_summary.json"
}

run_variant() {
  local variant="$1"
  local prior_name="$2"
  local prior_alpha="$3"
  local run_name="ptm_${variant}_generation_router_${prior_name}_oasis_${MAX_STEPS_K}_${SUITE_TS}"
  local output_dir="/gfs/space/private/zjc/ptm/outputs/${run_name}"
  local train_log="${LOG_DIR}/${run_name}.log"

  echo "[suite] train variant=${variant} prior_alpha=${prior_alpha} run=${run_name}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
    export RUN_NAME="${run_name}"
    export OUTPUT_DIR="${output_dir}"
    export PTM_MAX_STEPS="${MAX_STEPS}"
    export PTM_VISUAL_ROUTE_PRIOR_ALPHA="${prior_alpha}"
    export PTM_VISUAL_ROUTE_TOP_M="${PTM_VISUAL_ROUTE_TOP_M:-8}"
    export PTM_VISUAL_ROUTE_TAU="${PTM_VISUAL_ROUTE_TAU:-0.2}"
    export PTM_VISUAL_ROUTE_DIM="${PTM_VISUAL_ROUTE_DIM:-0}"
    export PTM_VISUAL_TOP_K="${PTM_VISUAL_TOP_K:-8}"
    export PTM_VISUAL_NUM_CANDIDATES="${PTM_VISUAL_NUM_CANDIDATES:-64}"
    export PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-64}"
    export PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-64}"
    export PTM_TRAIN_CONSUMER_ONLY=false
    export PTM_DETACH_FOR_GENERATION=false
    export PTM_INIT_MODE=oasis
    export PTM_BASE_CKPT=""
    export PTM_REQUIRE_BASE_CKPT=false
    bash ptm/scripts/train_ptm_v6_generation_router.sh
  ) 2>&1 | tee "${train_log}"

  local ckpt
  ckpt="$(ls -1t "${output_dir}/checkpoints/"*step"${MAX_STEPS}".ckpt | head -n 1)"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "[suite] missing ${MAX_STEPS} checkpoint under ${output_dir}/checkpoints" >&2
    exit 2
  fi
  echo "[suite] checkpoint=${ckpt}"
  run_external_eval "${variant}" "${run_name}" "${ckpt}" "${prior_alpha}"
}

echo "[suite] label=${SUITE_LABEL}"
echo "[suite] root=${SUITE_ROOT}"
echo "[suite] max_steps=${MAX_STEPS} gpu_list=${GPU_LIST}"
echo "[suite] shard_indices=${SHARD_INDICES_DIR}"
echo "[suite] worldmem16_cache=${WORLDMEM16_CACHE}"

run_variant "v6a" "prior" "1.0"
run_variant "v6b" "no_prior" "0.0"

touch "${SUITE_ROOT}/DONE"
echo "[suite] done"
