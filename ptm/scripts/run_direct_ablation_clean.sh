#!/usr/bin/env bash
set -euo pipefail

: "${PTM_CKPT:?set PTM_CKPT}"
PTM_FORMAL_NPZ_CACHE_DIR="/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache"
PTM_NPZ_CACHE_DIR="${PTM_NPZ_CACHE_DIR:-${PTM_FORMAL_NPZ_CACHE_DIR}}"

PTM_EVAL_LABEL="${PTM_EVAL_LABEL:-direct_ablation_clean_$(date +%Y%m%d_%H%M%S)}"
PTM_EVAL_ROOT="${PTM_EVAL_ROOT:-/gfs/space/private/zjc/ptm/outputs/${PTM_EVAL_LABEL}}"
PTM_DATA_ROOT="${PTM_DATA_ROOT:-ptm_minedojo_data/long_1500_360x640}"
PTM_NUM_SHARDS="${PTM_NUM_SHARDS:-8}"
PTM_DIRECT_BATCH_SIZE="${PTM_DIRECT_BATCH_SIZE:-4}"
PTM_DIRECT_LIMIT_BATCH="${PTM_DIRECT_LIMIT_BATCH:-1}"
PTM_DIRECT_NUM_WORKERS="${PTM_DIRECT_NUM_WORKERS:-0}"
PTM_NPZ_CACHE_SPLIT="${PTM_NPZ_CACHE_SPLIT:-test}"
PTM_ABLATIONS="${PTM_ABLATIONS:-normal zero hard_shuffle}"
PTM_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PTM_CACHE_ROOT="${PTM_CACHE_ROOT:-/gfs/space/private/zjc/.cache}"
PTM_CONTEXT_LENGTH="${PTM_CONTEXT_LENGTH:-4}"
PTM_FUTURE_LENGTH="${PTM_FUTURE_LENGTH:-4}"
PTM_N_FRAMES="${PTM_N_FRAMES:-$((PTM_CONTEXT_LENGTH + PTM_FUTURE_LENGTH))}"
PTM_VAL_CONTEXT_LENGTH="${PTM_VAL_CONTEXT_LENGTH:-600}"
PTM_VAL_FUTURE_LENGTH="${PTM_VAL_FUTURE_LENGTH:-100}"
PTM_N_FRAMES_VALID="${PTM_N_FRAMES_VALID:-$((PTM_VAL_CONTEXT_LENGTH + PTM_VAL_FUTURE_LENGTH))}"
PTM_MEMORY_CONDITION_LENGTH="${PTM_MEMORY_CONDITION_LENGTH:-8}"
PTM_FRAME_HEIGHT="${PTM_FRAME_HEIGHT:-360}"
PTM_FRAME_WIDTH="${PTM_FRAME_WIDTH:-640}"
PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-$((PTM_N_FRAMES_VALID + PTM_MEMORY_CONDITION_LENGTH))}"
PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-${PTM_MAX_HISTORY}}"
PTM_WINDOW_CENTERS="${PTM_WINDOW_CENTERS:-target,late50,late75,late100}"
PTM_USE_PTM_CROSS_ATTENTION="${PTM_USE_PTM_CROSS_ATTENTION:-true}"

if [[ "${PTM_DIRECT_LIMIT_BATCH}" != "1" ]]; then
  echo "clean direct ablation requires PTM_DIRECT_LIMIT_BATCH=1" >&2
  exit 1
fi

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PTM_CKPT
export PTM_DATA_ROOT
export PTM_NPZ_CACHE_SPLITS=all
export PTM_EVAL_TASK="${PTM_EVAL_TASK:-test}"
export PTM_TEST_BATCH_SIZE="${PTM_DIRECT_BATCH_SIZE}"
export PTM_LIMIT_TEST_BATCH="${PTM_DIRECT_LIMIT_BATCH}"
export PTM_TEST_NUM_WORKERS="${PTM_DIRECT_NUM_WORKERS}"
export PTM_TEST_SHUFFLE=false
export PTM_REQUIRED_MEMORY_STRATEGY="${PTM_REQUIRED_MEMORY_STRATEGY:-causal_slots}"
export PTM_CACHE_ROOT
export PTM_LOG_VIDEO=false

IFS=',' read -r -a GPUS <<< "${PTM_GPU_LIST}"
if (( ${#GPUS[@]} < PTM_NUM_SHARDS )); then
  echo "PTM_NUM_SHARDS=${PTM_NUM_SHARDS} exceeds CUDA_VISIBLE_DEVICES entries (${PTM_GPU_LIST})" >&2
  exit 1
fi

mkdir -p "${PTM_EVAL_ROOT}/logs" "${PTM_EVAL_ROOT}/shard_indices"
PTM_SUITE_ROOT="${PTM_EVAL_ROOT}"

python - \
  "${PTM_NPZ_CACHE_DIR}" \
  "${PTM_NPZ_CACHE_SPLIT}" \
  "${PTM_EVAL_ROOT}/shard_indices" \
  "${PTM_NUM_SHARDS}" \
  "${PTM_DIRECT_BATCH_SIZE}" \
  "${PTM_REQUIRED_MEMORY_STRATEGY}" \
  "${PTM_CONTEXT_LENGTH}" \
  "${PTM_FUTURE_LENGTH}" \
  "${PTM_MEMORY_CONDITION_LENGTH}" \
  "${PTM_WINDOW_CENTERS}" <<'PY'
import json
import random
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1])
split = sys.argv[2]
out_dir = Path(sys.argv[3])
num_shards = int(sys.argv[4])
batch_size = int(sys.argv[5])
required_strategy = sys.argv[6]
expected_context = int(sys.argv[7])
expected_future = int(sys.argv[8])
expected_memory = int(sys.argv[9])
expected_centers = [part.strip() for part in sys.argv[10].split(",") if part.strip()]
expected = num_shards * batch_size

manifest_path = cache_dir / "manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing NPZ cache manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
strategy = manifest.get("memory_strategy")
if strategy != required_strategy:
    raise SystemExit(f"invalid PTM cache memory_strategy={strategy!r}; expected {required_strategy!r}")
if int(manifest.get("context_length", -1)) != expected_context:
    raise SystemExit(f"direct cache must be context_length={expected_context}; got {manifest.get('context_length')!r}")
if int(manifest.get("future_length", -1)) != expected_future:
    raise SystemExit(f"direct cache must be future_length={expected_future}; got {manifest.get('future_length')!r}")
if int(manifest.get("memory_condition_length", -1)) != expected_memory:
    raise SystemExit(
        f"direct cache must have memory_condition_length={expected_memory}; "
        f"got {manifest.get('memory_condition_length')!r}"
    )
if manifest.get("window_centers") != expected_centers:
    raise SystemExit(f"direct cache window_centers={manifest.get('window_centers')!r}; expected {expected_centers!r}")

index_path = cache_dir / split / "index.jsonl"
if not index_path.exists():
    raise SystemExit(f"missing NPZ cache index: {index_path}")
entries = []
with index_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            entries.append(json.loads(line))
if len(entries) < expected:
    raise SystemExit(f"not enough cached samples for clean direct ablation: {len(entries)} < {expected}")

def episode(index: int) -> str:
    return str(entries[index].get("episode_dir", ""))

def family(index: int) -> str:
    return str(entries[index].get("episode_family", ""))

def batch_valid(indices: list[int]) -> bool:
    return len({episode(index) for index in indices}) >= 2

rng = random.Random(20260624)
all_indices = list(range(len(entries)))
groups = None
for _attempt in range(20000):
    shuffled = all_indices[:]
    rng.shuffle(shuffled)
    selected = shuffled[:expected]
    candidate = [selected[i * batch_size : (i + 1) * batch_size] for i in range(num_shards)]
    if all(batch_valid(group) for group in candidate):
        groups = candidate
        break

if groups is None:
    raise SystemExit("could not build 8 clean hard-shuffle shard batches with >=2 episodes each")

out_dir.mkdir(parents=True, exist_ok=True)
plan = []
for shard, group in enumerate(groups):
    path = out_dir / f"shard{shard}.txt"
    path.write_text("\n".join(str(index) for index in group) + "\n", encoding="utf-8")
    plan.append(
        {
            "shard": shard,
            "indices": group,
            "episode_dirs": [episode(index) for index in group],
            "episode_families": [family(index) for index in group],
        }
    )

plan_path = out_dir / "plan.json"
plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"plan": str(plan_path), "samples": expected, "shards": num_shards}, sort_keys=True), flush=True)
PY

run_mode() {
  local mode="$1"
  local mode_tmp="${PTM_EVAL_ROOT}/.tmp_${mode}_$$"
  local mode_dir="${PTM_EVAL_ROOT}/${mode}"
  rm -rf "${mode_tmp}" "${mode_dir}"
  mkdir -p "${mode_tmp}"

  local pids=()
  for (( shard=0; shard<PTM_NUM_SHARDS; shard++ )); do
    local gpu="${GPUS[$shard]}"
    local seed=$((20260624 + shard))
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export PTM_SHARD_INDEX="${shard}"
      mode_output_dir="${mode_tmp}/${mode}/shard${shard}"
      mkdir -p "${mode_output_dir}"
      export PTM_EVAL_OUTPUT_DIR="${mode_output_dir}"
      python main.py \
        +name="ptm_ablation_${mode}" \
        +seed="${seed}" \
        dataset=ptm_minedojo \
        dataset.save_dir="${PTM_DATA_ROOT}" \
        dataset.resolution="[${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]" \
        dataset.observation_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]" \
        dataset.n_frames="${PTM_N_FRAMES}" \
        +dataset.n_frames_valid="${PTM_N_FRAMES_VALID}" \
        dataset.context_length="${PTM_CONTEXT_LENGTH}" \
        dataset.future_length="${PTM_FUTURE_LENGTH}" \
        dataset.ptm_context_length="${PTM_CONTEXT_LENGTH}" \
        dataset.ptm_future_length="${PTM_FUTURE_LENGTH}" \
        +dataset.ptm_context_length_valid="${PTM_VAL_CONTEXT_LENGTH}" \
        +dataset.ptm_future_length_valid="${PTM_VAL_FUTURE_LENGTH}" \
        dataset.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}" \
        dataset.max_history_candidates="${PTM_MAX_HISTORY_CANDIDATES}" \
        +dataset.video_cache_size=0 \
        +dataset.npz_cache_dir="${PTM_NPZ_CACHE_DIR}" \
        +dataset.npz_cache_splits="[all]" \
        +dataset.npz_cache_indices_file="${PTM_SUITE_ROOT}/shard_indices/shard${shard}.txt" \
        load="${PTM_CKPT}" \
        +customized_load=true \
        experiment.tasks="[${PTM_EVAL_TASK}]" \
        experiment.test.batch_size="${PTM_DIRECT_BATCH_SIZE}" \
        experiment.test.limit_batch="${PTM_DIRECT_LIMIT_BATCH}" \
        experiment.test.data.num_workers="${PTM_DIRECT_NUM_WORKERS}" \
        experiment.test.data.shuffle=false \
        experiment.validation.batch_size="${PTM_DIRECT_BATCH_SIZE}" \
        experiment.validation.limit_batch="${PTM_DIRECT_LIMIT_BATCH}" \
        experiment.validation.data.num_workers="${PTM_DIRECT_NUM_WORKERS}" \
        experiment.validation.data.shuffle=false \
        algorithm.x_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]" \
        algorithm.context_frames="${PTM_VAL_CONTEXT_LENGTH}" \
        ++algorithm.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}" \
        ++algorithm.use_memory_attention=false \
        ++algorithm.use_ptm_memory=true \
        ++algorithm.use_ptm_reference_adapter=true \
        ++algorithm.use_ptm_cross_attention="${PTM_USE_PTM_CROSS_ATTENTION}" \
        ++algorithm.ptm_max_history="${PTM_MAX_HISTORY}" \
        ++algorithm.ptm_max_history_candidates="${PTM_MAX_HISTORY_CANDIDATES}" \
        ++algorithm.log_video=false \
        ++algorithm.video_log_stage=ptm \
        ++algorithm.ptm_ablation="${mode}" \
        ++algorithm.ptm_eval_only=true \
        ++algorithm.local_save_dir="${mode_output_dir}" \
        +output_dir="${mode_output_dir}" \
        wandb.mode=disabled
    ) > "${PTM_EVAL_ROOT}/logs/${mode}_shard${shard}_gpu${gpu}.log" 2>&1 &
    local pid="$!"
    pids+=("${pid}")
    echo "[direct:${mode}] shard=${shard} gpu=${gpu} pid=${pid} indices=${PTM_EVAL_ROOT}/shard_indices/shard${shard}.txt"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    rm -rf "${mode_tmp}"
    echo "[direct:${mode}] failed; removed temporary output" >&2
    return 1
  fi
  if [[ ! -d "${mode_tmp}/${mode}" ]]; then
    rm -rf "${mode_tmp}"
    echo "[direct:${mode}] missing completed mode directory" >&2
    return 1
  fi
  mv "${mode_tmp}/${mode}" "${mode_dir}"
  rm -rf "${mode_tmp}"
  echo "[direct:${mode}] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

echo "[direct] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[direct] ckpt=${PTM_CKPT}"
echo "[direct] root=${PTM_EVAL_ROOT}"
echo "[direct] cache=${PTM_NPZ_CACHE_DIR}"
echo "[direct] gpu_list=${PTM_GPU_LIST}"
echo "[direct] shards=${PTM_NUM_SHARDS} batch=${PTM_DIRECT_BATCH_SIZE} workers=${PTM_DIRECT_NUM_WORKERS}"

for mode in ${PTM_ABLATIONS}; do
  run_mode "${mode}"
done

summary_tmp="${PTM_EVAL_ROOT}/direct_summary.json.tmp"
python ptm/scripts/aggregate_ablation_outputs.py "${PTM_EVAL_ROOT}" --kind direct --output "${summary_tmp}"
python - "${summary_tmp}" "${PTM_NUM_SHARDS}" "${PTM_DIRECT_BATCH_SIZE}" "${PTM_ABLATIONS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2]) * int(sys.argv[3])
required = tuple(sys.argv[4].split())
payload = json.loads(path.read_text(encoding="utf-8"))
missing = [mode for mode in required if mode not in payload]
if missing:
    raise SystemExit(f"missing direct ablation modes: {missing}")
bad_counts = {
    mode: data.get("num_samples")
    for mode, data in payload.items()
    if mode in required and int(data.get("num_samples", -1)) != expected
}
if bad_counts:
    raise SystemExit(f"unexpected direct sample counts: {bad_counts}; expected {expected}")
print(json.dumps({"summary_ok": str(path), "samples_per_mode": expected}, sort_keys=True), flush=True)
PY
mv "${summary_tmp}" "${PTM_EVAL_ROOT}/direct_summary.json"
echo "[direct] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
