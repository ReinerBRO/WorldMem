#!/usr/bin/env bash
set -euo pipefail

: "${PTM_CKPT:?set PTM_CKPT}"
PTM_FORMAL_NPZ_CACHE_DIR="/gfs/space/private/zjc/ptm/ptm_minedojo_data/gen600x100_npz_cache"
PTM_NPZ_CACHE_DIR="${PTM_NPZ_CACHE_DIR:-${PTM_FORMAL_NPZ_CACHE_DIR}}"
PTM_ALLOW_NONFORMAL_NPZ_CACHE="${PTM_ALLOW_NONFORMAL_NPZ_CACHE:-false}"
if [[ "${PTM_NPZ_CACHE_DIR}" != "${PTM_FORMAL_NPZ_CACHE_DIR}" && "${PTM_ALLOW_NONFORMAL_NPZ_CACHE}" != "true" ]]; then
  echo "refusing non-formal PTM_NPZ_CACHE_DIR=${PTM_NPZ_CACHE_DIR}; expected ${PTM_FORMAL_NPZ_CACHE_DIR}" >&2
  echo "set PTM_ALLOW_NONFORMAL_NPZ_CACHE=true only for explicit external-dataset diagnostics" >&2
  exit 2
fi

PTM_EVAL_LABEL="${PTM_EVAL_LABEL:-generation_ablation_clean_$(date +%Y%m%d_%H%M%S)}"
PTM_EVAL_ROOT="${PTM_EVAL_ROOT:-/gfs/space/private/zjc/ptm/outputs/${PTM_EVAL_LABEL}}"
PTM_DATA_ROOT="${PTM_DATA_ROOT:-ptm_minedojo_data/long_1500_360x640}"
PTM_NUM_SHARDS="${PTM_NUM_SHARDS:-8}"
PTM_GENERATION_BATCH_SIZE="${PTM_GENERATION_BATCH_SIZE:-2}"
PTM_GENERATION_LIMIT_BATCH="${PTM_GENERATION_LIMIT_BATCH:-1}"
PTM_GENERATION_NUM_WORKERS="${PTM_GENERATION_NUM_WORKERS:-0}"
PTM_NPZ_CACHE_SPLIT="${PTM_NPZ_CACHE_SPLIT:-test}"
PTM_SHARD_INDICES_DIR="${PTM_SHARD_INDICES_DIR:-}"
PTM_CONTEXT_MEMORY_ONLY="${PTM_CONTEXT_MEMORY_ONLY:-false}"
PTM_CONTEXT_MEMORY_STRATEGY="${PTM_CONTEXT_MEMORY_STRATEGY:-strided}"
PTM_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PTM_CACHE_ROOT="${PTM_CACHE_ROOT:-/gfs/space/private/zjc/.cache}"
PTM_CONTEXT_LENGTH="${PTM_CONTEXT_LENGTH:-4}"
PTM_FUTURE_LENGTH="${PTM_FUTURE_LENGTH:-4}"
PTM_N_FRAMES="${PTM_N_FRAMES:-$((PTM_CONTEXT_LENGTH + PTM_FUTURE_LENGTH))}"
PTM_VAL_CONTEXT_LENGTH="${PTM_VAL_CONTEXT_LENGTH:-600}"
PTM_VAL_FUTURE_LENGTH="${PTM_VAL_FUTURE_LENGTH:-100}"
PTM_N_FRAMES_VALID="${PTM_N_FRAMES_VALID:-$((PTM_VAL_CONTEXT_LENGTH + PTM_VAL_FUTURE_LENGTH))}"
PTM_MEMORY_CONDITION_LENGTH="${PTM_MEMORY_CONDITION_LENGTH:-8}"
if [[ "${PTM_CONTEXT_MEMORY_ONLY}" == "true" ]]; then
  PTM_RAW_REFERENCE_LENGTH="${PTM_RAW_REFERENCE_LENGTH:-0}"
  PTM_ABLATIONS="${PTM_ABLATIONS:-normal zero_token shuffle_token}"
else
  PTM_RAW_REFERENCE_LENGTH="${PTM_RAW_REFERENCE_LENGTH:-${PTM_MEMORY_CONDITION_LENGTH}}"
  PTM_ABLATIONS="${PTM_ABLATIONS:-normal zero hard_shuffle}"
fi
# validation_ablation_modes mirrors PTM_ABLATIONS so each shard run only does
# the requested modes (supports token-level kill-switch modes).
PTM_VAL_ABLATION_MODES="${PTM_VAL_ABLATION_MODES:-${PTM_ABLATIONS}}"
PTM_FRAME_HEIGHT="${PTM_FRAME_HEIGHT:-360}"
PTM_FRAME_WIDTH="${PTM_FRAME_WIDTH:-640}"
if [[ "${PTM_CONTEXT_MEMORY_ONLY}" == "true" ]]; then
  PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-16}"
else
  PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-$((PTM_N_FRAMES_VALID + PTM_MEMORY_CONDITION_LENGTH))}"
fi
PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-${PTM_MAX_HISTORY}}"
PTM_TARGET_WINDOW_RADIUS="${PTM_TARGET_WINDOW_RADIUS:-5}"
PTM_LATE_HORIZON_START="${PTM_LATE_HORIZON_START:-50}"
PTM_USE_PTM_MEMORY="${PTM_USE_PTM_MEMORY:-true}"
PTM_USE_MEMORY_ATTENTION="${PTM_USE_MEMORY_ATTENTION:-true}"
PTM_USE_MEMORY_ATTENTION_RUNTIME="${PTM_USE_MEMORY_ATTENTION_RUNTIME:-false}"
PTM_USE_PTM_CROSS_ATTENTION="${PTM_USE_PTM_CROSS_ATTENTION:-true}"
PTM_USE_PTM_REFERENCE_ADAPTER="${PTM_USE_PTM_REFERENCE_ADAPTER:-false}"
PTM_VISUAL_MEMORY_SELECTION="${PTM_VISUAL_MEMORY_SELECTION:-false}"
PTM_VISUAL_TOP_K="${PTM_VISUAL_TOP_K:-8}"
PTM_VISUAL_NUM_CANDIDATES="${PTM_VISUAL_NUM_CANDIDATES:-${PTM_MAX_HISTORY_CANDIDATES}}"
PTM_VISUAL_POOL="${PTM_VISUAL_POOL:-grid2x2}"
PTM_VISUAL_CANDIDATE_SOURCE="${PTM_VISUAL_CANDIDATE_SOURCE:-context_strided}"
PTM_VISUAL_INCLUDE_SUMMARY_TOKENS="${PTM_VISUAL_INCLUDE_SUMMARY_TOKENS:-true}"
PTM_VISUAL_REMAP_MATCH_LABELS="${PTM_VISUAL_REMAP_MATCH_LABELS:-true}"

if [[ "${PTM_CONTEXT_MEMORY_ONLY}" == "true" ]]; then
  if [[ "${PTM_RAW_REFERENCE_LENGTH}" != "0" ]]; then
    echo "PTM_CONTEXT_MEMORY_ONLY=true requires PTM_RAW_REFERENCE_LENGTH=0" >&2
    exit 1
  fi
  if [[ "${PTM_USE_MEMORY_ATTENTION_RUNTIME}" != "false" ]]; then
    echo "PTM_CONTEXT_MEMORY_ONLY=true requires PTM_USE_MEMORY_ATTENTION_RUNTIME=false" >&2
    exit 1
  fi
fi

if [[ "${PTM_GENERATION_LIMIT_BATCH}" != "1" ]]; then
  echo "clean generation ablation requires PTM_GENERATION_LIMIT_BATCH=1" >&2
  exit 1
fi
if (( PTM_GENERATION_BATCH_SIZE < 2 )); then
  echo "clean generation ablation requires PTM_GENERATION_BATCH_SIZE >= 2 for hard-shuffle" >&2
  exit 1
fi
case "${PTM_NPZ_CACHE_SPLIT}" in
  val)
    expected_eval_task="validation"
    ;;
  test)
    expected_eval_task="test"
    ;;
  *)
    echo "clean generation ablation only accepts PTM_NPZ_CACHE_SPLIT=val or test; got ${PTM_NPZ_CACHE_SPLIT}" >&2
    exit 1
    ;;
esac
PTM_EVAL_TASK="${PTM_EVAL_TASK:-${expected_eval_task}}"
if [[ "${PTM_EVAL_TASK}" != "${expected_eval_task}" ]]; then
  echo "PTM_EVAL_TASK=${PTM_EVAL_TASK} does not match PTM_NPZ_CACHE_SPLIT=${PTM_NPZ_CACHE_SPLIT}; expected ${expected_eval_task}" >&2
  exit 1
fi

export PATH="/gfs/space/private/zjc/envs/worldmem/bin:${PATH}"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PTM_CKPT
export PTM_DATA_ROOT
export PTM_NPZ_CACHE_SPLITS=all
export PTM_EVAL_TASK
export PTM_TEST_BATCH_SIZE="${PTM_GENERATION_BATCH_SIZE}"
export PTM_LIMIT_TEST_BATCH="${PTM_GENERATION_LIMIT_BATCH}"
export PTM_TEST_NUM_WORKERS="${PTM_GENERATION_NUM_WORKERS}"
export PTM_TEST_SHUFFLE=false
export PTM_REQUIRED_MEMORY_STRATEGY="${PTM_REQUIRED_MEMORY_STRATEGY:-causal_slots}"
export PTM_CACHE_ROOT

IFS=',' read -r -a GPUS <<< "${PTM_GPU_LIST}"
if (( ${#GPUS[@]} < PTM_NUM_SHARDS )); then
  echo "PTM_NUM_SHARDS=${PTM_NUM_SHARDS} exceeds CUDA_VISIBLE_DEVICES entries (${PTM_GPU_LIST})" >&2
  exit 1
fi

mkdir -p "${PTM_EVAL_ROOT}/logs" "${PTM_EVAL_ROOT}/shard_indices"
PTM_SUITE_ROOT="${PTM_EVAL_ROOT}"

if [[ -n "${PTM_SHARD_INDICES_DIR}" ]]; then
  if [[ ! -f "${PTM_SHARD_INDICES_DIR}/plan.json" ]]; then
    echo "PTM_SHARD_INDICES_DIR=${PTM_SHARD_INDICES_DIR} is missing plan.json" >&2
    exit 1
  fi
  for (( shard=0; shard<PTM_NUM_SHARDS; shard++ )); do
    if [[ ! -f "${PTM_SHARD_INDICES_DIR}/shard${shard}.txt" ]]; then
      echo "PTM_SHARD_INDICES_DIR=${PTM_SHARD_INDICES_DIR} is missing shard${shard}.txt" >&2
      exit 1
    fi
  done
  cp "${PTM_SHARD_INDICES_DIR}/plan.json" "${PTM_EVAL_ROOT}/shard_indices/plan.json"
  for (( shard=0; shard<PTM_NUM_SHARDS; shard++ )); do
    cp "${PTM_SHARD_INDICES_DIR}/shard${shard}.txt" "${PTM_EVAL_ROOT}/shard_indices/shard${shard}.txt"
  done
  python - "${PTM_EVAL_ROOT}/shard_indices" "${PTM_NUM_SHARDS}" "${PTM_GENERATION_BATCH_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
num_shards = int(sys.argv[2])
batch_size = int(sys.argv[3])
expected = num_shards * batch_size
indices = []
for shard in range(num_shards):
    path = out_dir / f"shard{shard}.txt"
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != batch_size:
        raise SystemExit(f"{path} has {len(values)} samples; expected {batch_size}")
    indices.extend(values)
if len(indices) != len(set(indices)):
    raise SystemExit("prebuilt shard indices contain duplicate sample indices")
print(json.dumps({"plan": str(out_dir / "plan.json"), "samples": expected, "shards": num_shards}, sort_keys=True), flush=True)
PY
else
  python - \
    "${PTM_NPZ_CACHE_DIR}" \
    "${PTM_NPZ_CACHE_SPLIT}" \
    "${PTM_EVAL_ROOT}/shard_indices" \
    "${PTM_NUM_SHARDS}" \
    "${PTM_GENERATION_BATCH_SIZE}" \
    "${PTM_REQUIRED_MEMORY_STRATEGY}" \
    "${PTM_VAL_CONTEXT_LENGTH}" \
    "${PTM_VAL_FUTURE_LENGTH}" \
    "${PTM_ABLATIONS}" <<'PY'
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
requested_modes = set(sys.argv[9].split())
expected = num_shards * batch_size
requires_cross_episode = bool(requested_modes & {"hard_shuffle", "shuffle", "shuffle_token"})

manifest_path = cache_dir / "manifest.json"
if not manifest_path.exists():
    raise SystemExit(f"missing NPZ cache manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
strategy = manifest.get("memory_strategy")
if required_strategy and strategy is not None and strategy != required_strategy:
    raise SystemExit(f"invalid PTM cache memory_strategy={strategy!r}; expected {required_strategy!r}")
if int(manifest.get("context_length", -1)) != expected_context:
    raise SystemExit(f"generation cache must be context_length={expected_context}; got {manifest.get('context_length')!r}")
if int(manifest.get("future_length", -1)) != expected_future:
    raise SystemExit(f"generation cache must be future_length={expected_future}; got {manifest.get('future_length')!r}")
if int(manifest.get("memory_condition_length", -1)) != 8:
    raise SystemExit(
        "generation cache must have memory_condition_length=8; "
        f"got {manifest.get('memory_condition_length')!r}"
    )
if manifest.get("window_centers") != ["target"]:
    raise SystemExit(f"generation cache must have window_centers=['target']; got {manifest.get('window_centers')!r}")

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
    raise SystemExit(f"not enough cached samples for clean generation ablation: {len(entries)} < {expected}")

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
    if all((not requires_cross_episode) or batch_valid(group) for group in candidate):
        groups = candidate
        break

if groups is None:
    raise SystemExit("could not build clean hard-shuffle shard batches with >=2 episodes each")

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
fi

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
      mode_output_dir="${mode_tmp}/${mode}/shard${shard}"
      mkdir -p "${mode_output_dir}"
      python main.py \
        +name="generation_ablation_${mode}" \
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
        experiment.tasks="[${PTM_EVAL_TASK}]" \
        experiment.test.batch_size="${PTM_GENERATION_BATCH_SIZE}" \
        experiment.test.limit_batch="${PTM_GENERATION_LIMIT_BATCH}" \
        experiment.test.data.num_workers="${PTM_GENERATION_NUM_WORKERS}" \
        experiment.test.data.shuffle=false \
        experiment.validation.batch_size="${PTM_GENERATION_BATCH_SIZE}" \
        experiment.validation.limit_batch="${PTM_GENERATION_LIMIT_BATCH}" \
        experiment.validation.data.num_workers="${PTM_GENERATION_NUM_WORKERS}" \
        experiment.validation.data.shuffle=false \
        algorithm.x_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]" \
        algorithm.context_frames="${PTM_VAL_CONTEXT_LENGTH}" \
        ++algorithm.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}" \
        ++algorithm.raw_reference_length="${PTM_RAW_REFERENCE_LENGTH}" \
        ++algorithm.use_memory_attention="${PTM_USE_MEMORY_ATTENTION}" \
        ++algorithm.use_memory_attention_runtime="${PTM_USE_MEMORY_ATTENTION_RUNTIME}" \
        ++algorithm.use_ptm_memory="${PTM_USE_PTM_MEMORY}" \
        ++algorithm.use_ptm_reference_adapter="${PTM_USE_PTM_REFERENCE_ADAPTER}" \
        ++algorithm.use_ptm_cross_attention="${PTM_USE_PTM_CROSS_ATTENTION}" \
        ++algorithm.ptm_context_memory_only="${PTM_CONTEXT_MEMORY_ONLY}" \
        ++algorithm.ptm_context_memory_strategy="${PTM_CONTEXT_MEMORY_STRATEGY}" \
        ++algorithm.ptm_max_history="${PTM_MAX_HISTORY}" \
        ++algorithm.ptm_max_history_candidates="${PTM_MAX_HISTORY_CANDIDATES}" \
        ++algorithm.ptm_visual_memory_selection="${PTM_VISUAL_MEMORY_SELECTION}" \
        ++algorithm.ptm_visual_top_k="${PTM_VISUAL_TOP_K}" \
        ++algorithm.ptm_visual_num_candidates="${PTM_VISUAL_NUM_CANDIDATES}" \
        ++algorithm.ptm_visual_pool="${PTM_VISUAL_POOL}" \
        ++algorithm.ptm_visual_candidate_source="${PTM_VISUAL_CANDIDATE_SOURCE}" \
        ++algorithm.ptm_visual_include_summary_tokens="${PTM_VISUAL_INCLUDE_SUMMARY_TOKENS}" \
        ++algorithm.ptm_visual_remap_match_labels="${PTM_VISUAL_REMAP_MATCH_LABELS}" \
        ++algorithm.log_video=false \
        ++algorithm.max_log_videos=0 \
        ++algorithm.video_log_stage=ptm \
        ++algorithm.ptm_ablation="${mode}" \
        ++algorithm.validation_ablation_modes="[${mode}]" \
        ++algorithm.ptm_eval_only=false \
        ++algorithm.local_save_dir="${mode_output_dir}" \
        ++algorithm.generation_target_window_radius="${PTM_TARGET_WINDOW_RADIUS}" \
        ++algorithm.generation_late_horizon_start="${PTM_LATE_HORIZON_START}" \
        +output_dir="${mode_output_dir}" \
        wandb.mode=disabled
    ) > "${PTM_EVAL_ROOT}/logs/${mode}_shard${shard}_gpu${gpu}.log" 2>&1 &
    local pid="$!"
    pids+=("${pid}")
    echo "[generation:${mode}] shard=${shard} gpu=${gpu} pid=${pid} indices=${PTM_EVAL_ROOT}/shard_indices/shard${shard}.txt"
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed )); then
    rm -rf "${mode_tmp}"
    echo "[generation:${mode}] failed; removed temporary output" >&2
    return 1
  fi
  if [[ ! -d "${mode_tmp}/${mode}" ]]; then
    rm -rf "${mode_tmp}"
    echo "[generation:${mode}] missing completed mode directory" >&2
    return 1
  fi
  mv "${mode_tmp}/${mode}" "${mode_dir}"
  rm -rf "${mode_tmp}"
  echo "[generation:${mode}] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

echo "[generation] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[generation] ckpt=${PTM_CKPT}"
echo "[generation] root=${PTM_EVAL_ROOT}"
echo "[generation] cache=${PTM_NPZ_CACHE_DIR}"
echo "[generation] gpu_list=${PTM_GPU_LIST}"
echo "[generation] shards=${PTM_NUM_SHARDS} batch=${PTM_GENERATION_BATCH_SIZE} workers=${PTM_GENERATION_NUM_WORKERS}"
echo "[generation] context_memory_only=${PTM_CONTEXT_MEMORY_ONLY} context_memory_strategy=${PTM_CONTEXT_MEMORY_STRATEGY} raw_reference_length=${PTM_RAW_REFERENCE_LENGTH} use_memory_attention=${PTM_USE_MEMORY_ATTENTION} use_memory_attention_runtime=${PTM_USE_MEMORY_ATTENTION_RUNTIME}"

for mode in ${PTM_ABLATIONS}; do
  run_mode "${mode}"
done

summary_tmp="${PTM_EVAL_ROOT}/generation_summary.json.tmp"
python ptm/scripts/aggregate_ablation_outputs.py "${PTM_EVAL_ROOT}" --kind generation --output "${summary_tmp}"
python - "${summary_tmp}" "${PTM_NUM_SHARDS}" "${PTM_GENERATION_BATCH_SIZE}" "${PTM_ABLATIONS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2]) * int(sys.argv[3])
required = tuple(sys.argv[4].split())
payload = json.loads(path.read_text(encoding="utf-8"))
missing = [mode for mode in required if mode not in payload]
if missing:
    raise SystemExit(f"missing generation ablation modes: {missing}")
bad_counts = {
    mode: data.get("num_samples")
    for mode, data in payload.items()
    if mode in required and int(data.get("num_samples", -1)) != expected
}
if bad_counts:
    raise SystemExit(f"unexpected generation sample counts: {bad_counts}; expected {expected}")
print(json.dumps({"summary_ok": str(path), "samples_per_mode": expected}, sort_keys=True), flush=True)
PY
mv "${summary_tmp}" "${PTM_EVAL_ROOT}/generation_summary.json"
echo "[generation] done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
