#!/usr/bin/env bash
set -euo pipefail

PTM_CONTEXT_LENGTH="${PTM_CONTEXT_LENGTH:-4}"
PTM_FUTURE_LENGTH="${PTM_FUTURE_LENGTH:-4}"
PTM_N_FRAMES="${PTM_N_FRAMES:-$((PTM_CONTEXT_LENGTH + PTM_FUTURE_LENGTH))}"
PTM_VAL_CONTEXT_LENGTH="${PTM_VAL_CONTEXT_LENGTH:-600}"
PTM_VAL_FUTURE_LENGTH="${PTM_VAL_FUTURE_LENGTH:-100}"
PTM_N_FRAMES_VALID="${PTM_N_FRAMES_VALID:-$((PTM_VAL_CONTEXT_LENGTH + PTM_VAL_FUTURE_LENGTH))}"
PTM_MEMORY_CONDITION_LENGTH="${PTM_MEMORY_CONDITION_LENGTH:-8}"
PTM_DATA_ROOT="${PTM_DATA_ROOT:-ptm_minedojo_data/stage1}"
PTM_RUN_NAME="${PTM_RUN_NAME:-ptm_worldmem_stage1}"
PTM_MAX_STEPS="${PTM_MAX_STEPS:-6000}"
PTM_BATCH_SIZE="${PTM_BATCH_SIZE:-1}"
PTM_NUM_WORKERS="${PTM_NUM_WORKERS:-8}"
PTM_VAL_BATCH_SIZE="${PTM_VAL_BATCH_SIZE:-1}"
PTM_VAL_NUM_WORKERS="${PTM_VAL_NUM_WORKERS:-4}"
PTM_LIMIT_VAL_BATCH="${PTM_LIMIT_VAL_BATCH:-2}"
PTM_VAL_EVERY_N_STEP="${PTM_VAL_EVERY_N_STEP:-1500}"
PTM_CKPT_EVERY="${PTM_CKPT_EVERY:-1500}"
PTM_CKPT_SAVE_TOP_K="${PTM_CKPT_SAVE_TOP_K:--1}"
PTM_CKPT_SAVE_LAST="${PTM_CKPT_SAVE_LAST:-true}"
PTM_PRECISION="${PTM_PRECISION:-16-mixed}"
PTM_LOG_VIDEO="${PTM_LOG_VIDEO:-true}"
PTM_EVAL_ONLY="${PTM_EVAL_ONLY:-false}"
PTM_MAX_LOG_VIDEOS="${PTM_MAX_LOG_VIDEOS:-1}"
PTM_VIDEO_LOG_STAGE="${PTM_VIDEO_LOG_STAGE:-ptm}"
PTM_USE_MEMORY_ATTENTION="${PTM_USE_MEMORY_ATTENTION:-false}"
PTM_USE_MEMORY_ATTENTION_RUNTIME="${PTM_USE_MEMORY_ATTENTION_RUNTIME:-${PTM_USE_MEMORY_ATTENTION}}"
PTM_RAW_REFERENCE_LENGTH="${PTM_RAW_REFERENCE_LENGTH:-${PTM_MEMORY_CONDITION_LENGTH}}"
PTM_USE_PTM_REFERENCE_ADAPTER="${PTM_USE_PTM_REFERENCE_ADAPTER:-true}"
PTM_USE_PTM_CROSS_ATTENTION="${PTM_USE_PTM_CROSS_ATTENTION:-true}"
PTM_TRAIN_CONTEXT_MEMORY_ONLY="${PTM_TRAIN_CONTEXT_MEMORY_ONLY:-false}"
PTM_TRAIN_CONTEXT_TOKEN_SOURCE="${PTM_TRAIN_CONTEXT_TOKEN_SOURCE:-reference_tail}"
PTM_CONTEXT_MEMORY_ONLY="${PTM_CONTEXT_MEMORY_ONLY:-false}"
PTM_CONTEXT_MEMORY_STRATEGY="${PTM_CONTEXT_MEMORY_STRATEGY:-strided}"
PTM_DETACH_FOR_GENERATION="${PTM_DETACH_FOR_GENERATION:-false}"
PTM_CONTRAST_WEIGHT="${PTM_CONTRAST_WEIGHT:-0.0}"
PTM_CONTRAST_MARGIN="${PTM_CONTRAST_MARGIN:-0.02}"
PTM_TRAIN_CONSUMER_ONLY="${PTM_TRAIN_CONSUMER_ONLY:-false}"
PTM_VISUAL_MEMORY_SELECTION="${PTM_VISUAL_MEMORY_SELECTION:-false}"
PTM_VISUAL_TOP_K="${PTM_VISUAL_TOP_K:-8}"
PTM_VISUAL_NUM_CANDIDATES="${PTM_VISUAL_NUM_CANDIDATES:-${PTM_MAX_HISTORY_CANDIDATES:-64}}"
PTM_VISUAL_POOL="${PTM_VISUAL_POOL:-grid2x2}"
PTM_VISUAL_CANDIDATE_SOURCE="${PTM_VISUAL_CANDIDATE_SOURCE:-context_strided}"
PTM_VISUAL_INCLUDE_SUMMARY_TOKENS="${PTM_VISUAL_INCLUDE_SUMMARY_TOKENS:-true}"
PTM_VISUAL_REMAP_MATCH_LABELS="${PTM_VISUAL_REMAP_MATCH_LABELS:-true}"
PTM_VISUAL_ROUTING_MODE="${PTM_VISUAL_ROUTING_MODE:-proxy_topk}"
PTM_VISUAL_ROUTE_TOP_M="${PTM_VISUAL_ROUTE_TOP_M:-8}"
PTM_VISUAL_ROUTE_TAU="${PTM_VISUAL_ROUTE_TAU:-0.2}"
PTM_VISUAL_ROUTE_PRIOR_ALPHA="${PTM_VISUAL_ROUTE_PRIOR_ALPHA:-1.0}"
PTM_VISUAL_ROUTE_DIM="${PTM_VISUAL_ROUTE_DIM:-0}"
PTM_FRAME_HEIGHT="${PTM_FRAME_HEIGHT:-360}"
PTM_FRAME_WIDTH="${PTM_FRAME_WIDTH:-640}"
PTM_VIDEO_CACHE_SIZE="${PTM_VIDEO_CACHE_SIZE:-2}"
PTM_FORMAL_NPZ_CACHE_DIR="/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640_npz_cache"
PTM_NPZ_CACHE_DIR="${PTM_NPZ_CACHE_DIR:-${PTM_FORMAL_NPZ_CACHE_DIR}}"
PTM_NPZ_CACHE_DIR_VAL="${PTM_NPZ_CACHE_DIR_VAL:-}"
PTM_NPZ_CACHE_SPLITS="${PTM_NPZ_CACHE_SPLITS:-train}"
PTM_NPZ_VAL_INDICES_FILE="${PTM_NPZ_VAL_INDICES_FILE:-}"
PTM_VAL_INDICES_FILE="${PTM_VAL_INDICES_FILE:-}"
PTM_VERIFY_DATASET="${PTM_VERIFY_DATASET:-false}"
PTM_WINDOW_CENTERS="${PTM_WINDOW_CENTERS:-target,late50,late75,late100}"
PTM_VALIDATION_ABLATION_MODES="${PTM_VALIDATION_ABLATION_MODES:-normal}"
PTM_VALIDATION_VIDEO_MODE="${PTM_VALIDATION_VIDEO_MODE:-normal}"
PTM_GENERATION_TARGET_LOSS_WEIGHT="${PTM_GENERATION_TARGET_LOSS_WEIGHT:-1.0}"
PTM_GENERATION_LATE_LOSS_WEIGHT="${PTM_GENERATION_LATE_LOSS_WEIGHT:-0.5}"
PTM_GENERATION_TARGET_WINDOW_RADIUS="${PTM_GENERATION_TARGET_WINDOW_RADIUS:-1}"
PTM_GENERATION_LATE_HORIZON_START="${PTM_GENERATION_LATE_HORIZON_START:-50}"
if [[ "${PTM_NPZ_CACHE_DIR}" != "${PTM_FORMAL_NPZ_CACHE_DIR}" ]]; then
  echo "refusing non-formal PTM_NPZ_CACHE_DIR=${PTM_NPZ_CACHE_DIR}; expected ${PTM_FORMAL_NPZ_CACHE_DIR}" >&2
  exit 2
fi
PTM_REQUIRED_MEMORY_STRATEGY="${PTM_REQUIRED_MEMORY_STRATEGY:-causal_slots}"
PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-$((PTM_N_FRAMES_VALID + PTM_MEMORY_CONDITION_LENGTH))}"
PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-${PTM_MAX_HISTORY}}"
PTM_INIT_MODE="${PTM_INIT_MODE:-oasis}"
PTM_DIFFUSION_CKPT="${PTM_DIFFUSION_CKPT:-/gfs/space/private/zjc/models/oasis-500m/oasis500m.safetensors}"
PTM_VAE_CKPT="${PTM_VAE_CKPT:-/gfs/space/private/zjc/models/oasis-500m/vit-l-20.safetensors}"
PTM_ZERO_INIT_GATE="${PTM_ZERO_INIT_GATE:-true}"
PTM_BASE_CKPT="${PTM_BASE_CKPT:-}"
PTM_RESUME_CKPT="${PTM_RESUME_CKPT:-}"
PTM_REQUIRE_BASE_CKPT="${PTM_REQUIRE_BASE_CKPT:-true}"
PTM_LOCAL_SAVE_DIR="${PTM_LOCAL_SAVE_DIR:-}"
PTM_CACHE_ROOT="${PTM_CACHE_ROOT:-/gfs/space/private/zjc/.cache}"
PTM_NUM_NODES="${PTM_NUM_NODES:-1}"
PTM_WANDB_RESUME_ID="${PTM_WANDB_RESUME_ID:-}"
WANDB_ENTITY="${WANDB_ENTITY:-jinczhu12-hkust}"
WANDB_PROJECT="${WANDB_PROJECT:-PTM}"

to_hydra_list() {
  local value="$1"
  if [[ "${value}" == \[*\] ]]; then
    printf '%s' "${value}"
    return
  fi
  local IFS=,
  local parts=()
  read -ra parts <<< "${value}"
  local out="["
  local part
  for part in "${parts[@]}"; do
    part="${part//[[:space:]]/}"
    if [[ -z "${part}" ]]; then
      continue
    fi
    if [[ "${out}" != "[" ]]; then
      out+=","
    fi
    out+="${part}"
  done
  out+="]"
  printf '%s' "${out}"
}

PTM_NPZ_CACHE_SPLITS_HYDRA="$(to_hydra_list "${PTM_NPZ_CACHE_SPLITS}")"

export TORCH_HOME="${TORCH_HOME:-${PTM_CACHE_ROOT}/torch}"
export HF_HOME="${HF_HOME:-${PTM_CACHE_ROOT}/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PTM_CACHE_ROOT}/xdg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PTM_CACHE_ROOT}/matplotlib}"
mkdir -p "${TORCH_HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

if [[ "${PTM_VAL_EVERY_N_STEP}" =~ ^[0-9]+$ && "${PTM_MAX_STEPS}" =~ ^[0-9]+$ ]]; then
  if (( PTM_VAL_EVERY_N_STEP > PTM_MAX_STEPS )); then
    echo "refusing PTM_VAL_EVERY_N_STEP=${PTM_VAL_EVERY_N_STEP} > PTM_MAX_STEPS=${PTM_MAX_STEPS}; this would skip validation for the whole run." >&2
    exit 2
  fi
fi

python - \
  "${PTM_NPZ_CACHE_DIR}" \
  "${PTM_REQUIRED_MEMORY_STRATEGY}" \
  "${PTM_WINDOW_CENTERS}" \
  "${PTM_NPZ_CACHE_SPLITS}" \
  "${PTM_CONTEXT_LENGTH}" \
  "${PTM_FUTURE_LENGTH}" \
  "${PTM_MEMORY_CONDITION_LENGTH}" \
  "${PTM_VAL_CONTEXT_LENGTH}" \
  "${PTM_VAL_FUTURE_LENGTH}" \
  "${PTM_NPZ_CACHE_DIR_VAL}" <<'PY'
import json
import sys
from pathlib import Path

cache_dir = Path(sys.argv[1])
required = sys.argv[2]
expected_centers = [part.strip() for part in sys.argv[3].split(",") if part.strip()]
splits_raw = sys.argv[4].strip()
if splits_raw.startswith("[") and splits_raw.endswith("]"):
    splits_raw = splits_raw[1:-1]
splits = {part.strip() for part in splits_raw.split(",") if part.strip()}
train_context = int(sys.argv[5])
train_future = int(sys.argv[6])
train_memory = int(sys.argv[7])
val_context = int(sys.argv[8])
val_future = int(sys.argv[9])
val_cache_arg = sys.argv[10].strip()

def require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise SystemExit(f"invalid PTM NPZ cache {name}={actual!r}; expected {expected!r}")

def load_manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing NPZ cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strategy = manifest.get("memory_strategy")
    if strategy != required:
        raise SystemExit(f"invalid PTM cache memory_strategy={strategy!r}; expected {required!r}")
    return manifest

def validate_train_cache(manifest: dict) -> None:
    require_equal("context_length", int(manifest.get("context_length", -1)), train_context)
    require_equal("future_length", int(manifest.get("future_length", -1)), train_future)
    require_equal("memory_condition_length", int(manifest.get("memory_condition_length", -1)), train_memory)
    require_equal("window_centers", manifest.get("window_centers"), expected_centers)

def validate_eval_cache(manifest: dict) -> None:
    layout = (
        int(manifest.get("context_length", -1)),
        int(manifest.get("future_length", -1)),
        int(manifest.get("memory_condition_length", -1)),
    )
    long_eval_layouts = {
        (val_context, val_future, 0),
        (val_context, val_future, train_memory),
    }
    short_cache_layout = (train_context, train_future, train_memory)
    if layout in long_eval_layouts:
        require_equal("window_centers", manifest.get("window_centers"), ["target"])
    elif layout == short_cache_layout:
        require_equal("window_centers", manifest.get("window_centers"), expected_centers)
    else:
        raise SystemExit(
            "invalid PTM NPZ cache eval layout="
            f"{layout!r}; expected long eval one of {sorted(long_eval_layouts)!r} "
            f"or short cached {short_cache_layout!r}"
        )

uses_train = "all" in splits or "train" in splits
uses_eval = "all" in splits or bool({"val", "test"} & splits)
primary_manifest = load_manifest(cache_dir)
if uses_train:
    validate_train_cache(primary_manifest)
if uses_eval and not val_cache_arg:
    validate_eval_cache(primary_manifest)
if uses_eval and val_cache_arg:
    validate_eval_cache(load_manifest(Path(val_cache_arg)))
PY

ARGS=(
  +name="${PTM_RUN_NAME}"
  dataset=ptm_minedojo
  dataset.save_dir="${PTM_DATA_ROOT}"
  dataset.resolution="[${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]"
  dataset.observation_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]"
  dataset.n_frames="${PTM_N_FRAMES}"
  +dataset.n_frames_valid="${PTM_N_FRAMES_VALID}"
  dataset.context_length="${PTM_CONTEXT_LENGTH}"
  dataset.future_length="${PTM_FUTURE_LENGTH}"
  dataset.ptm_context_length="${PTM_CONTEXT_LENGTH}"
  dataset.ptm_future_length="${PTM_FUTURE_LENGTH}"
  +dataset.ptm_context_length_valid="${PTM_VAL_CONTEXT_LENGTH}"
  +dataset.ptm_future_length_valid="${PTM_VAL_FUTURE_LENGTH}"
  dataset.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}"
  dataset.max_history_candidates="${PTM_MAX_HISTORY_CANDIDATES}"
  +dataset.video_cache_size="${PTM_VIDEO_CACHE_SIZE}"
  +dataset.npz_cache_splits="${PTM_NPZ_CACHE_SPLITS_HYDRA}"
  algorithm.x_shape="[3,${PTM_FRAME_HEIGHT},${PTM_FRAME_WIDTH}]"
  algorithm.context_frames="${PTM_VAL_CONTEXT_LENGTH}"
  ++algorithm.memory_condition_length="${PTM_MEMORY_CONDITION_LENGTH}"
  ++algorithm.raw_reference_length="${PTM_RAW_REFERENCE_LENGTH}"
  ++algorithm.use_memory_attention="${PTM_USE_MEMORY_ATTENTION}"
  ++algorithm.use_memory_attention_runtime="${PTM_USE_MEMORY_ATTENTION_RUNTIME}"
  ++algorithm.use_ptm_memory=true
  ++algorithm.use_ptm_reference_adapter="${PTM_USE_PTM_REFERENCE_ADAPTER}"
  ++algorithm.use_ptm_cross_attention="${PTM_USE_PTM_CROSS_ATTENTION}"
  ++algorithm.ptm_train_context_memory_only="${PTM_TRAIN_CONTEXT_MEMORY_ONLY}"
  ++algorithm.ptm_train_context_token_source="${PTM_TRAIN_CONTEXT_TOKEN_SOURCE}"
  ++algorithm.ptm_context_memory_only="${PTM_CONTEXT_MEMORY_ONLY}"
  ++algorithm.ptm_context_memory_strategy="${PTM_CONTEXT_MEMORY_STRATEGY}"
  ++algorithm.log_video="${PTM_LOG_VIDEO}"
  ++algorithm.ptm_eval_only="${PTM_EVAL_ONLY}"
  ++algorithm.max_log_videos="${PTM_MAX_LOG_VIDEOS}"
  ++algorithm.video_log_stage="${PTM_VIDEO_LOG_STAGE}"
  algorithm.num_memory_tokens="${PTM_MEMORY_TOKENS:-16}"
  ++algorithm.ptm_max_history="${PTM_MAX_HISTORY}"
  ++algorithm.ptm_max_history_candidates="${PTM_MAX_HISTORY_CANDIDATES}"
  ++algorithm.ptm_loss_weight="${PTM_LOSS_WEIGHT:-0.25}"
  ++algorithm.ptm_bottleneck_weight="${PTM_BOTTLENECK_WEIGHT:-0.001}"
  ++algorithm.ptm_detach_for_generation="${PTM_DETACH_FOR_GENERATION}"
  ++algorithm.ptm_contrast_weight="${PTM_CONTRAST_WEIGHT}"
  ++algorithm.ptm_contrast_margin="${PTM_CONTRAST_MARGIN}"
  ++algorithm.ptm_train_consumer_only="${PTM_TRAIN_CONSUMER_ONLY}"
  ++algorithm.ptm_visual_memory_selection="${PTM_VISUAL_MEMORY_SELECTION}"
  ++algorithm.ptm_visual_top_k="${PTM_VISUAL_TOP_K}"
  ++algorithm.ptm_visual_num_candidates="${PTM_VISUAL_NUM_CANDIDATES}"
  ++algorithm.ptm_visual_pool="${PTM_VISUAL_POOL}"
  ++algorithm.ptm_visual_candidate_source="${PTM_VISUAL_CANDIDATE_SOURCE}"
  ++algorithm.ptm_visual_include_summary_tokens="${PTM_VISUAL_INCLUDE_SUMMARY_TOKENS}"
  ++algorithm.ptm_visual_remap_match_labels="${PTM_VISUAL_REMAP_MATCH_LABELS}"
  ++algorithm.ptm_visual_routing_mode="${PTM_VISUAL_ROUTING_MODE}"
  ++algorithm.ptm_visual_route_top_m="${PTM_VISUAL_ROUTE_TOP_M}"
  ++algorithm.ptm_visual_route_tau="${PTM_VISUAL_ROUTE_TAU}"
  ++algorithm.ptm_visual_route_prior_alpha="${PTM_VISUAL_ROUTE_PRIOR_ALPHA}"
  ++algorithm.ptm_visual_route_dim="${PTM_VISUAL_ROUTE_DIM}"
  ++algorithm.generation_target_loss_weight="${PTM_GENERATION_TARGET_LOSS_WEIGHT}"
  ++algorithm.generation_late_loss_weight="${PTM_GENERATION_LATE_LOSS_WEIGHT}"
  ++algorithm.generation_target_window_radius="${PTM_GENERATION_TARGET_WINDOW_RADIUS}"
  ++algorithm.generation_late_horizon_start="${PTM_GENERATION_LATE_HORIZON_START}"
  ++algorithm.validation_ablation_modes="$(to_hydra_list "${PTM_VALIDATION_ABLATION_MODES}")"
  ++algorithm.validation_video_mode="${PTM_VALIDATION_VIDEO_MODE}"
  experiment.training.max_steps="${PTM_MAX_STEPS}"
  experiment.training.batch_size="${PTM_BATCH_SIZE}"
  experiment.training.data.num_workers="${PTM_NUM_WORKERS}"
  experiment.training.precision="${PTM_PRECISION}"
  experiment.num_nodes="${PTM_NUM_NODES}"
  experiment.training.checkpointing.every_n_train_steps="${PTM_CKPT_EVERY}"
  experiment.training.checkpointing.save_top_k="${PTM_CKPT_SAVE_TOP_K}"
  +experiment.training.checkpointing.save_last="${PTM_CKPT_SAVE_LAST}"
  experiment.validation.batch_size="${PTM_VAL_BATCH_SIZE}"
  experiment.validation.data.num_workers="${PTM_VAL_NUM_WORKERS}"
  experiment.validation.limit_batch="${PTM_LIMIT_VAL_BATCH}"
  experiment.validation.val_every_n_step="${PTM_VAL_EVERY_N_STEP}"
  wandb.mode="${WANDB_MODE:-disabled}"
  wandb.entity="${WANDB_ENTITY}"
  wandb.project="${WANDB_PROJECT}"
)

if [[ -n "${PTM_OUTPUT_DIR:-}" ]]; then
  ARGS+=(+output_dir="${PTM_OUTPUT_DIR}")
fi

if [[ -n "${PTM_LOCAL_SAVE_DIR:-}" ]]; then
  ARGS+=(++algorithm.local_save_dir="${PTM_LOCAL_SAVE_DIR}")
fi

if [[ -n "${PTM_NPZ_CACHE_DIR}" ]]; then
  ARGS+=(+dataset.npz_cache_dir="${PTM_NPZ_CACHE_DIR}")
fi

if [[ -n "${PTM_NPZ_CACHE_DIR_VAL}" ]]; then
  ARGS+=(+dataset.npz_cache_dir_val="${PTM_NPZ_CACHE_DIR_VAL}")
fi

if [[ -n "${PTM_NPZ_VAL_INDICES_FILE}" ]]; then
  ARGS+=(+dataset.npz_cache_indices_file_val="${PTM_NPZ_VAL_INDICES_FILE}")
fi

if [[ -n "${PTM_VAL_INDICES_FILE}" ]]; then
  ARGS+=(+dataset.indices_file_val="${PTM_VAL_INDICES_FILE}")
fi

if [[ -n "${PTM_WANDB_RESUME_ID}" ]]; then
  ARGS+=(resume="${PTM_WANDB_RESUME_ID}")
fi

if [[ -n "${PTM_RESUME_CKPT}" ]]; then
  if [[ ! -f "${PTM_RESUME_CKPT}" ]]; then
    echo "PTM_RESUME_CKPT not found: ${PTM_RESUME_CKPT}" >&2
    exit 2
  fi
  ARGS+=(load="${PTM_RESUME_CKPT}" +customized_load=false +seperate_load=false +zero_init_gate=false)
else
case "${PTM_INIT_MODE}" in
  oasis)
    ARGS+=(
      +diffusion_model_path="${PTM_DIFFUSION_CKPT}"
      +vae_path="${PTM_VAE_CKPT}"
      +customized_load=true
      +seperate_load=true
      +zero_init_gate="${PTM_ZERO_INIT_GATE}"
    )
    ;;
  checkpoint)
    if [[ -n "${PTM_BASE_CKPT:-}" ]]; then
      if [[ -f "${PTM_BASE_CKPT}" ]]; then
        ARGS+=(load="${PTM_BASE_CKPT}" +customized_load=true +seperate_load=false)
      elif [[ "${PTM_REQUIRE_BASE_CKPT}" == "true" ]]; then
        echo "PTM_BASE_CKPT not found: ${PTM_BASE_CKPT}" >&2
        exit 2
      else
        echo "warning: PTM_BASE_CKPT not found, training from scratch: ${PTM_BASE_CKPT}" >&2
      fi
    fi
    ;;
  scratch)
    ;;
  *)
    echo "unsupported PTM_INIT_MODE=${PTM_INIT_MODE}; expected oasis, checkpoint, or scratch" >&2
    exit 2
    ;;
esac
fi

if [[ "${PTM_VERIFY_DATASET}" == "true" ]]; then
  python -m ptm.data.verify_dataset --data_root "${PTM_DATA_ROOT}"
fi
python main.py "${ARGS[@]}"
