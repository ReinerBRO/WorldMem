#!/usr/bin/env bash
set -euo pipefail

: "${PTM_CKPT:?set PTM_CKPT to a trained PTM Lightning checkpoint}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

export PTM_CONTEXT_MEMORY_ONLY="${PTM_CONTEXT_MEMORY_ONLY:-true}"
export PTM_RAW_REFERENCE_LENGTH="${PTM_RAW_REFERENCE_LENGTH:-0}"
# Keep the raw memory-attention modules in the model so V1/V2 checkpoints
# trained with use_memory_attention=true load strictly, but disable the raw
# memory-attention forward path for the clean context-only diagnostic.
export PTM_USE_MEMORY_ATTENTION="${PTM_USE_MEMORY_ATTENTION:-true}"
export PTM_USE_MEMORY_ATTENTION_RUNTIME="${PTM_USE_MEMORY_ATTENTION_RUNTIME:-false}"
export PTM_USE_PTM_REFERENCE_ADAPTER="${PTM_USE_PTM_REFERENCE_ADAPTER:-false}"
export PTM_USE_PTM_CROSS_ATTENTION="${PTM_USE_PTM_CROSS_ATTENTION:-true}"
export PTM_ABLATIONS="${PTM_ABLATIONS:-normal zero_token shuffle_token}"

# Existing V1/V2/V3 and gate checkpoints were trained with ptm_max_history=16.
# Keep this default so checkpoint position embeddings load without shape drift.
export PTM_MAX_HISTORY="${PTM_MAX_HISTORY:-16}"
export PTM_MAX_HISTORY_CANDIDATES="${PTM_MAX_HISTORY_CANDIDATES:-16}"

export PTM_EVAL_LABEL="${PTM_EVAL_LABEL:-generation_context_memory_p0_$(date +%Y%m%d_%H%M%S)}"

exec bash "${SCRIPT_DIR}/run_generation_ablation_clean.sh"
