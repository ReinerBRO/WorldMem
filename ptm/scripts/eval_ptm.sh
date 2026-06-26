#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PTM_PREDICTIONS:-}" ]]; then
  PREDICTIONS="${PTM_PREDICTIONS}"
else
  PREDICTIONS="$(find outputs -path '*/ptm_eval/*_future_test_predictions_step*.jsonl' -type f 2>/dev/null | sort | tail -n 1)"
  if [[ -z "${PREDICTIONS}" ]]; then
    echo "set PTM_PREDICTIONS or run validation/test once to create outputs/*/ptm_eval/*_future_test_predictions_step*.jsonl" >&2
    exit 1
  fi
fi
python -m ptm.eval.eval_loop_revisit --predictions "${PREDICTIONS}" --out outputs/ptm/loop_revisit_metrics.json
python -m ptm.eval.eval_landmark_persistence --predictions "${PREDICTIONS}" --out outputs/ptm/landmark_persistence_metrics.json
python -m ptm.eval.eval_object_persistence --predictions "${PREDICTIONS}" --out outputs/ptm/object_persistence_metrics.json
python -m ptm.eval.summarize_results --runs outputs/ptm --out_dir outputs/ptm/summary
