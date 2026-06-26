#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640}"
SPLIT="${SPLIT:-train}"
START_EP="${START_EP:-0}"
END_EP="${END_EP:-0}"
RUN_TAGS="${RUN_TAGS:-}"
RUN_TAG_FILE="${RUN_TAG_FILE:-}"
GENERATOR_PATTERN="${GENERATOR_PATTERN:-ptm.data.minedojo_generator --out ${DATA_ROOT} --split ${SPLIT}}"

if [ -n "${RUN_TAG_FILE}" ] && [ -s "${RUN_TAG_FILE}" ]; then
  RUN_TAGS="$(tr '\n' ' ' < "${RUN_TAG_FILE}") ${RUN_TAGS}"
fi

kill_pattern() {
  local signal="$1"
  local pattern="$2"
  local pids
  pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
  if [ -z "${pids}" ]; then
    return 0
  fi
  while read -r pid; do
    [ -n "${pid}" ] || continue
    [ "${pid}" = "$$" ] && continue
    kill "-${signal}" "${pid}" 2>/dev/null || true
  done <<< "${pids}"
}

for tag in ${RUN_TAGS}; do
  kill_pattern TERM "${tag}"
done
kill_pattern TERM "${GENERATOR_PATTERN}"
sleep "${CLEANUP_TERM_WAIT_SECONDS:-5}"
for tag in ${RUN_TAGS}; do
  kill_pattern KILL "${tag}"
done
kill_pattern KILL "${GENERATOR_PATTERN}"

root="${DATA_ROOT}/${SPLIT}"
cleared=0
for ep in $(seq "${START_EP}" "${END_EP}"); do
  lock_dir="${root}/episode_$(printf "%06d" "${ep}").lock"
  if [ -d "${lock_dir}" ]; then
    rm -rf "${lock_dir}"
    cleared=$((cleared + 1))
  fi
done

complete=0
for ep in $(seq "${START_EP}" "${END_EP}"); do
  d="${root}/episode_$(printf "%06d" "${ep}")"
  if [ -s "${d}/frames.mp4" ] && [ -s "${d}/metadata.json" ]; then
    complete=$((complete + 1))
  fi
done

remaining=0
for tag in ${RUN_TAGS}; do
  count="$(pgrep -fc "${tag}" 2>/dev/null || true)"
  remaining=$((remaining + count))
done
gen_remaining="$(pgrep -fc "${GENERATOR_PATTERN}" 2>/dev/null || true)"
remaining=$((remaining + gen_remaining))

echo "CLEARED_LOCKS=${cleared}"
echo "COMPLETE_IN_RANGE=${complete}"
echo "REMAINING_MATCHED_PROCESSES=${remaining}"
