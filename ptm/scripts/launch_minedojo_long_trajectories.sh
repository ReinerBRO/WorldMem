#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
DATA_ROOT="${DATA_ROOT:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640}"
LOG_ROOT="${LOG_ROOT:-/gfs/space/private/zjc/logs}"
SPLIT="${SPLIT:-train}"
START_EP="${START_EP:-0}"
END_EP="${END_EP:-999}"
WORKERS="${WORKERS:-24}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
GPU_COUNT="${GPU_COUNT:-8}"
HOST_LABEL="${HOST_LABEL:-$(hostname | cut -d. -f1)}"
RUN_TAG="${RUN_TAG:-ptm_long1500_${SPLIT}_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/${RUN_TAG}}"
DISPLAY_BASE="${DISPLAY_BASE:-2200}"
XORG_ROOT="${XORG_ROOT:-/gfs/space/private/zjc/aptroot/jammy_xorg/root}"

if [ "${END_EP}" -lt "${START_EP}" ]; then
  echo "END_EP must be >= START_EP" >&2
  exit 2
fi
if [ "${WORKERS}" -lt 1 ]; then
  echo "WORKERS must be >= 1" >&2
  exit 2
fi
if [ "${GPU_COUNT}" -lt 1 ]; then
  echo "GPU_COUNT must be >= 1" >&2
  exit 2
fi

mkdir -p "${LOGDIR}" "${DATA_ROOT}/${SPLIT}"
PIDFILE="${LOGDIR}/${HOST_LABEL}_${SPLIT}.pids"
if [ "${PIDFILE_APPEND:-0}" = "1" ]; then
  : >> "${PIDFILE}"
else
  : > "${PIDFILE}"
fi

cd "${PROJECT_ROOT}"

total=$((END_EP - START_EP + 1))
base_count=$((total / WORKERS))
remainder=$((total % WORKERS))
cursor="${START_EP}"

for local_idx in $(seq 0 $((WORKERS - 1))); do
  count="${base_count}"
  if [ "${local_idx}" -lt "${remainder}" ]; then
    count=$((count + 1))
  fi
  if [ "${count}" -le 0 ]; then
    continue
  fi
  worker_start="${cursor}"
  worker_end=$((cursor + count - 1))
  cursor=$((worker_end + 1))

  idx=$((WORKER_OFFSET + local_idx))
  worker="${HOST_LABEL}_long_$(printf "%02d" "${idx}")"
  display_num=$((DISPLAY_BASE + idx))
  cuda_dev=$((idx % GPU_COUNT))
  log="${LOGDIR}/worker_${worker}_${SPLIT}_${worker_start}_${worker_end}.log"
  (
    export LOGDIR="${LOGDIR}"
    export PROJECT_ROOT="${PROJECT_ROOT}"
    export DATA_ROOT="${DATA_ROOT}"
    export MICRO_SPLIT="${SPLIT}"
    export MICRO_DISPLAY_NUM="${display_num}"
    export MICRO_CUDA_DEV="${cuda_dev}"
    export MICRO_GRADLE_SUFFIX="${RUN_TAG}_${worker}"
    export MICRO_GRADLE_USER_HOME="/gfs/space/private/zjc/.gradle_ptm_${RUN_TAG}_${worker}"
    export MICRO_HOME="/gfs/space/private/zjc/home_ptm_${RUN_TAG}_${worker}"
    export MICRO_TMPDIR="/gfs/space/private/zjc/tmp"
    export MICRO_XORG_ROOT="${XORG_ROOT}"
    export MICRO_FRAMES_PER_EPISODE="${MICRO_FRAMES_PER_EPISODE:-1500}"
    export MICRO_SCHEDULE_TOTAL="${MICRO_SCHEDULE_TOTAL:-$((END_EP + 1))}"
    export MICRO_HEIGHT="${MICRO_HEIGHT:-360}"
    export MICRO_WIDTH="${MICRO_WIDTH:-640}"
    export MICRO_FAMILIES="${MICRO_FAMILIES:-balanced}"
    export MICRO_BACKEND="${MICRO_BACKEND:-minedojo}"
    export MICRO_FRAME_STORAGE="${MICRO_FRAME_STORAGE:-mp4}"
    export MICRO_SEED="${MICRO_SEED:-20260622}"
    export MICRO_HISTORY_LENGTH="${MICRO_HISTORY_LENGTH:-600}"
    export MICRO_FUTURE_LENGTH="${MICRO_FUTURE_LENGTH:-100}"
    export MICRO_TEST_STRIDE="${MICRO_TEST_STRIDE:-50}"
    export MICRO_BATCH_EPISODES="${MICRO_BATCH_EPISODES:-1}"
    export MICRO_BATCH_TIMEOUT="${MICRO_BATCH_TIMEOUT:-}"
    export MICRO_EP_TIMEOUT="${MICRO_EP_TIMEOUT:-7200}"
    export MICRO_GENERATOR_EPISODE_TIMEOUT="${MICRO_GENERATOR_EPISODE_TIMEOUT:-5400}"
    export MICRO_LOCK_STALE_SECONDS="${MICRO_LOCK_STALE_SECONDS:-7200}"
    export MICRO_CONTINUE_ON_ERROR="${MICRO_CONTINUE_ON_ERROR:-1}"
    export MICRO_REUSE_ENV="${MICRO_REUSE_ENV:-1}"
    export MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE="${MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE:-200}"
    export MICRO_EPISODE_RETRIES="${MICRO_EPISODE_RETRIES:-5}"
    export MICRO_EPISODE_RETRY_SEED_STRIDE="${MICRO_EPISODE_RETRY_SEED_STRIDE:-1000003}"
    nohup bash ptm/scripts/run_minedojo_episode_range.sh "${worker}" "${worker_start}" "${worker_end}" "${RUN_TAG}" > "${log}" 2>&1 &
    echo "$! split=${SPLIT} worker=${worker} range=${worker_start}-${worker_end} log=${log}"
  ) >> "${PIDFILE}"
  sleep "${LAUNCH_STAGGER_SECONDS:-1}"
done

echo "RUN_TAG=${RUN_TAG}"
echo "LOGDIR=${LOGDIR}"
echo "PIDFILE=${PIDFILE}"
echo "SPLIT=${SPLIT} RANGE=${START_EP}-${END_EP} WORKERS=${WORKERS} DATA_ROOT=${DATA_ROOT}"
