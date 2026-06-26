#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
DATA_ROOT="${DATA_ROOT:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/stage1_360x640}"
LOG_ROOT="${LOG_ROOT:-/gfs/space/private/zjc/logs}"
HOST_LABEL="${HOST_LABEL:-$(hostname | cut -d- -f1)}"
WORKERS="${WORKERS:-24}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
RUN_TAG="${RUN_TAG:-ptm_stage1_360x640_full_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-${LOG_ROOT}/${RUN_TAG}}"
START_EP="${START_EP:-0}"
END_EP="${END_EP:-2999}"
DISPLAY_BASE="${DISPLAY_BASE:-1200}"
XORG_ROOT="${XORG_ROOT:-/gfs/space/private/zjc/aptroot/ubuntu_xvfb}"

mkdir -p "${LOGDIR}" "${DATA_ROOT}/train"
PIDFILE="${LOGDIR}/${HOST_LABEL}.pids"
if [ "${PIDFILE_APPEND:-0}" = "1" ]; then
  : >> "${PIDFILE}"
else
  : > "${PIDFILE}"
fi

cd "${PROJECT_ROOT}"

for local_idx in $(seq 0 $((WORKERS - 1))); do
  idx=$((WORKER_OFFSET + local_idx))
  worker="${HOST_LABEL}_g$(printf "%02d" "${idx}")"
  display_num=$((DISPLAY_BASE + idx))
  cuda_dev=$((idx % 8))
  log="${LOGDIR}/worker_${worker}_${START_EP}_${END_EP}.log"
  (
    export LOGDIR="${LOGDIR}"
    export PROJECT_ROOT="${PROJECT_ROOT}"
    export DATA_ROOT="${DATA_ROOT}"
    export MICRO_DISPLAY_NUM="${display_num}"
    export MICRO_CUDA_DEV="${cuda_dev}"
    if [ -n "${GRADLE_HOME_PREFIX:-}" ]; then
      export MICRO_GRADLE_SUFFIX="${HOST_LABEL}_$(printf "%02d" "${idx}")"
      export MICRO_GRADLE_USER_HOME="${GRADLE_HOME_PREFIX}$(printf "%02d" "${idx}")"
    else
      export MICRO_GRADLE_SUFFIX="${RUN_TAG}_${worker}"
      export MICRO_GRADLE_USER_HOME="/gfs/space/private/zjc/.gradle_ptm_${RUN_TAG}_${worker}"
    fi
    export MICRO_HOME="/gfs/space/private/zjc/home_ptm_${RUN_TAG}_${worker}"
    export MICRO_TMPDIR="/gfs/space/private/zjc/tmp"
    export MICRO_XORG_ROOT="${XORG_ROOT}"
    export MICRO_FRAMES_PER_EPISODE="${MICRO_FRAMES_PER_EPISODE:-128}"
    export MICRO_SCHEDULE_TOTAL="${MICRO_SCHEDULE_TOTAL:-3000}"
    export MICRO_HEIGHT="${MICRO_HEIGHT:-360}"
    export MICRO_WIDTH="${MICRO_WIDTH:-640}"
    export MICRO_FAMILIES="${MICRO_FAMILIES:-balanced}"
    export MICRO_BACKEND="${MICRO_BACKEND:-minedojo}"
    export MICRO_FRAME_STORAGE="${MICRO_FRAME_STORAGE:-mp4}"
    export MICRO_SEED="${MICRO_SEED:-20260622}"
    export MICRO_BATCH_EPISODES="${MICRO_BATCH_EPISODES:-100000}"
    export MICRO_BATCH_TIMEOUT="${MICRO_BATCH_TIMEOUT:-21600}"
    export MICRO_EP_TIMEOUT="${MICRO_EP_TIMEOUT:-1800}"
    export MICRO_LOCK_STALE_SECONDS="${MICRO_LOCK_STALE_SECONDS:-7200}"
    export MICRO_CONTINUE_ON_ERROR="${MICRO_CONTINUE_ON_ERROR:-1}"
    export MICRO_REUSE_ENV="${MICRO_REUSE_ENV:-1}"
    export MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE="${MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE:-200}"
    nohup bash ptm/scripts/run_episode_refill_range.sh "${worker}" "${START_EP}" "${END_EP}" "${RUN_TAG}" > "${log}" 2>&1 &
    echo "$! ${worker} ${log}"
  ) >> "${PIDFILE}"
  sleep "${LAUNCH_STAGGER_SECONDS:-1}"
done

echo "RUN_TAG=${RUN_TAG}"
echo "LOGDIR=${LOGDIR}"
echo "PIDFILE=${PIDFILE}"
echo "HOST_LABEL=${HOST_LABEL} WORKERS=${WORKERS} WORKER_OFFSET=${WORKER_OFFSET} RANGE=${START_EP}-${END_EP} DATA_ROOT=${DATA_ROOT}"
