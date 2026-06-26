#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
DATA_ROOT="${DATA_ROOT:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/stage1_360x640}"
RUN_TAG="${RUN_TAG:-ptm_stage1_360x640_refill_$(date +%Y%m%d_%H%M%S)}"
LOGDIR="${LOGDIR:-/gfs/space/private/zjc/logs/${RUN_TAG}}"
EPISODES="${EPISODES:-62 65 69 200 204 207}"
HOST_LABEL="${HOST_LABEL:-$(hostname | cut -d. -f1)}"
DISPLAY_BASE="${DISPLAY_BASE:-1700}"
CUDA_BASE="${CUDA_BASE:-0}"
GRADLE_HOME_PREFIX="${GRADLE_HOME_PREFIX:-/gfs/space/private/zjc/.gradle_ptm_refill_}"
XORG_ROOT="${XORG_ROOT:-/gfs/space/private/zjc/aptroot/jammy_xorg/root}"

mkdir -p "${LOGDIR}" "${DATA_ROOT}/train"
PIDFILE="${LOGDIR}/${HOST_LABEL}.pids"
: > "${PIDFILE}"

idx=0
for ep in ${EPISODES}; do
  ep_dir="${DATA_ROOT}/train/episode_$(printf "%06d" "${ep}")"
  if [ -s "${ep_dir}/metadata.json" ] && [ -s "${ep_dir}/frames.mp4" ]; then
    echo "REFILL_SKIP ep=${ep} reason=complete"
    idx=$((idx + 1))
    continue
  fi

  worker="episode_refill_${ep}"
  log="${LOGDIR}/${worker}.log"
  display_num=$((DISPLAY_BASE + idx))
  cuda_dev=$(((CUDA_BASE + idx) % 8))
  gradle_suffix="$(printf "%02d" "${idx}")"

  (
    export LOGDIR PROJECT_ROOT DATA_ROOT
    export MICRO_DISPLAY_NUM="${display_num}"
    export MICRO_CUDA_DEV="${cuda_dev}"
    export MICRO_HOME="/gfs/space/private/zjc/home_ptm_refill_${ep}_${RUN_TAG}"
    export MICRO_GRADLE_USER_HOME="${GRADLE_HOME_PREFIX}${gradle_suffix}"
    export MICRO_XORG_ROOT="${XORG_ROOT}"
    export MICRO_HEIGHT=360
    export MICRO_WIDTH=640
    export MICRO_FRAMES_PER_EPISODE=128
    export MICRO_FRAME_STORAGE=mp4
    export MICRO_BATCH_EPISODES=1
    export MICRO_BATCH_TIMEOUT=900
    export MICRO_EP_TIMEOUT=900
    export MICRO_REUSE_ENV=0
    export MICRO_CONTINUE_ON_ERROR=1
    export MICRO_GENERATOR_EPISODE_TIMEOUT=240
    export MICRO_EPISODE_RETRIES=20
    export MICRO_EPISODE_RETRY_SEED_STRIDE=1000003
    nohup bash ptm/scripts/run_episode_refill_range.sh "${worker}" "${ep}" "${ep}" "${RUN_TAG}" </dev/null > "${log}" 2>&1 &
    echo "$! ${worker} ${ep} ${log}" >> "${PIDFILE}"
  )
  echo "REFILL_LAUNCHED ep=${ep} worker=${worker} display=:${display_num} cuda=${cuda_dev} log=${log}"
  idx=$((idx + 1))
done

echo "REFILL_PIDFILE ${PIDFILE}"
