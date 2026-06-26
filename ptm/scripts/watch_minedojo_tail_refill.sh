#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
LOG_DIR="${LOG_DIR:-/gfs/space/private/zjc/logs}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/ptm_minedojo_data/stage1/train}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-3000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_DIR}/ptm_stage1_minedojo_current}"
MAIN_PID_FILE="${MAIN_PID_FILE:-${RUN_LOG_DIR}.pids}"
TAIL_PID_FILE="${TAIL_PID_FILE:-${RUN_LOG_DIR}.tail_helpers.pids}"
LOCK_DIR="${LOCK_DIR:-${LOG_DIR}/ptm_watch_minedojo_tail_refill.lock}"
STATUS_FILE="${STATUS_FILE:-${LOG_DIR}/ptm_watch_minedojo_tail_refill.status}"
TAIL_WORKERS="${TAIL_WORKERS:-1 2 3 4 5 6 7}"

mkdir -p "${LOG_DIR}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "tail refill watcher already locked at ${LOCK_DIR}"
  exit 0
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

log() {
  local msg="$*"
  printf '[%s] %s\n' "$(date -Is)" "${msg}"
  printf '[%s] %s\n' "$(date -Is)" "${msg}" > "${STATUS_FILE}"
}

count_mp4() {
  find "${DATA_ROOT}" -type f -name 'frames.mp4' 2>/dev/null | wc -l | tr -d ' '
}

pid_stats() {
  local file="$1"
  local total=0
  local alive=0
  local dead=""
  if [[ -f "${file}" ]]; then
    while read -r pid _rest; do
      [[ -n "${pid}" && "${pid}" != \#* ]] || continue
      total=$((total + 1))
      if kill -0 "${pid}" 2>/dev/null; then
        alive=$((alive + 1))
      else
        dead="${dead} ${pid}"
      fi
    done < "${file}"
  fi
  printf '%s|%s|%s' "${alive}" "${total}" "${dead:-none}"
}

worker_enabled() {
  case " ${TAIL_WORKERS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

tail_expected_count() {
  set -- ${TAIL_WORKERS}
  printf '%s' "$#"
}

compact_tail_pidfile() {
  [[ -f "${TAIL_PID_FILE}" ]] || return 0

  local tmp="${TAIL_PID_FILE}.tmp.$$"
  local line pid token current_worker worker_num
  while read -r line; do
    [[ -n "${line}" ]] || continue
    set -- ${line}
    pid="$1"
    current_worker=""
    shift || true
    for token in "$@"; do
      case "${token}" in
        worker=*) current_worker="${token#worker=}" ;;
      esac
    done
    worker_num="${current_worker#tail_}"
    [[ "${current_worker}" == tail_* ]] || continue
    worker_enabled "${worker_num}" || continue
    kill -0 "${pid}" 2>/dev/null || continue
    echo "${line}"
  done < "${TAIL_PID_FILE}" > "${tmp}"
  mv "${tmp}" "${TAIL_PID_FILE}"
}

refill_tail_if_needed() {
  local stats alive total dead expected
  compact_tail_pidfile
  stats="$(pid_stats "${TAIL_PID_FILE}")"
  alive="${stats%%|*}"
  stats="${stats#*|}"
  total="${stats%%|*}"
  dead="${stats#*|}"
  expected="$(tail_expected_count)"
  if [[ "${alive}" -lt "${expected}" ]]; then
    log "refilling tail helpers: alive=${alive}/${expected} tracked=${total} dead=${dead} workers=${TAIL_WORKERS}"
    cd "${PROJECT_ROOT}"
    TAIL_WORKERS="${TAIL_WORKERS}" RUN_ID="tailboost_refill_$(date +%Y%m%d_%H%M%S)" bash ptm/scripts/launch_minedojo_tail_helpers.sh
  fi
}

main() {
  log "tail refill watcher started data=${DATA_ROOT} run_log_dir=${RUN_LOG_DIR}"
  while true; do
    local mp4 main_stats tail_stats
    mp4="$(count_mp4)"
    compact_tail_pidfile
    main_stats="$(pid_stats "${MAIN_PID_FILE}")"
    tail_stats="$(pid_stats "${TAIL_PID_FILE}")"
    log "data mp4=${mp4}/${EXPECTED_EPISODES} main=${main_stats} tail=${tail_stats}"
    if [[ "${mp4}" -ge "${EXPECTED_EPISODES}" ]]; then
      log "data target reached; tail refill watcher exiting"
      break
    fi
    refill_tail_if_needed
    sleep "${POLL_SECONDS}"
  done
}

main "$@"
