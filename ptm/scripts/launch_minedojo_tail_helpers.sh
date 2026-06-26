#!/usr/bin/env bash
set -euo pipefail

LOGDIR="${LOGDIR:-/gfs/space/private/zjc/logs/ptm_stage1_minedojo_current}"
PIDFILE="${PIDFILE:-${LOGDIR}.pids}"
HELPER_PIDFILE="${HELPER_PIDFILE:-${LOGDIR}.tail_helpers.pids}"
PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
DATA_ROOT="${DATA_ROOT:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/stage1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
TAIL_WORKERS="${TAIL_WORKERS:-1 2 3 4 5 6 7}"

mkdir -p "${LOGDIR}"
touch "${HELPER_PIDFILE}"

helper_alive() {
  local worker_id="$1"
  local line pid token current_worker
  while read -r line; do
    [ -n "${line}" ] || continue
    set -- ${line}
    pid="$1"
    current_worker=""
    shift || true
    for token in "$@"; do
      case "${token}" in
        worker=*) current_worker="${token#worker=}" ;;
      esac
    done
    if [[ "${current_worker}" == "${worker_id}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done < "${HELPER_PIDFILE}"
  return 1
}

worker_enabled() {
  case " ${TAIL_WORKERS} " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

replace_helper_pidfile_line() {
  local worker_id="$1"
  local newline="$2"
  local tmp="${HELPER_PIDFILE}.tmp.$$"
  local line token current_worker
  while read -r line; do
    [ -n "${line}" ] || continue
    set -- ${line}
    current_worker=""
    shift || true
    for token in "$@"; do
      case "${token}" in
        worker=*) current_worker="${token#worker=}" ;;
      esac
    done
    if [[ "${current_worker}" != "${worker_id}" ]]; then
      echo "${line}"
    fi
  done < "${HELPER_PIDFILE}" > "${tmp}"
  echo "${newline}" >> "${tmp}"
  mv "${tmp}" "${HELPER_PIDFILE}"
}

write_helper_script() {
  local script_path="$1"
  local worker="$2"
  local display="$3"
  local tail_base="$4"
  local tail_count="$5"
  cat > "${script_path}" <<SCRIPT
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_ROOT}"
export HOME="\${PTM_HELPER_HOME:-/gfs/space/private/zjc/home_ptm_tail_${worker}}"
export TMPDIR="/gfs/space/private/zjc/tmp"
export GRADLE_USER_HOME="/gfs/space/private/zjc/.gradle_ptm_tail_${worker}"
export JAVA_HOME="/gfs/space/private/zjc/jdks/temurin8"
export PATH="/gfs/space/private/zjc/bin:/gfs/space/private/zjc/jdks/temurin8/bin:/gfs/space/private/zjc/envs/ptm_minedojo/bin:/gfs/space/private/zjc/envs/xvfb_main/bin:\${PATH}"
export LD_LIBRARY_PATH="/gfs/space/private/zjc/nvidia_xorg_580105/NVIDIA-Linux-x86_64-580.105.08-no-compat32:\${PTM_EXTRA_GL_LIBS:-/gfs/space/private/zjc/envs/xvfb_main/lib}:/gfs/space/private/zjc/jdks/temurin8/jre/lib/amd64/server:/gfs/space/private/zjc/jdks/temurin8/jre/lib/amd64:/gfs/space/private/zjc/jdks/temurin8/jre/lib/amd64/xawt\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}"
export RIKKA_XORG_REDIRECT_ROOT="/gfs/space/private/zjc/aptroot/ubuntu_xvfb"
export XKB_CONFIG_ROOT="/gfs/space/private/zjc/aptroot/ubuntu_xvfb/usr/share/X11/xkb"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MINEDOJO_HEADLESS=1
export MINEDOJO_USE_EXISTING_DISPLAY=1
export PYTHONPATH="${PROJECT_ROOT}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${worker}"
export JAVA_TOOL_OPTIONS='-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true'
export DISPLAY="${display}"
if [ ! -S "/tmp/.X11-unix/X\${DISPLAY#:}" ]; then
  echo "display socket not ready: \${DISPLAY}" >&2
  exit 21
fi
"/gfs/space/private/zjc/bin/glx_probe"
echo "RUN_ID=${RUN_ID} WORKER=tail_${worker} RANGE_BASE=${tail_base} RANGE_COUNT=${tail_count} DISPLAY=${display}"
date -Is
python -u -m ptm.data.minedojo_generator \\
  --out "${DATA_ROOT}" \\
  --split train \\
  --num_episodes "${tail_count}" \\
  --episode_offset "${tail_base}" \\
  --schedule_total "3000" \\
  --skip_existing \\
  --frames_per_episode 128 \\
  --families balanced \\
  --backend minedojo \\
  --height 128 \\
  --width 128 \\
  --frame_storage mp4 \\
  --seed 20260622
date -Is
SCRIPT
  chmod +x "${script_path}"
}

while read -r line; do
  [ -n "${line}" ] || continue
  set -- ${line}
  parent_pid="$1"
  shift || true
  worker=""
  base=""
  count=""
  display=""
  for token in "$@"; do
    case "${token}" in
      worker=*) worker="${token#worker=}" ;;
      base=*) base="${token#base=}" ;;
      count=*) count="${token#count=}" ;;
      display=*) display="${token#display=}" ;;
    esac
  done
  [[ "${worker}" =~ ^[0-7]$ ]] || continue
  if ! worker_enabled "${worker}"; then
    echo "skip worker=${worker}: disabled by TAIL_WORKERS=${TAIL_WORKERS}"
    continue
  fi
  [ -n "${base}" ] && [ -n "${count}" ] && [ -n "${display}" ] || continue
  if ! kill -0 "${parent_pid}" 2>/dev/null; then
    echo "skip worker=${worker}: parent not alive"
    continue
  fi

  helper_worker="tail_${worker}"
  if helper_alive "${helper_worker}"; then
    echo "skip worker=${helper_worker}: helper already alive"
    continue
  fi

  head_count=$((count / 2))
  tail_base=$((base + head_count))
  tail_count=$((count - head_count))
  script_path="${LOGDIR}/run_worker_${helper_worker}.sh"
  helper_log="${LOGDIR}/worker_${helper_worker}_range_${tail_base}_${tail_count}_${RUN_ID}.log"
  echo "plan worker=${helper_worker} display=${display} range=${tail_base}-$((tail_base + tail_count - 1)) log=${helper_log}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    continue
  fi

  write_helper_script "${script_path}" "${worker}" "${display}" "${tail_base}" "${tail_count}"
  nohup bash "${script_path}" > "${helper_log}" 2>&1 &
  helper_pid=$!
  replace_helper_pidfile_line "${helper_worker}" "${helper_pid} worker=${helper_worker} parent_worker=${worker} base=${tail_base} count=${tail_count} display=${display} log=${helper_log} run=${script_path}"
  echo "started worker=${helper_worker} pid=${helper_pid}"
done < "${PIDFILE}"
