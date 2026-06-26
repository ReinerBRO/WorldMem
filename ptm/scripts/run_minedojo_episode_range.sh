#!/usr/bin/env bash
set -uo pipefail

WORKER="$1"
START_EP="$2"
END_EP="$3"
RUN_TAG="$4"

LOGDIR="${LOGDIR:-/gfs/space/private/zjc/logs/ptm_minedojo_current}"
PROJECT_ROOT="${PROJECT_ROOT:-/gfs/space/private/zjc/ptm}"
DATA_ROOT="${DATA_ROOT:-/gfs/space/private/zjc/ptm/ptm_minedojo_data/long_1500_360x640}"
MICRO_SPLIT="${MICRO_SPLIT:-train}"
MICRO_FRAMES_PER_EPISODE="${MICRO_FRAMES_PER_EPISODE:-1500}"
MICRO_SCHEDULE_TOTAL="${MICRO_SCHEDULE_TOTAL:-1000}"
MICRO_HEIGHT="${MICRO_HEIGHT:-360}"
MICRO_WIDTH="${MICRO_WIDTH:-640}"
MICRO_FAMILIES="${MICRO_FAMILIES:-balanced}"
MICRO_BACKEND="${MICRO_BACKEND:-minedojo}"
MICRO_FRAME_STORAGE="${MICRO_FRAME_STORAGE:-mp4}"
MICRO_SEED="${MICRO_SEED:-20260622}"
MICRO_HISTORY_LENGTH="${MICRO_HISTORY_LENGTH:-600}"
MICRO_FUTURE_LENGTH="${MICRO_FUTURE_LENGTH:-100}"
MICRO_TEST_STRIDE="${MICRO_TEST_STRIDE:-50}"

case "${WORKER}" in
  0) DISPLAY_NUM=260; CUDA_DEV=0 ;;
  1) DISPLAY_NUM=261; CUDA_DEV=1 ;;
  2) DISPLAY_NUM=262; CUDA_DEV=2 ;;
  3) DISPLAY_NUM=263; CUDA_DEV=3 ;;
  4) DISPLAY_NUM=264; CUDA_DEV=4 ;;
  5) DISPLAY_NUM=265; CUDA_DEV=5 ;;
  6) DISPLAY_NUM=266; CUDA_DEV=6 ;;
  7) DISPLAY_NUM=267; CUDA_DEV=7 ;;
  *)
    if [ -z "${MICRO_DISPLAY_NUM:-}" ] || [ -z "${MICRO_CUDA_DEV:-}" ]; then
      echo "bad worker: ${WORKER}; set MICRO_DISPLAY_NUM and MICRO_CUDA_DEV for worker ids outside 0-7" >&2
      exit 2
    fi
    DISPLAY_NUM="${MICRO_DISPLAY_NUM}"
    CUDA_DEV="${MICRO_CUDA_DEV}"
    ;;
esac

DISPLAY_NUM="${MICRO_DISPLAY_NUM:-${DISPLAY_NUM}}"
CUDA_DEV="${MICRO_CUDA_DEV:-${CUDA_DEV}}"
GRADLE_SUFFIX="${MICRO_GRADLE_SUFFIX:-${WORKER}}"

mkdir -p "${LOGDIR}" "${DATA_ROOT}/${MICRO_SPLIT}"
echo "MINEDOJO_RANGE_BOOT tag=${RUN_TAG} split=${MICRO_SPLIT} worker=${WORKER} range=${START_EP}-${END_EP} display=:${DISPLAY_NUM} cuda=${CUDA_DEV} time=$(date -Is)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found in PATH=${PATH}" >&2
  exit 2
fi
PCI_ID="$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader,nounits | awk -F, -v idx="${CUDA_DEV}" '$1 + 0 == idx { gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit }')"
if [[ "${PCI_ID}" =~ ^([0-9A-Fa-f]{4,8}:)?([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})[.]([0-9A-Fa-f])$ ]]; then
  BUSID="PCI:$((16#${BASH_REMATCH[2]})):$((16#${BASH_REMATCH[3]})):$((16#${BASH_REMATCH[4]}))"
else
  echo "could not resolve PCI BusID for CUDA device ${CUDA_DEV}: ${PCI_ID}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}" || exit 1
export HOME="${MICRO_HOME:-/gfs/space/private/zjc/home_ptm_minedojo_${GRADLE_SUFFIX}}"
export TMPDIR="${MICRO_TMPDIR:-/gfs/space/private/zjc/tmp}"
export GRADLE_USER_HOME="${MICRO_GRADLE_USER_HOME:-/gfs/space/private/zjc/.gradle_ptm_minedojo_${GRADLE_SUFFIX}}"
mkdir -p "${HOME}" "${TMPDIR}" "${GRADLE_USER_HOME}"
cat > "${GRADLE_USER_HOME}/gradle.properties" <<'EOF'
org.gradle.daemon=false
org.gradle.parallel=false
org.gradle.workers.max=1
org.gradle.jvmargs=-Xmx1024m -XX:MaxMetaspaceSize=512m -Dfile.encoding=UTF-8
EOF
GRADLE_CACHE_SOURCE="${MICRO_GRADLE_CACHE_SOURCE:-/gfs/space/private/zjc/.gradle_ptm_worker_26}"
GRADLE_DIST_REL="wrapper/dists/gradle-4.10.2-all"
if [ ! -e "${GRADLE_USER_HOME}/${GRADLE_DIST_REL}" ] && [ -d "${GRADLE_CACHE_SOURCE}/${GRADLE_DIST_REL}" ]; then
  mkdir -p "${GRADLE_USER_HOME}/wrapper/dists"
  ln -s "${GRADLE_CACHE_SOURCE}/${GRADLE_DIST_REL}" "${GRADLE_USER_HOME}/wrapper/dists/gradle-4.10.2-all"
fi

# ForgeGradle mutates caches/minecraft during applyBinaryPatches. Sharing it
# across concurrent MineDojo workers can corrupt Pack200 intermediates.
MINECRAFT_CACHE_REL="caches/minecraft"
if [ ! -e "${GRADLE_USER_HOME}/${MINECRAFT_CACHE_REL}" ] && [ -d "${GRADLE_CACHE_SOURCE}/${MINECRAFT_CACHE_REL}" ]; then
  mkdir -p "${GRADLE_USER_HOME}/caches"
  cp -a "${GRADLE_CACHE_SOURCE}/${MINECRAFT_CACHE_REL}" "${GRADLE_USER_HOME}/caches/"
fi
for cache_rel in \
  "caches/modules-2" \
  "caches/jars-3" \
  "caches/transforms-1"
do
  if [ ! -e "${GRADLE_USER_HOME}/${cache_rel}" ] && [ -e "${GRADLE_CACHE_SOURCE}/${cache_rel}" ]; then
    mkdir -p "$(dirname "${GRADLE_USER_HOME}/${cache_rel}")"
    ln -s "${GRADLE_CACHE_SOURCE}/${cache_rel}" "${GRADLE_USER_HOME}/${cache_rel}"
  fi
done
NVIDIA_XORG_ROOT="/gfs/space/private/zjc/nvidia_xorg_580105"
NVIDIA_RUNLIB="${NVIDIA_XORG_ROOT}/NVIDIA-Linux-x86_64-580.105.08-no-compat32"
XORG_ROOT="${MICRO_XORG_ROOT:-/gfs/space/private/zjc/aptroot/jammy_xorg/root}"
XORG_EXTRA_LD_LIBRARY_PATH="${MICRO_XORG_EXTRA_LD_LIBRARY_PATH-/gfs/space/private/zjc/envs/xvfb_main/lib}"
XORG_LIB_DIR="${XORG_ROOT}/usr/lib/x86_64-linux-gnu"
RUNTIME_X11_LIB_DIR="${MICRO_RUNTIME_X11_LIB_DIR:-/gfs/space/private/zjc/ptm/runtime/x11/root/usr/lib/x86_64-linux-gnu}"
RUNTIME_X11_LD_LIBRARY_PATH=""
if [ -d "${RUNTIME_X11_LIB_DIR}" ]; then
  RUNTIME_X11_LD_LIBRARY_PATH="${RUNTIME_X11_LIB_DIR}"
fi
XORG_MODULE_DIR="${XORG_ROOT}/usr/lib/xorg/modules"
XORG_BIN="${XORG_ROOT}/usr/lib/xorg/Xorg"
export JAVA_HOME="/gfs/space/private/zjc/jdks/temurin8"
export PATH="/gfs/space/private/zjc/bin:/gfs/space/private/zjc/jdks/temurin8/bin:/gfs/space/private/zjc/envs/ptm_minedojo/bin:/gfs/space/private/zjc/envs/xvfb_main/bin:${PATH}"
export LD_LIBRARY_PATH="${NVIDIA_RUNLIB}${RUNTIME_X11_LD_LIBRARY_PATH:+:${RUNTIME_X11_LD_LIBRARY_PATH}}:${XORG_LIB_DIR}${XORG_EXTRA_LD_LIBRARY_PATH:+:${XORG_EXTRA_LD_LIBRARY_PATH}}:/gfs/space/private/zjc/jdks/temurin8/jre/lib/amd64/server:/gfs/space/private/zjc/jdks/temurin8/jre/lib/amd64:/gfs/space/private/zjc/jdks/temurin8/jre/lib/amd64/xawt${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RIKKA_XORG_REDIRECT_ROOT="${MICRO_XORG_REDIRECT_ROOT:-${XORG_ROOT}}"
export XKB_CONFIG_ROOT="${MICRO_XKB_CONFIG_ROOT:-${XORG_ROOT}/usr/share/X11/xkb}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MINEDOJO_HEADLESS=1
export MINEDOJO_USE_EXISTING_DISPLAY=1
export PYTHONPATH="${PROJECT_ROOT}"
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_DEV}"
export JAVA_TOOL_OPTIONS="-Dorg.lwjgl.opengl.Display.allowSoftwareOpenGL=true"
export DISPLAY=":${DISPLAY_NUM}"
REQUIRED_NATIVE_LIBS=(libXrandr.so.2 libXcursor.so.1 libXxf86vm.so.1 libjawt.so)
if ! python - "${REQUIRED_NATIVE_LIBS[@]}" <<'PY'
import ctypes
import sys

missing = []
for lib in sys.argv[1:]:
    try:
        ctypes.CDLL(lib)
    except OSError as exc:
        missing.append(f"{lib}: {exc}")
if missing:
    print("MINEDOJO_NATIVE_LIB_MISSING")
    for item in missing:
        print(item)
    sys.exit(1)
PY
then
  echo "Native runtime preflight failed. LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >&2
  exit 23
fi
EP_TIMEOUT="${MICRO_EP_TIMEOUT:-7200}"
BATCH_EPISODES="${MICRO_BATCH_EPISODES:-1}"
BATCH_TIMEOUT="${MICRO_BATCH_TIMEOUT:-}"
MICRO_LOCK_STALE_SECONDS="${MICRO_LOCK_STALE_SECONDS:-7200}"
MICRO_CONTINUE_ON_ERROR="${MICRO_CONTINUE_ON_ERROR:-1}"
MICRO_REUSE_ENV="${MICRO_REUSE_ENV:-1}"
MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE="${MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE:-200}"
MICRO_GENERATOR_EPISODE_TIMEOUT="${MICRO_GENERATOR_EPISODE_TIMEOUT:-5400}"
MICRO_EPISODE_RETRIES="${MICRO_EPISODE_RETRIES:-5}"
MICRO_EPISODE_RETRY_SEED_STRIDE="${MICRO_EPISODE_RETRY_SEED_STRIDE:-1000003}"
GENERATOR_EXTRA_ARGS=(
  --lock_stale_seconds "${MICRO_LOCK_STALE_SECONDS}"
  --episode_timeout_seconds "${MICRO_GENERATOR_EPISODE_TIMEOUT}"
  --episode_retries "${MICRO_EPISODE_RETRIES}"
  --episode_retry_seed_stride "${MICRO_EPISODE_RETRY_SEED_STRIDE}"
)
if [ "${MICRO_CONTINUE_ON_ERROR}" = "1" ]; then
  GENERATOR_EXTRA_ARGS+=(--continue_on_error)
fi
if [ "${MICRO_REUSE_ENV}" = "1" ]; then
  GENERATOR_EXTRA_ARGS+=(--reuse_env --fast_reset_random_teleport_range "${MICRO_FAST_RESET_RANDOM_TELEPORT_RANGE}")
fi
if [ "${MICRO_DISABLE_EPISODE_LOCKS:-0}" = "1" ]; then
  GENERATOR_EXTRA_ARGS+=(--no_episode_locks)
fi
if [ "${MICRO_DISABLE_ATOMIC_WRITE:-0}" = "1" ]; then
  GENERATOR_EXTRA_ARGS+=(--no_atomic_write)
fi

if ! [[ "${BATCH_EPISODES}" =~ ^[0-9]+$ ]] || [ "${BATCH_EPISODES}" -lt 1 ]; then
  echo "bad MICRO_BATCH_EPISODES=${BATCH_EPISODES}; must be a positive integer" >&2
  exit 2
fi

XORG_PID=""
cleanup() {
  if [ -n "${XORG_PID}" ]; then
    kill "${XORG_PID}" >/dev/null 2>&1 || true
    wait "${XORG_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  if [ ! -s "${LOGDIR}/xorg_worker_${WORKER}.conf" ]; then
    cat > "${LOGDIR}/xorg_worker_${WORKER}.conf" <<EOF
Section "ServerLayout"
    Identifier "Layout0"
    Screen 0 "Screen0"
EndSection
Section "Device"
    Identifier "Device0"
    Driver "nvidia"
    BusID "${BUSID}"
    Option "AllowEmptyInitialConfiguration" "true"
EndSection
Section "Screen"
    Identifier "Screen0"
    Device "Device0"
    DefaultDepth 24
EndSection
EOF
  fi
  MODULE_PATH="${NVIDIA_XORG_ROOT}/xorg_modules,${NVIDIA_XORG_ROOT}/xorg_modules/drivers,${NVIDIA_XORG_ROOT}/xorg_modules/extensions,${XORG_MODULE_DIR}"
  LD_PRELOAD="/gfs/space/private/zjc/bin/libxorg_path_redirect.so:/gfs/space/private/zjc/bin/libexecve_xkb_redirect.so" \
    "${XORG_BIN}" ":${DISPLAY_NUM}" \
    -config "${LOGDIR}/xorg_worker_${WORKER}.conf" \
    -modulepath "${MODULE_PATH}" \
    -logfile "${LOGDIR}/xorg_worker_${WORKER}_display_${DISPLAY_NUM}_${RUN_TAG}.log" \
    -noreset +extension GLX -nolisten tcp -ac > "${LOGDIR}/xorg_worker_${WORKER}_display_${DISPLAY_NUM}_${RUN_TAG}.stdout" 2>&1 &
  XORG_PID=$!
  for _ in $(seq 1 240); do
    [ -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ] && break
    if ! kill -0 "${XORG_PID}" 2>/dev/null; then
      echo "Xorg exited before socket ready worker=${WORKER} display=${DISPLAY}" >&2
      tail -120 "${LOGDIR}/xorg_worker_${WORKER}_display_${DISPLAY_NUM}_${RUN_TAG}.log" 2>/dev/null || true
      exit 20
    fi
    sleep 0.5
  done
fi

if [ ! -S "/tmp/.X11-unix/X${DISPLAY_NUM}" ]; then
  echo "display socket not ready: ${DISPLAY}" >&2
  exit 21
fi

"/gfs/space/private/zjc/bin/glx_probe" || exit 22
echo "MINEDOJO_RANGE_START tag=${RUN_TAG} split=${MICRO_SPLIT} worker=${WORKER} range=${START_EP}-${END_EP} display=${DISPLAY} busid=${BUSID} time=$(date -Is)"

complete_episode() {
  local ep="$1"
  local d="${DATA_ROOT}/${MICRO_SPLIT}/episode_$(printf "%06d" "${ep}")"
  [ -s "${d}/frames.mp4" ] && [ -s "${d}/metadata.json" ]
}

missing_episode_list() {
  local start="$1"
  local end="$2"
  local ep
  for ep in $(seq "${start}" "${end}"); do
    if ! complete_episode "${ep}"; then
      printf "%s " "${ep}"
    fi
  done
}

count_words() {
  local count=0
  local item
  for item in "$@"; do
    count=$((count + 1))
  done
  printf "%s" "${count}"
}

run_generator_range() {
  local start="$1"
  local end="$2"
  local timeout_seconds="$3"
  local num_episodes=$((end - start + 1))
  timeout "${timeout_seconds}s" python -u -m ptm.data.minedojo_generator \
    --out "${DATA_ROOT}" \
    --split "${MICRO_SPLIT}" \
    --num_episodes "${num_episodes}" \
    --episode_offset "${start}" \
    --schedule_total "${MICRO_SCHEDULE_TOTAL}" \
    --skip_existing \
    --frames_per_episode "${MICRO_FRAMES_PER_EPISODE}" \
    --families "${MICRO_FAMILIES}" \
    --backend "${MICRO_BACKEND}" \
    --height "${MICRO_HEIGHT}" \
    --width "${MICRO_WIDTH}" \
    --frame_storage "${MICRO_FRAME_STORAGE}" \
    --history_length "${MICRO_HISTORY_LENGTH}" \
    --future_length "${MICRO_FUTURE_LENGTH}" \
    --test_stride "${MICRO_TEST_STRIDE}" \
    --seed "${MICRO_SEED}" \
    "${GENERATOR_EXTRA_ARGS[@]}"
}

while true; do
  made=0
  missing_eps="$(missing_episode_list "${START_EP}" "${END_EP}")"
  # shellcheck disable=SC2086
  missing=$(count_words ${missing_eps})
  if [ "${missing}" -eq 0 ]; then
    echo "MINEDOJO_RANGE_PASS worker=${WORKER} split=${MICRO_SPLIT} range=${START_EP}-${END_EP} missing_seen=0 made=0 time=$(date -Is)"
    break
  fi

  if [ "${BATCH_EPISODES}" -le 1 ]; then
    for ep in ${missing_eps}; do
      echo "MINEDOJO_EP_START worker=${WORKER} split=${MICRO_SPLIT} ep=${ep} time=$(date -Is)"
      run_generator_range "${ep}" "${ep}" "${EP_TIMEOUT}"
      rc=$?
      if complete_episode "${ep}"; then
        made=$((made + 1))
        echo "MINEDOJO_EP_DONE worker=${WORKER} split=${MICRO_SPLIT} ep=${ep} rc=${rc} time=$(date -Is)"
      else
        echo "MINEDOJO_EP_FAIL worker=${WORKER} split=${MICRO_SPLIT} ep=${ep} rc=${rc} time=$(date -Is)"
      fi
    done
  else
    batch_start="${START_EP}"
    while [ "${batch_start}" -le "${END_EP}" ]; do
      batch_end=$((batch_start + BATCH_EPISODES - 1))
      if [ "${batch_end}" -gt "${END_EP}" ]; then
        batch_end="${END_EP}"
      fi
      batch_missing_eps="$(missing_episode_list "${batch_start}" "${batch_end}")"
      # shellcheck disable=SC2086
      batch_missing=$(count_words ${batch_missing_eps})
      if [ "${batch_missing}" -eq 0 ]; then
        batch_start=$((batch_end + 1))
        continue
      fi
      if [ -n "${BATCH_TIMEOUT}" ]; then
        timeout_seconds="${BATCH_TIMEOUT}"
      else
        timeout_seconds=$((EP_TIMEOUT * batch_missing))
      fi
      if [ "${timeout_seconds}" -lt "${EP_TIMEOUT}" ]; then
        timeout_seconds="${EP_TIMEOUT}"
      fi
      echo "MINEDOJO_BATCH_START worker=${WORKER} split=${MICRO_SPLIT} range=${batch_start}-${batch_end} missing=${batch_missing} timeout=${timeout_seconds}s time=$(date -Is)"
      run_generator_range "${batch_start}" "${batch_end}" "${timeout_seconds}"
      rc=$?
      batch_made=0
      for ep in ${batch_missing_eps}; do
        if complete_episode "${ep}"; then
          made=$((made + 1))
          batch_made=$((batch_made + 1))
          echo "MINEDOJO_EP_DONE worker=${WORKER} split=${MICRO_SPLIT} ep=${ep} rc=${rc} batch=${batch_start}-${batch_end} time=$(date -Is)"
        else
          echo "MINEDOJO_EP_FAIL worker=${WORKER} split=${MICRO_SPLIT} ep=${ep} rc=${rc} batch=${batch_start}-${batch_end} time=$(date -Is)"
        fi
      done
      echo "MINEDOJO_BATCH_DONE worker=${WORKER} split=${MICRO_SPLIT} range=${batch_start}-${batch_end} missing=${batch_missing} made=${batch_made} rc=${rc} time=$(date -Is)"
      batch_start=$((batch_end + 1))
    done
  fi

  echo "MINEDOJO_RANGE_PASS worker=${WORKER} split=${MICRO_SPLIT} range=${START_EP}-${END_EP} missing_seen=${missing} made=${made} time=$(date -Is)"
  if [ "${made}" -eq 0 ]; then
    echo "MINEDOJO_NO_PROGRESS worker=${WORKER} split=${MICRO_SPLIT} range=${START_EP}-${END_EP} time=$(date -Is)" >&2
    sleep 10
  fi
  sleep 1
done

echo "MINEDOJO_RANGE_DONE tag=${RUN_TAG} split=${MICRO_SPLIT} worker=${WORKER} range=${START_EP}-${END_EP} time=$(date -Is)"
