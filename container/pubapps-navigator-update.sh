#!/bin/bash
# pubapps-navigator-update.sh
# Unattended security updates for the Navigator container on UF PubApps.
#
# Rebuilds the image on a freshly pulled base with current OS packages,
# verifies the candidate on a spare port, and swaps it in only when nobody has
# used the service recently. The swap rolls back automatically when the new
# container does not become healthy.
#
# A systemd user timer runs this script daily. The script itself does work only
# when the last successful update is older than the interval, so a day that is
# too busy or a VM that was down simply retries on the next run.
#
# Usage:
#   ./pubapps-navigator-update.sh [run|check|install|uninstall|status] [options]
#
# Actions:
#   run        Update if due and idle (default).
#   check      Report whether an update is due and whether the service is idle.
#   install    Install and start the systemd user timer.
#   uninstall  Stop and remove the timer. State and logs stay.
#   status     Show the timer, last success, and recent log lines.
#
# Options:
#   --force              Ignore the update interval. The idle check still applies.
#   --dry-run            Build and verify the candidate image but do not swap it.
#   --git-pull           Run git pull --ff-only before building.
#   --interval-days N    Days between successful updates. Default 14.
#   --idle-minutes N     Minutes without log or data writes that count as idle. Default 60.
#   --port PORT          Port of the running service. Default 8080.
#
# Environment overrides: AAU_INTERVAL_DAYS, AAU_IDLE_MINUTES, AAU_GIT_PULL,
# AAU_VERIFY_PORT, AAU_MIN_FREE_GB, AAU_PROJECT_ROOT, AAU_IMAGE_REPO,
# AAU_SERVICE_NAME. The last two exist so the swap can be tested against a
# throwaway service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${AAU_PROJECT_ROOT:-$(dirname "$SCRIPT_DIR")}"

IMAGE_REPO="${AAU_IMAGE_REPO:-localhost/agents-among-us-navigator}"
SERVICE_NAME="${AAU_SERVICE_NAME:-agents-among-us}"
QUADLET_FILE="${HOME}/.config/containers/systemd/${SERVICE_NAME}.container"
UNIT_DIR="${HOME}/.config/systemd/user"
UPDATE_UNIT="aau-navigator-update"
STATE_DIR="${XDG_STATE_HOME:-${HOME}/.local/state}/aau-navigator-update"
LOG_FILE="${STATE_DIR}/update.log"
BUILD_LOG="${STATE_DIR}/build.log"
STAMP_FILE="${STATE_DIR}/last-success"

PORT="${PORT:-8080}"
VERIFY_PORT="${AAU_VERIFY_PORT:-18080}"
INTERVAL_DAYS="${AAU_INTERVAL_DAYS:-14}"
IDLE_MINUTES="${AAU_IDLE_MINUTES:-60}"
GIT_PULL="${AAU_GIT_PULL:-0}"
MIN_FREE_GB="${AAU_MIN_FREE_GB:-5}"
ACTION="run"
FORCE=0
DRY_RUN=0
VERIFY_NAME=""

usage() {
  sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
}

die() {
  echo "Error: $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    run|check|install|uninstall|status) ACTION="$1"; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --git-pull) GIT_PULL=1; shift ;;
    --interval-days) INTERVAL_DAYS="$2"; shift 2 ;;
    --idle-minutes) IDLE_MINUTES="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

for value in "$INTERVAL_DAYS" "$IDLE_MINUTES" "$PORT" "$VERIFY_PORT" "$MIN_FREE_GB"; do
  case "$value" in
    ''|*[!0-9]*) die "numeric option expected, got '${value}'" ;;
  esac
done

# --- logging and state ------------------------------------------------------

log() {
  mkdir -p "$STATE_DIR"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

trim_log() {
  if [ -f "$LOG_FILE" ] && [ "$(wc -l < "$LOG_FILE")" -gt 2000 ]; then
    tail -n 1000 "$LOG_FILE" > "${LOG_FILE}.tmp" && mv "${LOG_FILE}.tmp" "$LOG_FILE"
  fi
}

fmt_epoch() {
  if [ "$1" -eq 0 ]; then
    echo "never"
  else
    date -d "@$1" '+%Y-%m-%d %H:%M' 2>/dev/null || date -r "$1" '+%Y-%m-%d %H:%M'
  fi
}

last_success() {
  if [ -f "$STAMP_FILE" ]; then cat "$STAMP_FILE"; else echo 0; fi
}

# Due when the last success is older than the interval. The 6 hour slack keeps a
# run that lands slightly early from waiting another full day.
update_due() {
  local last now slack=21600
  last="$(last_success)"
  now="$(date +%s)"
  [ $((now - last)) -ge $((INTERVAL_DAYS * 86400 - slack)) ]
}

# --- activity detection -----------------------------------------------------

# Host directories mounted at /app/logs and /app/frontend/data. The Quadlet
# file is the source of truth; fall back to the deploy script defaults.
activity_dirs() {
  local dirs=""
  if [ -f "$QUADLET_FILE" ]; then
    dirs="$(sed -n -E 's#^Volume=([^:]+):/app/(logs|frontend/data)(:.*)?$#\1#p' "$QUADLET_FILE")"
  fi
  if [ -z "$dirs" ]; then
    dirs="${HOME}/agents-among-us/logs
${HOME}/agents-among-us/frontend/data"
  fi
  printf '%s\n' "$dirs"
}

# Prints one recently written file and returns 0 when the service is in use.
# File writes are the signal because a game process can be hung for weeks while
# still reported as running, and a hung game must not block security updates.
recent_activity() {
  local dir hit
  while IFS= read -r dir; do
    [ -d "$dir" ] || continue
    hit="$(find "$dir" -type f -mmin "-${IDLE_MINUTES}" -print -quit 2>/dev/null || true)"
    if [ -n "$hit" ]; then
      printf '%s\n' "$hit"
      return 0
    fi
  done < <(activity_dirs)
  return 1
}

game_reported_running() {
  local body
  body="$(curl -fsS --max-time 5 "http://localhost:${PORT}/api/game_status" 2>/dev/null || true)"
  printf '%s' "$body" | grep -q '"running": *true'
}

# --- container helpers ------------------------------------------------------

wait_health() {
  local url="$1" limit="$2" waited=0
  while [ "$waited" -lt "$limit" ]; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  return 1
}

# Hash of the OS package list, the Python package list, and the git commit the
# image was built from. Equal hashes mean a rebuild changed nothing.
image_fingerprint() {
  local image="$1" commit pkgs
  commit="$(podman image inspect --format '{{ index .Labels "aau.commit" }}' "$image" 2>/dev/null || true)"
  pkgs="$(podman run --rm --entrypoint sh "$image" -c \
    'dpkg-query -W; /app/.venv/bin/python -c "import importlib.metadata as m; print(*sorted(d.metadata[\"Name\"].lower() + \"==\" + d.version for d in m.distributions()), sep=chr(10))"')" || return 1
  printf '%s\n%s\n' "$commit" "$pkgs" | sha256sum | cut -d' ' -f1
}

free_gb_for_images() {
  local root
  root="$(podman info --format '{{.Store.GraphRoot}}' 2>/dev/null || echo "$HOME")"
  df -Pk "$root" | awk 'NR==2 {print int($4 / 1048576)}'
}

verify_candidate() {
  VERIFY_NAME="aau-verify-$$"
  podman rm -f -t 2 "$VERIFY_NAME" >/dev/null 2>&1 || true
  if ! podman run -d --name "$VERIFY_NAME" -e "PORT=${VERIFY_PORT}" -e APP_MODE=navigator \
        -e LLM_MODE=LOCAL -p "${VERIFY_PORT}:${VERIFY_PORT}" "${IMAGE_REPO}:candidate" >/dev/null 2>&1; then
    log "verify: could not start the candidate (is port ${VERIFY_PORT} in use?)"
    return 1
  fi
  if ! wait_health "http://localhost:${VERIFY_PORT}/api/health" 90; then
    log "verify: health check failed"
    podman logs --tail 20 "$VERIFY_NAME" >> "$LOG_FILE" 2>&1 || true
    return 1
  fi
  if ! podman exec -w /app "$VERIFY_NAME" /app/.venv/bin/python main.py --help >/dev/null 2>&1; then
    log "verify: main.py failed to import"
    return 1
  fi
}

cleanup() {
  if [ -n "$VERIFY_NAME" ]; then
    podman rm -f -t 2 "$VERIFY_NAME" >/dev/null 2>&1 || true
  fi
  podman rmi "${IMAGE_REPO}:candidate" >/dev/null 2>&1 || true
}

# --- actions ----------------------------------------------------------------

do_check() {
  local last hit
  last="$(last_success)"
  echo "last success:   $(fmt_epoch "$last")"
  if update_due; then
    echo "update due:     yes (interval ${INTERVAL_DAYS} days)"
  else
    echo "update due:     no (interval ${INTERVAL_DAYS} days)"
  fi
  if hit="$(recent_activity)"; then
    echo "idle:           no, written within ${IDLE_MINUTES} min: ${hit}"
  else
    echo "idle:           yes (no writes within ${IDLE_MINUTES} min)"
  fi
  if game_reported_running; then
    echo "game process:   reported running by the app"
  else
    echo "game process:   none reported"
  fi
  echo "service:        $(systemctl --user is-active "$SERVICE_NAME" 2>&1 || true)"
  echo "running image:  $(podman inspect "$SERVICE_NAME" --format '{{.ImageName}} {{.Image}}' 2>/dev/null || echo 'container not found')"
}

do_run() {
  local hit free before after commit

  mkdir -p "$STATE_DIR"
  trim_log
  exec 9>"${STATE_DIR}/lock"
  if ! flock -n 9; then
    log "another update is already running; exiting"
    exit 0
  fi
  trap cleanup EXIT

  if [ "$FORCE" -eq 0 ] && ! update_due; then
    log "not due: last success $(fmt_epoch "$(last_success)"), interval ${INTERVAL_DAYS} days"
    exit 0
  fi

  if [ "$DRY_RUN" -eq 0 ]; then
    if hit="$(recent_activity)"; then
      log "deferred: activity within the last ${IDLE_MINUTES} minutes (${hit}); retrying on the next run"
      exit 0
    fi
    if game_reported_running; then
      log "note: the app reports a running game with no recent output; the restart will end it"
    fi
  fi

  free="$(free_gb_for_images)"
  if [ "${free:-0}" -lt "$MIN_FREE_GB" ]; then
    log "aborted: only ${free} GB free for images, need ${MIN_FREE_GB} GB (try podman image prune -f)"
    exit 1
  fi

  if [ "$GIT_PULL" = "1" ]; then
    log "git pull --ff-only"
    git -C "$PROJECT_ROOT" pull --ff-only >> "$LOG_FILE" 2>&1 || { log "aborted: git pull failed"; exit 1; }
  fi
  commit="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

  log "building candidate image from commit ${commit} (pull base, no cache)"
  if ! (cd "$PROJECT_ROOT" && nice -n 10 podman build --pull=newer --no-cache \
        --label "aau.commit=${commit}" -t "${IMAGE_REPO}:candidate" \
        -f Containerfile.navigator .) > "$BUILD_LOG" 2>&1; then
    log "aborted: build failed; last lines of ${BUILD_LOG}:"
    tail -n 15 "$BUILD_LOG" | tee -a "$LOG_FILE"
    exit 1
  fi

  before="$(image_fingerprint "${IMAGE_REPO}:latest" || echo unknown-current)"
  after="$(image_fingerprint "${IMAGE_REPO}:candidate")" || { log "aborted: could not inspect the candidate"; exit 1; }
  if [ "$before" = "$after" ]; then
    log "no package or code changes since the running image; keeping it"
    if [ "$DRY_RUN" -eq 0 ]; then date +%s > "$STAMP_FILE"; fi
    exit 0
  fi

  log "candidate differs from the running image; verifying on port ${VERIFY_PORT}"
  verify_candidate || { log "aborted: candidate failed verification; the running service is unchanged"; exit 1; }

  if [ "$DRY_RUN" -eq 1 ]; then
    log "dry run: candidate verified; not swapping"
    exit 0
  fi

  # The build and verification take minutes, so check again before restarting.
  if hit="$(recent_activity)"; then
    log "deferred: activity appeared while building (${hit}); retrying on the next run"
    exit 0
  fi

  log "swapping: tagging current image as rollback and restarting ${SERVICE_NAME}"
  podman tag "${IMAGE_REPO}:latest" "${IMAGE_REPO}:rollback"
  podman tag "${IMAGE_REPO}:candidate" "${IMAGE_REPO}:latest"
  systemctl --user restart "$SERVICE_NAME"

  if wait_health "http://localhost:${PORT}/api/health" 120; then
    date +%s > "$STAMP_FILE"
    podman image prune -f >/dev/null 2>&1 || true
    log "update complete: ${SERVICE_NAME} is healthy on the new image"
    exit 0
  fi

  log "new container failed its health check; rolling back"
  podman tag "${IMAGE_REPO}:rollback" "${IMAGE_REPO}:latest"
  systemctl --user restart "$SERVICE_NAME"
  if wait_health "http://localhost:${PORT}/api/health" 120; then
    log "rollback complete; ${SERVICE_NAME} is healthy on the previous image"
  else
    log "ROLLBACK ALSO FAILED: manual intervention required (journalctl --user -u ${SERVICE_NAME})"
  fi
  exit 1
}

do_install() {
  mkdir -p "$UNIT_DIR" "$STATE_DIR"

  cat > "${UNIT_DIR}/${UPDATE_UNIT}.service" <<EOF
# Auto-generated by pubapps-navigator-update.sh install; do not edit by hand.
[Unit]
Description=Agents Among Us navigator security update (rebuild and restart when idle)

[Service]
Type=oneshot
Environment=AAU_PROJECT_ROOT=${PROJECT_ROOT}
Environment=AAU_INTERVAL_DAYS=${INTERVAL_DAYS}
Environment=AAU_IDLE_MINUTES=${IDLE_MINUTES}
Environment=AAU_GIT_PULL=${GIT_PULL}
Environment=PORT=${PORT}
ExecStart=${SCRIPT_DIR}/pubapps-navigator-update.sh run
TimeoutStartSec=2h
EOF

  cat > "${UNIT_DIR}/${UPDATE_UNIT}.timer" <<EOF
# Auto-generated by pubapps-navigator-update.sh install; do not edit by hand.
[Unit]
Description=Daily trigger for the Agents Among Us navigator update (acts every ${INTERVAL_DAYS} days)

[Timer]
OnCalendar=*-*-* 03:30:00
RandomizedDelaySec=20min
Persistent=true

[Install]
WantedBy=timers.target
EOF

  loginctl enable-linger "$USER" >/dev/null 2>&1 || true
  systemctl --user daemon-reload
  systemctl --user enable --now "${UPDATE_UNIT}.timer"
  echo "Installed. The timer fires daily at about 03:30; the update runs when ${INTERVAL_DAYS} days have passed and the service has been idle for ${IDLE_MINUTES} minutes."
  systemctl --user list-timers "${UPDATE_UNIT}.timer" --no-pager
}

do_uninstall() {
  systemctl --user disable --now "${UPDATE_UNIT}.timer" 2>/dev/null || true
  rm -f "${UNIT_DIR}/${UPDATE_UNIT}.service" "${UNIT_DIR}/${UPDATE_UNIT}.timer"
  systemctl --user daemon-reload
  echo "Removed the ${UPDATE_UNIT} timer. State and logs remain in ${STATE_DIR}."
}

do_status() {
  systemctl --user list-timers "${UPDATE_UNIT}.timer" --no-pager 2>&1 || true
  echo
  do_check
  echo
  echo "--- last 15 log lines (${LOG_FILE})"
  if [ -f "$LOG_FILE" ]; then tail -n 15 "$LOG_FILE"; else echo "no log yet"; fi
}

case "$ACTION" in
  run) do_run ;;
  check) do_check ;;
  install) do_install ;;
  uninstall) do_uninstall ;;
  status) do_status ;;
esac
