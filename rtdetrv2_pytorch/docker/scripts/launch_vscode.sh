#!/usr/bin/env bash
set -eu pipefail

VSCODE_USER_DATA_DIR="${VSCODE_USER_DATA_DIR:-/tmp/vscode-root}"

is_non_gui_command() {
  case "${1:-}" in
    --help|--version|--status|--verbose|--list-extensions|--show-versions|--install-extension|--uninstall-extension)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

discover_vscode_bin() {
  if [[ -n "${VSCODE_BIN:-}" && -x "${VSCODE_BIN}" ]]; then
    echo "${VSCODE_BIN}"
    return 0
  fi

  local candidates=(
    "/opt/vscode/bin/code"
    "/opt/vscode/code"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  candidate="$(find /opt/programs -maxdepth 10 -type f -path '*/bin/code' | head -n1 || true)"
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  candidate="$(find /opt/programs -maxdepth 10 -type f -name 'code' | head -n1 || true)"
  if [[ -n "${candidate}" && -x "${candidate}" ]]; then
    echo "${candidate}"
    return 0
  fi

  return 1
}

VSCODE_BIN="$(discover_vscode_bin || true)"
if [[ -z "${VSCODE_BIN}" ]]; then
  echo "VS Code launcher not found under /opt/vscode or /opt/programs"
  echo "Expected source path on host: /home/ronbar/programs"
  exit 1
fi

mkdir -p "$VSCODE_USER_DATA_DIR"

if ! is_non_gui_command "${1:-}"; then
  if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
    echo "VS Code GUI requires display forwarding, but DISPLAY/WAYLAND_DISPLAY is not set."
    echo "Set display env in docker-compose and mount /tmp/.X11-unix for X11 forwarding."
    exit 1
  fi
fi

if [[ "${VSCODE_LAUNCH_DEBUG:-0}" == "1" ]]; then
  echo "Using VS Code binary: ${VSCODE_BIN}"
  echo "DISPLAY=${DISPLAY:-<unset>} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>}"
fi

exec "$VSCODE_BIN" --no-sandbox --user-data-dir "$VSCODE_USER_DATA_DIR" "$@"
