#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout}"

if [[ ! -f "${ISAACLAB_ROOT}/uv.lock" ]]; then
  echo "Isaac Lab checkout with uv.lock not found: ${ISAACLAB_ROOT}"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv, then rerun this command."
  exit 1
fi

cd "${ISAACLAB_ROOT}"
exec env -u VIRTUAL_ENV uv run --extra isaacsim python "$@"
