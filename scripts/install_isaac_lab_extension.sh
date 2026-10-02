#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:?Set ISAACLAB_ROOT to the Isaac Lab checkout}"
EXTENSION_ROOT="${PROJECT_ROOT}/isaac_lab/source/labflow_isaac"

if [[ ! -f "${ISAACLAB_ROOT}/uv.lock" ]]; then
  echo "Isaac Lab checkout with uv.lock not found: ${ISAACLAB_ROOT}"
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv, then rerun this command."
  exit 1
fi

(
  cd "${ISAACLAB_ROOT}"
  env -u VIRTUAL_ENV uv sync --extra isaacsim
  env -u VIRTUAL_ENV uv pip install \
    --python "${ISAACLAB_ROOT}/.venv/bin/python" \
    --no-deps \
    --editable "${PROJECT_ROOT}" \
    --editable "${EXTENSION_ROOT}"
)

printf 'Installed LabFlow packages into %s\n' "${ISAACLAB_ROOT}"
