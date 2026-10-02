#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "train" ]]; then
  shift
fi

ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/isaaclab}"
PROJECT_ROOT="${LABFLOW_PROJECT_ROOT:-/workspace/labflow-dreamer}"

exec "${PROJECT_ROOT}/scripts/isaac_lab_python.sh" \
  "${PROJECT_ROOT}/isaac_lab/scripts/train_dreamer.py" \
  "$@"
