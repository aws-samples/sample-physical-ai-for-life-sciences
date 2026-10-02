#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to labflow_isaac_dreamer.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
POLICY_STEPS="${POLICY_STEPS:-3000}"
DEVICE="${DEVICE:-cuda:0}"
VISUALIZER="${VISUALIZER:-}"
MAX_VISIBLE_ENVS="${MAX_VISIBLE_ENVS:-}"

ARGS=(
  --model-path "${MODEL_PATH}"
  --num-envs "${NUM_ENVS}"
  --policy-steps "${POLICY_STEPS}"
  --device "${DEVICE}"
)

if [[ -n "${VISUALIZER}" ]]; then
  ARGS+=(--visualizer "${VISUALIZER}")
  if [[ -n "${MAX_VISIBLE_ENVS}" ]]; then
    ARGS+=(--max_visible_envs "${MAX_VISIBLE_ENVS}")
  fi
fi

exec "${PROJECT_ROOT}/scripts/isaac_lab_python.sh" \
  "${PROJECT_ROOT}/isaac_lab/scripts/run_dreamer_policy.py" \
  "${ARGS[@]}"
