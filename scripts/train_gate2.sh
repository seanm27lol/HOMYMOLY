#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${HOMYMOLY_GATE2_CONFIG:-${PROJECT_ROOT}/configs/gate2.yaml}"
STATE_DIR="${HOMYMOLY_GATE2_STATE_DIR:-${PROJECT_ROOT}/artifacts/gate2/scheduler}"
COMPLETE_PATH="${STATE_DIR}/training.complete"

mkdir -p "${STATE_DIR}"
exec 9>"${STATE_DIR}/training.lock"
if ! flock -n 9; then
  echo "Another HOMYMOLY Gate-2 training process holds ${STATE_DIR}/training.lock"
  exit 0
fi
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
  CONFIG_REAL="$(realpath -- "${CONFIG_PATH}")"
  case "${CONFIG_REAL}" in
    "${PROJECT_ROOT}/configs/"*)
      CONTAINER_CONFIG="/workspace/configs/${CONFIG_REAL#"${PROJECT_ROOT}/configs/"}"
      ;;
    *)
      echo "Docker training configs must be stored under ${PROJECT_ROOT}/configs" >&2
      exit 2
      ;;
  esac
  docker compose --profile training run --rm trainer \
    train --config "${CONTAINER_CONFIG}" --resume
else
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Neither Docker Compose nor the HOMYMOLY virtual environment is available" >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m homymoly train --config "${CONFIG_PATH}" --resume
fi
FINGERPRINT_PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${FINGERPRINT_PYTHON}" ]]; then
  FINGERPRINT_PYTHON="$(command -v python3)"
fi
TEMP_COMPLETE="${COMPLETE_PATH}.$$"
"${FINGERPRINT_PYTHON}" "${PROJECT_ROOT}/scripts/gpu_idle_train.py" \
  --project-root "${PROJECT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --print-fingerprint >"${TEMP_COMPLETE}"
mv "${TEMP_COMPLETE}" "${COMPLETE_PATH}"
