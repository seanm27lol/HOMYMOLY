#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${HOMYMOLY_CONFIG:-${PROJECT_ROOT}/configs/stage1.yaml}"
TENSORBOARD_HOST="${HOMYMOLY_TENSORBOARD_HOST:-127.0.0.1}"
TENSORBOARD_PORT="${HOMYMOLY_TENSORBOARD_PORT:-6006}"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "TensorBoard is unavailable. Install HOMYMOLY with the dashboard extra." >&2
  echo "python -m pip install -e '${PROJECT_ROOT}[dashboard]'" >&2
  exit 2
fi

if [[ -n "${HOMYMOLY_TENSORBOARD_LOGDIR:-}" ]]; then
  TENSORBOARD_DIR="${HOMYMOLY_TENSORBOARD_LOGDIR}"
  mkdir -p "${TENSORBOARD_DIR}"
else
  TENSORBOARD_DIR="$(
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      python -m homymoly paths \
        --config "${CONFIG_PATH}" \
        --kind tensorboard \
        --create
  )"
fi

echo "Starting HOMYMOLY TensorBoard at http://${TENSORBOARD_HOST}:${TENSORBOARD_PORT}"
echo "Log directory: ${TENSORBOARD_DIR}"

if [[ "${TENSORBOARD_HOST}" == "0.0.0.0" || "${TENSORBOARD_HOST}" == "::" ]]; then
  exec tensorboard \
    --logdir "${TENSORBOARD_DIR}" \
    --bind_all \
    --port "${TENSORBOARD_PORT}" \
    "$@"
fi

exec tensorboard \
  --logdir "${TENSORBOARD_DIR}" \
  --host "${TENSORBOARD_HOST}" \
  --port "${TENSORBOARD_PORT}" \
  "$@"
