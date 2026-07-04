#!/usr/bin/env bash
set -euo pipefail

INSTALL_CMD=': # local checkout uses the existing .venv-train environment'
START_CMD='.venv-train/bin/rallyclip gui'

echo "Working directory: $(pwd)"

echo "Install command: ${INSTALL_CMD}"
eval "${INSTALL_CMD}"

echo "Start command: ${START_CMD}"
if [[ "${RUN_START_COMMAND:-0}" == "1" ]]; then
  eval "${START_CMD}"
fi
