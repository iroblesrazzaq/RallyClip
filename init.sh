#!/usr/bin/env bash
set -euo pipefail

INSTALL_CMD=': # local checkout uses the existing .venv-train environment'
VERIFY_CMD='PYTHONPATH=build/lib .venv-train/bin/python -m compileall -q build/lib'
START_CMD='.venv-train/bin/rallyclip gui'

echo "Working directory: $(pwd)"

echo "Install command: ${INSTALL_CMD}"
eval "${INSTALL_CMD}"

echo "Verify command: ${VERIFY_CMD}"
if ! eval "${VERIFY_CMD}"; then
  echo "BASELINE BROKEN -- fix before feature work"
  exit 1
fi

echo "Start command: ${START_CMD}"
if [[ "${RUN_START_COMMAND:-0}" == "1" ]]; then
  eval "${START_CMD}"
fi
