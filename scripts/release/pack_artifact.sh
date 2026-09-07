#!/usr/bin/env bash
# Zip DEFAULT_ARTIFACT_DIR into rallyclip_v0.5.0.zip (and a .sha256 sidecar).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/release/lib.sh
source "${ROOT_DIR}/scripts/release/lib.sh"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [output-dir]" >&2
  exit 2
fi

OUT_DIR="${1:-${ROOT_DIR}/dist}"
mkdir -p "${OUT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m runtime.artifact pack --repo-root "${ROOT_DIR}" "${OUT_DIR}"
