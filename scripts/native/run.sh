#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RALLYCLIP_NATIVE_REPO="$ROOT"
export RALLYCLIP_NATIVE_PYTHON="${RALLYCLIP_BUILD_PYTHON:-$(command -v python3)}"
cd "$ROOT"
swift run --package-path macos --scratch-path build/native-swift RallyClip
