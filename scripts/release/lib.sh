#!/usr/bin/env bash
# Shared helpers for macOS release scripts. Source from repo-relative scripts.
# shellcheck shell=bash

release_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${here}/../.." && pwd
}

release_project_version() {
  local root
  root="$(release_repo_root)"
  python3 - "${root}/pyproject.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

print(tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["project"]["version"])
PY
}

release_macos_arch_label() {
  case "$(uname -m)" in
    arm64|aarch64) echo arm64 ;;
    x86_64) echo x86_64 ;;
    *) uname -m ;;
  esac
}

release_dmg_basename() {
  local version="${1:?version required}"
  local arch="${2:?arch required}"
  echo "RallyClip-${version}-macOS-${arch}.dmg"
}

release_default_sign_identity() {
  echo "${MACOS_SIGN_IDENTITY:-Developer ID Application: Ismael Robles-Razzaq (L9W8X6N9B9)}"
}
