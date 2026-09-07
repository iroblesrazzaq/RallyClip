#!/usr/bin/env bash
# After PyInstaller: sign RallyClip.app, wrap a DMG, sign/notarize/staple it.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/release/lib.sh
source "${ROOT_DIR}/scripts/release/lib.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/RallyClip.app [output-dir]" >&2
  exit 2
fi

APP_PATH="$1"
OUT_DIR="${2:-${ROOT_DIR}/dist}"
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
VERSION="$(release_project_version)"
ARCH="$(release_macos_arch_label)"
DMG_NAME="$(release_dmg_basename "${VERSION}" "${ARCH}")"
DMG_PATH="${OUT_DIR}/${DMG_NAME}"
SIGN_IDENTITY="$(release_default_sign_identity)"

write_dmg_outputs() {
  shasum -a 256 "${DMG_PATH}" | tee "${DMG_PATH}.sha256"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "dmg_path=${DMG_PATH}"
      echo "dmg_name=${DMG_NAME}"
      echo "sha256_path=${DMG_PATH}.sha256"
      echo "version=${VERSION}"
      echo "arch=${ARCH}"
    } >> "${GITHUB_OUTPUT}"
  fi
  echo "Release artifact: ${DMG_PATH}"
}

if [[ "${RALLYCLIP_SKIP_SIGNING:-}" == "1" ]]; then
  echo "RALLYCLIP_SKIP_SIGNING=1; wrapping an unsigned DMG."
  "${ROOT_DIR}/scripts/release/make_macos_dmg.sh" "${APP_PATH}" "${DMG_PATH}" "${VERSION}"
else
  "${ROOT_DIR}/scripts/release/sign_macos_app.sh" "${APP_PATH}" "${SIGN_IDENTITY}"
  "${ROOT_DIR}/scripts/release/make_macos_dmg.sh" "${APP_PATH}" "${DMG_PATH}" "${VERSION}"
  codesign --force --sign "${SIGN_IDENTITY}" --timestamp "${DMG_PATH}"
fi

# Persist path/checksum before notarization so a timeout still leaves a
# signed DMG that can be resumed, stapled, and uploaded.
write_dmg_outputs

if [[ "${RALLYCLIP_SKIP_SIGNING:-}" != "1" && "${RALLYCLIP_SKIP_NOTARIZE:-}" != "1" ]]; then
  "${ROOT_DIR}/scripts/release/notarize_macos_dmg.sh" "${DMG_PATH}"
  write_dmg_outputs
elif [[ "${RALLYCLIP_SKIP_NOTARIZE:-}" == "1" ]]; then
  echo "RALLYCLIP_SKIP_NOTARIZE=1; DMG is signed but not notarized."
fi
