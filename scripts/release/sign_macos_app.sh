#!/usr/bin/env bash
# Sign a PyInstaller RallyClip.app inside-out for Developer ID + notarization.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/release/lib.sh
source "${ROOT_DIR}/scripts/release/lib.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/RallyClip.app [signing-identity]" >&2
  exit 2
fi

APP_PATH="$1"
SIGN_IDENTITY="${2:-$(release_default_sign_identity)}"
ENTITLEMENTS="${ROOT_DIR}/packaging/macos/RallyClip.entitlements"
MAIN_EXECUTABLE="${APP_PATH}/Contents/MacOS/RallyClip"
WEBENGINE_HELPER="${APP_PATH}/Contents/Frameworks/PySide6/Qt/lib/QtWebEngineCore.framework/Versions/Current/Helpers/QtWebEngineProcess.app"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "App bundle not found: ${APP_PATH}" >&2
  exit 1
fi
if [[ ! -f "${ENTITLEMENTS}" ]]; then
  echo "Entitlements file not found: ${ENTITLEMENTS}" >&2
  exit 1
fi
if [[ ! -f "${MAIN_EXECUTABLE}" ]]; then
  echo "Main executable not found: ${MAIN_EXECUTABLE}" >&2
  exit 1
fi

if ! command -v codesign >/dev/null 2>&1; then
  echo "codesign not found; this script must run on macOS." >&2
  exit 1
fi

sign_nested() {
  local path="$1"
  codesign --force --options runtime --timestamp --sign "${SIGN_IDENTITY}" "${path}"
}

sign_with_entitlements() {
  local path="$1"
  codesign --force --options runtime --timestamp \
    --entitlements "${ENTITLEMENTS}" \
    --sign "${SIGN_IDENTITY}" \
    "${path}"
}

# Nested Mach-O first (deepest paths first), without app entitlements.
# Entitlements belong on the main executable and outer bundle only.
while IFS= read -r macho; do
  [[ -z "${macho}" ]] && continue
  [[ "${macho}" == "${MAIN_EXECUTABLE}" ]] && continue
  sign_nested "${macho}"
done < <(
  find "${APP_PATH}/Contents" -type f -print0 \
    | while IFS= read -r -d '' candidate; do
        if file -b "${candidate}" 2>/dev/null | grep -q "Mach-O"; then
          slash_count="${candidate//[^\/]/}"
          printf "%d\t%s\n" "${#slash_count}" "${candidate}"
        fi
      done \
    | sort -nr \
    | cut -f2-
)

if [[ -d "${WEBENGINE_HELPER}" ]]; then
  sign_with_entitlements "${WEBENGINE_HELPER}"
fi

sign_with_entitlements "${MAIN_EXECUTABLE}"
sign_with_entitlements "${APP_PATH}"

codesign --verify --deep --strict --verbose=2 "${APP_PATH}"
codesign -d --entitlements :- "${MAIN_EXECUTABLE}" 2>/dev/null
echo "Signed ${APP_PATH} with ${SIGN_IDENTITY}"
