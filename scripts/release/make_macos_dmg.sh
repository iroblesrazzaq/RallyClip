#!/usr/bin/env bash
# Build a drag-to-Applications UDZO DMG from RallyClip.app.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/release/lib.sh
source "${ROOT_DIR}/scripts/release/lib.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/RallyClip.app /path/to/RallyClip-VERSION-macOS-ARCH.dmg [version]" >&2
  exit 2
fi

APP_PATH="$1"
DMG_PATH="$2"
VERSION="${3:-$(release_project_version)}"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "App bundle not found: ${APP_PATH}" >&2
  exit 1
fi
if ! command -v hdiutil >/dev/null 2>&1; then
  echo "hdiutil not found; this script must run on macOS." >&2
  exit 1
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/rallyclip-dmg.XXXXXX")"
cleanup() {
  rm -rf "${STAGE}"
}
trap cleanup EXIT

mkdir -p "${STAGE}"
cp -R "${APP_PATH}" "${STAGE}/RallyClip.app"
ln -s /Applications "${STAGE}/Applications"

mkdir -p "$(dirname "${DMG_PATH}")"
rm -f "${DMG_PATH}"

hdiutil create \
  -volname "RallyClip ${VERSION}" \
  -srcfolder "${STAGE}" \
  -ov \
  -format UDZO \
  -imagekey zlib-level=9 \
  "${DMG_PATH}"

echo "Created ${DMG_PATH}"
