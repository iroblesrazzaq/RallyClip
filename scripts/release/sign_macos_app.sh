#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 /path/to/RallyClip.app 'Developer ID Application: Name (TEAMID)'" >&2
  exit 2
fi

APP_PATH="$1"
SIGN_IDENTITY="$2"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENTITLEMENTS="$ROOT_DIR/packaging/macos/RallyClip.entitlements"
MAIN_EXECUTABLE="$APP_PATH/Contents/MacOS/RallyClip"
WEBENGINE_HELPER="$APP_PATH/Contents/Frameworks/PySide6/Qt/lib/QtWebEngineCore.framework/Versions/Current/Helpers/QtWebEngineProcess.app"

if [[ ! -d "$APP_PATH" ]]; then
  echo "App bundle not found: $APP_PATH" >&2
  exit 1
fi

if [[ ! -f "$ENTITLEMENTS" ]]; then
  echo "Entitlements file not found: $ENTITLEMENTS" >&2
  exit 1
fi

if [[ -d "$WEBENGINE_HELPER" ]]; then
  codesign --force --options runtime \
    --entitlements "$ENTITLEMENTS" \
    --sign "$SIGN_IDENTITY" \
    "$WEBENGINE_HELPER"
fi

codesign --force --options runtime \
  --entitlements "$ENTITLEMENTS" \
  --sign "$SIGN_IDENTITY" \
  "$MAIN_EXECUTABLE"

codesign --force --deep --options runtime \
  --entitlements "$ENTITLEMENTS" \
  --sign "$SIGN_IDENTITY" \
  "$APP_PATH"

codesign --verify --deep --strict --verbose=2 "$APP_PATH"
codesign -d --entitlements :- "$MAIN_EXECUTABLE" 2>/dev/null
if [[ -d "$WEBENGINE_HELPER" ]]; then
  codesign -d --entitlements :- "$WEBENGINE_HELPER" 2>/dev/null
fi
