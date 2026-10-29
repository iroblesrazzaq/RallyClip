#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY="${RALLYCLIP_BUILD_PYTHON:-python3}"
OUT="${1:-$ROOT/dist/native}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
PYTHONPATH=src "$PY" -c 'from pathlib import Path; from runtime.artifact import verify_artifact_dir; from runtime.defaults import DEFAULT_ARTIFACT_DIR; verify_artifact_dir(Path(DEFAULT_ARTIFACT_DIR))'
if [[ "${RALLYCLIP_REUSE_BACKEND:-0}" != "1" ]]; then
  "$PY" -m PyInstaller --noconfirm --workpath build/native-helper --distpath build/native-helper-dist packaging/macos/NativeBackend.spec
fi
swift build --package-path macos -c release --scratch-path build/native-swift
BIN_DIR="$(swift build --package-path macos -c release --scratch-path build/native-swift --show-bin-path)"
STAGE="$(mktemp -d "$OUT/.native-stage.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
APP="$STAGE/RallyClip.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/RallyClip" "$APP/Contents/MacOS/RallyClip"
ditto "$BIN_DIR/RallyClipNative_RallyClip.bundle" "$APP/Contents/Resources/RallyClipNative_RallyClip.bundle"
ditto build/native-helper-dist/RallyClipBackend "$APP/Contents/Resources/Backend"
cp docs/rallyclip.icns "$APP/Contents/Resources/RallyClip.icns"
"$PY" - "$ROOT" "$APP" <<'PY'
import plistlib, sys
from pathlib import Path
try:
    import tomllib
except ImportError:
    import tomli as tomllib
root, app = map(Path, sys.argv[1:])
version = tomllib.loads((root / 'pyproject.toml').read_text())['project']['version']
plist = {
    'CFBundleName': 'RallyClip', 'CFBundleDisplayName': 'RallyClip',
    'CFBundleIdentifier': 'com.iroblesrazzaq.rallyclip', 'CFBundleExecutable': 'RallyClip',
    'CFBundleShortVersionString': version, 'CFBundleVersion': version,
    'CFBundlePackageType': 'APPL', 'CFBundleIconFile': 'RallyClip.icns',
    'LSMinimumSystemVersion': '14.0', 'NSHighResolutionCapable': True,
    'NSSupportsAutomaticGraphicsSwitching': True,
    'LSApplicationCategoryType': 'public.app-category.sports',
    'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True},
}
(app / 'Contents/Info.plist').write_bytes(plistlib.dumps(plist))
PY
# Local builds use ad-hoc signing. CI applies Developer ID inside-out before
# wrapping and notarizing the DMG with the existing Apple secrets.
codesign --force --deep --sign - "$APP"
if [[ -d "$OUT/RallyClip.app" ]]; then rm -rf "$OUT/RallyClip.app"; fi
mv "$APP" "$OUT/RallyClip.app"
echo "Built $OUT/RallyClip.app"
