#!/usr/bin/env bash
# Check the built app's resources, then launch it and fail if it quits early, as
# a Bundle.module trap does. Bundle.module falls back to an absolute path in the
# SwiftPM build dir, which exists only on the build machine, so that dir is moved
# away while this runs.
set -euo pipefail
APP="$(cd "${1:?usage: launch_smoke.sh RallyClip.app}" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UP_SECONDS="${RALLYCLIP_SMOKE_SECONDS:-10}"
BUILD="$ROOT/build/native-swift"
HIDDEN=""
# Own data dir: the backend recreates native-backend.log, and the user's library stays untouched.
export RALLYCLIP_NATIVE_DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rallyclip-smoke.XXXXXX")"
pid=""
cleanup() {
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  pkill -f "$APP/Contents/Resources/Backend/" 2>/dev/null || true
  if [[ -n "$HIDDEN" ]]; then mv "$HIDDEN" "$BUILD"; fi
  rm -rf "$RALLYCLIP_NATIVE_DATA_DIR"
}
trap cleanup EXIT
if [[ -d "$BUILD" ]]; then HIDDEN="$BUILD.smoke-hidden"; mv "$BUILD" "$HIDDEN"; fi
# The logo, through the same lookup RallyLogo uses (a hidden window may not draw it).
"$APP/Contents/MacOS/RallyClip" --check-resources || { echo "RallyClip --check-resources failed" >&2; exit 1; }
echo "resources ok"
# First launch shows the intro; -welcomeSeen '<true/>' (argument domain, not saved;
# a plist bool, since @AppStorage reads "YES" as a string) opens the library.
for args in "" "-welcomeSeen <true/>"; do
  # shellcheck disable=SC2086
  "$APP/Contents/MacOS/RallyClip" $args &
  pid=$!
  sleep "$UP_SECONDS"
  if ! kill -0 "$pid" 2>/dev/null; then
    code=0; wait "$pid" || code=$?
    echo "RallyClip quit within ${UP_SECONDS}s (exit $code, args '${args}')" >&2
    exit 1
  fi
  kill "$pid"; wait "$pid" 2>/dev/null || true; pid=""
  pkill -f "$APP/Contents/Resources/Backend/" 2>/dev/null || true
  echo "launch ok: args '${args}'"
done
