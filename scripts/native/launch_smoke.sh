#!/usr/bin/env bash
# Launch the built app and fail if it quits early, as a Bundle.module trap does.
# Bundle.module falls back to an absolute path in the SwiftPM build dir, which
# exists only on the build machine, so that dir is moved away while this runs.
set -euo pipefail
APP="$(cd "${1:?usage: launch_smoke.sh RallyClip.app}" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UP_SECONDS="${RALLYCLIP_SMOKE_SECONDS:-10}"
BUILD="$ROOT/build/native-swift"
HIDDEN=""
pid=""
cleanup() {
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  pkill -f "$APP/Contents/Resources/Backend/" 2>/dev/null || true
  if [[ -n "$HIDDEN" ]]; then mv "$HIDDEN" "$BUILD"; fi
}
trap cleanup EXIT
if [[ -d "$BUILD" ]]; then HIDDEN="$BUILD.smoke-hidden"; mv "$BUILD" "$HIDDEN"; fi
# First launch shows the intro; -welcomeSeen YES (argument domain, not saved)
# opens the library window, which draws the logo from the resource bundle.
for args in "" "-welcomeSeen YES"; do
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
