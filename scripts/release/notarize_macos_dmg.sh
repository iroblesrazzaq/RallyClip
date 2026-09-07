#!/usr/bin/env bash
# Submit a signed DMG to Apple notarization, wait, and staple the ticket.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/RallyClip.dmg" >&2
  exit 2
fi

DMG_PATH="$1"
if [[ ! -f "${DMG_PATH}" ]]; then
  echo "DMG not found: ${DMG_PATH}" >&2
  exit 1
fi

if [[ -z "${APPSTORE_API_KEY_ID:-}" || -z "${APPSTORE_ISSUER_ID:-}" || -z "${APPSTORE_API_PRIVATE_KEY:-}" ]]; then
  echo "Set APPSTORE_API_KEY_ID, APPSTORE_ISSUER_ID, and APPSTORE_API_PRIVATE_KEY." >&2
  exit 1
fi

if ! command -v xcrun >/dev/null 2>&1; then
  echo "xcrun not found; this script must run on macOS with Xcode CLT." >&2
  exit 1
fi

WORKDIR="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
WORKDIR="${WORKDIR%/}"
KEY_PATH="${WORKDIR}/AuthKey_${APPSTORE_API_KEY_ID}.p8"

cleanup_key() {
  rm -f "${KEY_PATH}"
}
trap cleanup_key EXIT

# The p8 must not be world-readable; notarytool rejects overly open keys.
umask 077
printf '%s\n' "${APPSTORE_API_PRIVATE_KEY}" > "${KEY_PATH}"
chmod 600 "${KEY_PATH}"

echo "Submitting ${DMG_PATH} to Apple notarization..."
SUBMIT_JSON="$(
  xcrun notarytool submit "${DMG_PATH}" \
    --key "${KEY_PATH}" \
    --key-id "${APPSTORE_API_KEY_ID}" \
    --issuer "${APPSTORE_ISSUER_ID}" \
    --wait \
    --timeout 30m \
    --output-format json
)"
echo "${SUBMIT_JSON}"

STATUS="$(printf '%s\n' "${SUBMIT_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('status',''))")"
REQUEST_ID="$(printf '%s\n' "${SUBMIT_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin).get('id',''))")"

if [[ "${STATUS}" != "Accepted" ]]; then
  echo "Notarization did not accept the DMG (status=${STATUS})." >&2
  if [[ -n "${REQUEST_ID}" ]]; then
    xcrun notarytool log "${REQUEST_ID}" \
      --key "${KEY_PATH}" \
      --key-id "${APPSTORE_API_KEY_ID}" \
      --issuer "${APPSTORE_ISSUER_ID}" || true
  fi
  exit 1
fi

xcrun stapler staple "${DMG_PATH}"
xcrun stapler validate "${DMG_PATH}"
echo "Notarized and stapled ${DMG_PATH}"
