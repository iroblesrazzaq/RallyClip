#!/usr/bin/env bash
# Submit a signed DMG to Apple notarization, wait, and staple the ticket.
#
# Apple does not send a webhook when notarization finishes. `notarytool submit`
# returns a submission id immediately; `--wait` is only polling. We submit first
# so the id is in the logs even if the wait later times out, then poll.
#
# Resume a previous submission (same signed DMG) with:
#   RALLYCLIP_NOTARY_SUBMISSION_ID=<uuid> "$0" /path/to/RallyClip.dmg
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/RallyClip.dmg" >&2
  exit 2
fi

DMG_PATH="$1"
NOTARY_TIMEOUT="${RALLYCLIP_NOTARY_TIMEOUT:-2h}"
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

notary_auth=(
  --key "${KEY_PATH}"
  --key-id "${APPSTORE_API_KEY_ID}"
  --issuer "${APPSTORE_ISSUER_ID}"
)

json_field() {
  python3 -c "import json,sys; print(json.load(sys.stdin).get(sys.argv[1],''))" "$1"
}

record_submission_id() {
  local request_id="$1"
  echo "Apple notarization submission id: ${request_id}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "notary_submission_id=${request_id}" >> "${GITHUB_OUTPUT}"
  fi
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    printf 'Apple notarization submission: `%s`\n' "${request_id}" >> "${GITHUB_STEP_SUMMARY}"
  fi
}

REQUEST_ID="${RALLYCLIP_NOTARY_SUBMISSION_ID:-}"
if [[ -z "${REQUEST_ID}" ]]; then
  echo "Submitting ${DMG_PATH} to Apple notarization (no wait)..."
  SUBMIT_JSON="$(
    xcrun notarytool submit "${DMG_PATH}" \
      "${notary_auth[@]}" \
      --output-format json
  )"
  echo "${SUBMIT_JSON}"
  REQUEST_ID="$(printf '%s\n' "${SUBMIT_JSON}" | json_field id)"
  if [[ -z "${REQUEST_ID}" ]]; then
    echo "notarytool submit did not return a submission id." >&2
    exit 1
  fi
  record_submission_id "${REQUEST_ID}"
else
  echo "Resuming Apple notarization submission ${REQUEST_ID}"
  record_submission_id "${REQUEST_ID}"
fi

echo "Waiting for Apple (timeout ${NOTARY_TIMEOUT})..."
set +e
WAIT_JSON="$(
  xcrun notarytool wait "${REQUEST_ID}" \
    "${notary_auth[@]}" \
    --timeout "${NOTARY_TIMEOUT}" \
    --output-format json
)"
WAIT_CODE=$?
set -e
echo "${WAIT_JSON}"

STATUS=""
if [[ -n "${WAIT_JSON}" ]]; then
  STATUS="$(printf '%s\n' "${WAIT_JSON}" | json_field status || true)"
  if [[ -z "${STATUS}" ]]; then
    STATUS="$(printf '%s\n' "${WAIT_JSON}" | json_field Status || true)"
  fi
fi

if [[ "${WAIT_CODE}" -ne 0 || "${STATUS}" != "Accepted" ]]; then
  echo "Notarization did not accept the DMG (status=${STATUS:-unknown}, id=${REQUEST_ID}, wait_exit=${WAIT_CODE})." >&2
  echo "Apple keeps processing after this job exits. Resume with:" >&2
  echo "  RALLYCLIP_NOTARY_SUBMISSION_ID=${REQUEST_ID} $0 ${DMG_PATH}" >&2
  xcrun notarytool log "${REQUEST_ID}" "${notary_auth[@]}" || true
  exit 1
fi

xcrun stapler staple "${DMG_PATH}"
xcrun stapler validate "${DMG_PATH}"
echo "Notarized and stapled ${DMG_PATH}"
