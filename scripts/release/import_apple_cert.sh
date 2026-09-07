#!/usr/bin/env bash
# Import a base64-encoded Developer ID .p12 into a temporary keychain for CI.
# Never print the certificate or password.
set +x
set -euo pipefail

{ set +x; } 2>/dev/null

if [[ -z "${MACOS_CERTIFICATE_P12_BASE64:-}" ]]; then
  echo "MACOS_CERTIFICATE_P12_BASE64 is empty; nothing to import." >&2
  exit 1
fi
if [[ -z "${MACOS_CERTIFICATE_PASSWORD:-}" ]]; then
  echo "MACOS_CERTIFICATE_PASSWORD is empty; cannot import the .p12." >&2
  exit 1
fi

umask 077
PARENT_TEMP="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
PARENT_TEMP="${PARENT_TEMP%/}"
WORKDIR="$(mktemp -d "${PARENT_TEMP}/rallyclip-signing.XXXXXX")"
chmod 700 "${WORKDIR}"
CERT_PATH="${WORKDIR}/developer-id.p12"
KEYCHAIN_PATH="${WORKDIR}/signing.keychain-db"
KEYCHAIN_PASSWORD="${RALLYCLIP_KEYCHAIN_PASSWORD:-$(openssl rand -base64 32)}"

cleanup_cert() {
  rm -f "${CERT_PATH}"
}
trap cleanup_cert EXIT

printf '%s' "${MACOS_CERTIFICATE_P12_BASE64}" | tr -d '\n\r ' | base64 --decode > "${CERT_PATH}"
chmod 600 "${CERT_PATH}"

security delete-keychain "${KEYCHAIN_PATH}" >/dev/null 2>&1 || true
security create-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security set-keychain-settings -lut 21600 "${KEYCHAIN_PATH}"
security unlock-keychain -p "${KEYCHAIN_PASSWORD}" "${KEYCHAIN_PATH}"
security import "${CERT_PATH}" \
  -k "${KEYCHAIN_PATH}" \
  -P "${MACOS_CERTIFICATE_PASSWORD}" \
  -A \
  -T /usr/bin/codesign \
  -T /usr/bin/security
security set-key-partition-list \
  -S apple-tool:,apple:,codesign: \
  -s \
  -k "${KEYCHAIN_PASSWORD}" \
  "${KEYCHAIN_PATH}" >/dev/null
security list-keychain -d user -s "${KEYCHAIN_PATH}"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "RALLYCLIP_KEYCHAIN_PATH=${KEYCHAIN_PATH}" >> "${GITHUB_ENV}"
fi

echo "Imported Developer ID certificate into ${KEYCHAIN_PATH}"
