#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-pybullet"
RUNTIME_ROOT="${REPO_ROOT}/.local/roboclaw"
ASSET_ROOT="${RUNTIME_ROOT}/so101-classic-control"
TRACE_DIR="${RUNTIME_ROOT}/so101-traces"
SOURCE_SHA="583899971f978b1b03664a1fa25dd377cfe429c6"
SOURCE_URL="https://codeload.github.com/sanjaypokkali/SO101-Classic-Control/tar.gz/${SOURCE_SHA}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_command python3
require_command curl
require_command tar

mkdir -p "${RUNTIME_ROOT}" "${TRACE_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_DIR}/bin/pip" install -r "${SCRIPT_DIR}/requirements-so101-sim.txt" >/dev/null

INSTALLED_SHA=""
if [[ -f "${ASSET_ROOT}/.source-sha" ]]; then
  INSTALLED_SHA="$(tr -d '\n' < "${ASSET_ROOT}/.source-sha")"
fi

if [[ "${INSTALLED_SHA}" != "${SOURCE_SHA}" || ! -f "${ASSET_ROOT}/so101_new_calib.urdf" ]]; then
  TMP_DIR="$(mktemp -d)"
  cleanup() {
    rm -rf "${TMP_DIR}"
  }
  trap cleanup EXIT

  rm -rf "${ASSET_ROOT}"
  mkdir -p "${ASSET_ROOT}"
  curl -L --fail "${SOURCE_URL}" -o "${TMP_DIR}/so101.tar.gz"
  tar -xzf "${TMP_DIR}/so101.tar.gz" -C "${TMP_DIR}"
  SOURCE_DIR="$(find "${TMP_DIR}" -maxdepth 1 -type d -name 'SO101-Classic-Control-*' | head -n 1)"

  if [[ -z "${SOURCE_DIR}" ]]; then
    echo "failed to locate extracted SO101 assets" >&2
    exit 1
  fi

  cp -R "${SOURCE_DIR}/urdf/." "${ASSET_ROOT}/"
  printf '%s\n' "${SOURCE_SHA}" > "${ASSET_ROOT}/.source-sha"
  trap - EXIT
  cleanup
fi

echo "SO101 simulator ready"
echo "python=${VENV_DIR}/bin/python"
echo "urdf=${ASSET_ROOT}/so101_new_calib.urdf"
echo "trace_dir=${TRACE_DIR}"
