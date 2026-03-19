#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=./common.sh
source "${SCRIPT_DIR}/common.sh"

PROFILE="${DEFAULT_DOCKER_PROFILE}"
if [ "${1:-}" = "--profile" ]; then
  [ -n "${2:-}" ] || die "missing value for --profile"
  PROFILE="$(docker_profile "${2}")"
  shift 2
fi

INSTANCE="${1:-}"
require_instance "${INSTANCE}"
ensure_image_exists "${INSTANCE}" "${PROFILE}"
ensure_instance_dir "${INSTANCE}" "${PROFILE}"
configure_proxy_env

"${SCRIPT_DIR}/start-dev.sh" --profile "${PROFILE}" "${INSTANCE}" >/dev/null
docker exec -it -w /roboclaw-source "$(dev_container_name "${INSTANCE}" "${PROFILE}")" /bin/sh
