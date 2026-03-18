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
shift || true
require_instance "${INSTANCE}"
ensure_image_exists "${INSTANCE}" "${PROFILE}"
ensure_instance_dir "${INSTANCE}" "${PROFILE}"
configure_proxy_env
prepare_auth_mounts "${INSTANCE}" "${PROFILE}"
AUTH_PATH="$(host_codex_auth_path || true)"
OAUTH_CLI_KIT_AUTH_DIR="$(host_oauth_cli_kit_auth_dir || true)"
LEROBOT_CALIBRATION_DIR="$(host_lerobot_calibration_dir || true)"
SCSERVO_SDK_DIR="$(host_scservo_sdk_dir || true)"

if [ "$#" -eq 0 ]; then
  set -- status
fi

"${SCRIPT_DIR}/bootstrap-instance.sh" --profile "${PROFILE}" "${INSTANCE}"

DOCKER_ARGS=(
  --rm
  --network host
  --user "$(id -u):$(id -g)"
  -e HOME=/roboclaw-instance/home
  -e ROBOCLAW_CONFIG_PATH=/roboclaw-instance/config.json
  -e ROBOCLAW_WORKSPACE_PATH=/roboclaw-instance/workspace
  -e ROBOCLAW_ROS2_NAMESPACE_PREFIX="$(ros2_namespace_prefix "${INSTANCE}" "${PROFILE}")"
  -e HTTP_PROXY="${HTTP_PROXY:-}"
  -e HTTPS_PROXY="${HTTPS_PROXY:-}"
  -e ALL_PROXY="${ALL_PROXY:-}"
  -e http_proxy="${http_proxy:-}"
  -e https_proxy="${https_proxy:-}"
  -e all_proxy="${all_proxy:-}"
  -v "$(instance_dir "${INSTANCE}" "${PROFILE}"):/roboclaw-instance"
)

if [ -n "${AUTH_PATH}" ]; then
  DOCKER_ARGS+=(-v "${AUTH_PATH}:/roboclaw-instance/home/.codex/auth.json:ro")
fi

if [ -n "${OAUTH_CLI_KIT_AUTH_DIR}" ]; then
  DOCKER_ARGS+=(-v "${OAUTH_CLI_KIT_AUTH_DIR}:/roboclaw-instance/home/.local/share/oauth-cli-kit/auth")
fi

if [ -n "${LEROBOT_CALIBRATION_DIR}" ]; then
  DOCKER_ARGS+=(-v "${LEROBOT_CALIBRATION_DIR}:/roboclaw-instance/home/.cache/huggingface/lerobot/calibration:ro")
fi

if [ -n "${SCSERVO_SDK_DIR}" ]; then
  DOCKER_ARGS+=(-v "${SCSERVO_SDK_DIR}:/usr/local/lib/python3.11/dist-packages/scservo_sdk:ro")
fi

append_hardware_device_args DOCKER_ARGS

docker run "${DOCKER_ARGS[@]}" \
  "$(image_ref "${INSTANCE}" "${PROFILE}")" \
  "$@"
