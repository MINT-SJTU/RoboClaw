#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ROBOCLAW_DOCKER_HOME="${ROBOCLAW_DOCKER_HOME:-${HOME}/.roboclaw-docker}"
DEFAULT_DOCKER_PROFILE="${ROBOCLAW_DOCKER_PROFILE:-ubuntu2404}"
DEFAULT_MATRIX_PROFILES="${ROBOCLAW_DOCKER_MATRIX_PROFILES:-ubuntu2204,ubuntu2204-ros2,ubuntu2404,ubuntu2404-ros2}"

die() {
  echo "error: $*" >&2
  exit 1
}

require_instance() {
  local instance="${1:-}"
  [ -n "${instance}" ] || die "instance name is required"
  [[ "${instance}" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid instance name: ${instance}"
}

docker_profile() {
  local profile="${1:-${DEFAULT_DOCKER_PROFILE}}"
  case "${profile}" in
    ubuntu2204|ubuntu2204-ros2|ubuntu2404|ubuntu2404-ros2)
      printf '%s\n' "${profile}"
      ;;
    *)
      die "unknown docker profile: ${profile}"
      ;;
  esac
}

docker_profile_base_image() {
  local profile
  profile="$(docker_profile "${1:-}")"
  case "${profile}" in
    ubuntu2204|ubuntu2204-ros2)
      printf '%s\n' "ubuntu:22.04"
      ;;
    ubuntu2404|ubuntu2404-ros2)
      printf '%s\n' "ubuntu:24.04"
      ;;
  esac
}

docker_profile_ros_distro() {
  local profile
  profile="$(docker_profile "${1:-}")"
  case "${profile}" in
    ubuntu2204-ros2)
      printf '%s\n' "humble"
      ;;
    ubuntu2404-ros2)
      printf '%s\n' "jazzy"
      ;;
    *)
      printf '%s\n' "none"
      ;;
  esac
}

docker_profile_installs_ros2() {
  local profile
  profile="$(docker_profile "${1:-}")"
  case "${profile}" in
    *-ros2)
      printf '%s\n' "1"
      ;;
    *)
      printf '%s\n' "0"
      ;;
  esac
}

list_docker_profiles() {
  printf '%s\n' ubuntu2204 ubuntu2204-ros2 ubuntu2404 ubuntu2404-ros2
}

parse_profile_flag() {
  local profile="${DEFAULT_DOCKER_PROFILE}"
  if [ "${1:-}" = "--profile" ]; then
    [ -n "${2:-}" ] || die "missing value for --profile"
    profile="$(docker_profile "${2}")"
    shift 2
  fi
  printf '%s\n' "${profile}"
}

join_profiles_csv() {
  local first=1
  local profile
  for profile in "$@"; do
    if [ "${first}" -eq 1 ]; then
      printf '%s' "${profile}"
      first=0
    else
      printf ',%s' "${profile}"
    fi
  done
  printf '\n'
}

split_profiles_csv() {
  local csv="${1:-}"
  [ -n "${csv}" ] || return 0
  local old_ifs="${IFS}"
  IFS=',' read -r -a _profiles <<< "${csv}"
  IFS="${old_ifs}"
  local profile
  for profile in "${_profiles[@]}"; do
    docker_profile "${profile}"
  done
}

instance_key() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  printf '%s--%s\n' "${instance}" "${profile}"
}

ros2_namespace_prefix() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  local raw
  raw="$(instance_key "${instance}" "${profile}")"
  raw="${raw//[^A-Za-z0-9_]/_}"
  printf '/roboclaw/%s\n' "${raw}"
}

instance_dir() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  printf '%s/instances/%s\n' "${ROBOCLAW_DOCKER_HOME}" "$(instance_key "${instance}" "${profile}")"
}

image_ref() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  printf 'roboclaw:%s-%s-%s\n' "${instance}" "${profile}" "$(current_commit_short)"
}

current_commit_short() {
  git -C "${REPO_ROOT}" rev-parse --short HEAD
}

require_clean_git() {
  if [ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
    die "git worktree is dirty; commit or stash changes before building the Docker image"
  fi
}

compose_project() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  printf 'roboclaw-%s-%s\n' "${instance}" "${profile}"
}

dev_container_name() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  printf 'roboclaw-dev-%s-%s\n' "${instance}" "${profile}"
}

find_proxy_port() {
  local ss_output line
  ss_output="$(ss -ltnpH 2>/dev/null || true)"
  if [ -z "${ss_output}" ]; then
    return 1
  fi

  if printf '%s\n' "${ss_output}" | awk '/verge-mihomo/ && $4 ~ /127\.0\.0\.1:7897$/ { print "7897"; exit }' | grep -q .; then
    printf '7897\n'
    return 0
  fi

  local port
  for port in 7897 7890 7891 20170 7895 7898 7899; do
    if printf '%s\n' "${ss_output}" | awk -v port="${port}" '$4 ~ ("127\\.0\\.0\\.1:" port "$") { found=1 } END { exit(found ? 0 : 1) }'; then
      printf '%s\n' "${port}"
      return 0
    fi
  done

  line="$(printf '%s\n' "${ss_output}" | grep -E '127\.0\.0\.1:.*(verge-mihomo|clash|mihomo|sing-box|xray|v2ray|dae|hysteria)' | head -n 1 || true)"
  if [ -n "${line}" ]; then
    printf '%s\n' "${line}" | awk '{split($4, a, ":"); print a[length(a)]}'
    return 0
  fi

  return 1
}

configure_proxy_env() {
  local proxy_port="${1:-}"
  if [ -z "${proxy_port}" ]; then
    proxy_port="$(find_proxy_port || true)"
  fi
  if [ -z "${proxy_port}" ]; then
    unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
    return 0
  fi

  export HTTP_PROXY="http://127.0.0.1:${proxy_port}"
  export HTTPS_PROXY="http://127.0.0.1:${proxy_port}"
  export ALL_PROXY="socks5://127.0.0.1:${proxy_port}"
  export http_proxy="${HTTP_PROXY}"
  export https_proxy="${HTTPS_PROXY}"
  export all_proxy="${ALL_PROXY}"
}

ensure_image_exists() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  docker image inspect "$(image_ref "${instance}" "${profile}")" >/dev/null 2>&1 || \
    die "image $(image_ref "${instance}" "${profile}") not found; run scripts/docker/build-image.sh --profile ${profile} ${instance}"
}

ensure_instance_dir() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  mkdir -p \
    "$(instance_dir "${instance}" "${profile}")/workspace" \
    "$(instance_dir "${instance}" "${profile}")/home" \
    "$(instance_dir "${instance}" "${profile}")/home/.codex"
}

host_codex_auth_path() {
  local path="${HOME}/.codex/auth.json"
  if [ -f "${path}" ]; then
    printf '%s\n' "${path}"
  fi
}

host_oauth_cli_kit_auth_dir() {
  local path="${HOME}/.local/share/oauth-cli-kit/auth"
  if [ -d "${path}" ]; then
    printf '%s\n' "${path}"
  fi
}

host_lerobot_calibration_dir() {
  local path="${HOME}/.cache/huggingface/lerobot/calibration"
  if [ -d "${path}" ]; then
    printf '%s\n' "${path}"
  fi
}

host_scservo_sdk_dir() {
  python - <<'PY' 2>/dev/null
import importlib.util
import pathlib
spec = importlib.util.find_spec("scservo_sdk")
if spec and spec.origin:
    print(pathlib.Path(spec.origin).resolve().parent)
PY
}

instance_oauth_cli_kit_auth_dir() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  printf '%s/home/.local/share/oauth-cli-kit/auth\n' "$(instance_dir "${instance}" "${profile}")"
}

prepare_auth_mounts() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  local codex_auth_path oauth_dir instance_oauth_dir
  codex_auth_path="$(host_codex_auth_path || true)"
  oauth_dir="$(host_oauth_cli_kit_auth_dir || true)"
  instance_oauth_dir="$(instance_oauth_cli_kit_auth_dir "${instance}" "${profile}")"

  mkdir -p "${instance_oauth_dir}"

  if [ -n "${oauth_dir}" ]; then
    rm -f "${instance_oauth_dir}/codex.json"
  elif [ -n "${codex_auth_path}" ]; then
    rm -rf "${instance_oauth_dir}"
    mkdir -p "${instance_oauth_dir}"
  fi
}

host_serial_devices() {
  local device
  for device in /dev/ttyACM* /dev/ttyUSB*; do
    if [ -e "${device}" ]; then
      printf '%s\n' "${device}"
    fi
  done
}

host_serial_symlink_dir() {
  if [ -d /dev/serial/by-id ]; then
    printf '%s\n' "/dev/serial/by-id"
  fi
}

append_hardware_device_args() {
  local -n docker_args_ref="$1"
  local seen_group_ids=" "
  local device group_id serial_symlink_dir

  while IFS= read -r device; do
    [ -n "${device}" ] || continue
    docker_args_ref+=(--device "${device}:${device}")
    group_id="$(stat -c '%g' "${device}")"
    if [[ "${seen_group_ids}" != *" ${group_id} "* ]]; then
      docker_args_ref+=(--group-add "${group_id}")
      seen_group_ids+="$(printf '%s ' "${group_id}")"
    fi
  done < <(host_serial_devices || true)

  serial_symlink_dir="$(host_serial_symlink_dir || true)"
  if [ -n "${serial_symlink_dir}" ]; then
    docker_args_ref+=(-v "${serial_symlink_dir}:${serial_symlink_dir}:ro")
  fi
}

compose_cmd() {
  local instance="${1}"
  local profile
  profile="$(docker_profile "${2:-}")"
  shift
  shift
  if docker compose version >/dev/null 2>&1; then
    ROBOCLAW_IMAGE="$(image_ref "${instance}" "${profile}")" \
    ROBOCLAW_INSTANCE_DIR="$(instance_dir "${instance}" "${profile}")" \
    ROBOCLAW_UID="$(id -u)" \
    ROBOCLAW_GID="$(id -g)" \
    docker compose -f "${REPO_ROOT}/docker-compose.yml" -p "$(compose_project "${instance}" "${profile}")" "$@"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    ROBOCLAW_IMAGE="$(image_ref "${instance}" "${profile}")" \
    ROBOCLAW_INSTANCE_DIR="$(instance_dir "${instance}" "${profile}")" \
    ROBOCLAW_UID="$(id -u)" \
    ROBOCLAW_GID="$(id -g)" \
    docker-compose -f "${REPO_ROOT}/docker-compose.yml" -p "$(compose_project "${instance}" "${profile}")" "$@"
    return
  fi

  die "neither 'docker compose' nor 'docker-compose' is available"
}
