#!/usr/bin/env bash
set -euo pipefail

if [ "${ROBOCLAW_ROS2_DISTRO:-none}" != "none" ] && [ -f "/opt/ros/${ROBOCLAW_ROS2_DISTRO}/setup.sh" ]; then
  # ROS setup scripts assume several tracing vars may be unset.
  set +u
  # shellcheck disable=SC1090
  source "/opt/ros/${ROBOCLAW_ROS2_DISTRO}/setup.sh"
  set -u
fi

exec /usr/local/bin/roboclaw-real "$@"
