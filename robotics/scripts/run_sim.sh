#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$REPO_ROOT/robotics/ros_ws/install/setup.bash"
DEFAULT_MAP="$REPO_ROOT/robotics/ros_ws/src/roboclaw_tb3_sim/maps/map.yaml"

MODE="gazebo"
WORLD_LAUNCH="turtlebot3_world.launch.py"
MAP_PATH="$DEFAULT_MAP"
MODEL="${TURTLEBOT3_MODEL:-burger}"
DOMAIN_ID="${ROS_DOMAIN_ID:-2}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--mode gazebo|nav|nav-only] [--world LAUNCH_FILE] [--map PATH] [--model MODEL] [--ros-domain-id ID] [--rviz|--no-rviz]

Modes:
  gazebo   Launch Gazebo only.
  nav      Launch Gazebo first, then launch Nav2. RViz is enabled by default.
  nav-only Launch Nav2 against an already running Gazebo runtime.

Examples:
  bash robotics/scripts/run_sim.sh --mode gazebo
  bash robotics/scripts/run_sim.sh --mode nav --map "robotics/ros_ws/src/roboclaw_tb3_sim/maps/map.yaml"
  bash robotics/scripts/run_sim.sh --mode nav
  bash robotics/scripts/run_sim.sh --mode nav-only
  bash robotics/scripts/run_sim.sh --mode nav --no-rviz
EOF
}

GAZEBO_PID=""
WITH_RVIZ="true"

cleanup() {
  if [[ -n "$GAZEBO_PID" ]] && kill -0 "$GAZEBO_PID" 2>/dev/null; then
    echo "Stopping Gazebo (pid=$GAZEBO_PID)"
    kill "$GAZEBO_PID" 2>/dev/null || true
    wait "$GAZEBO_PID" 2>/dev/null || true
  fi
}

launch_gazebo_background() {
  echo "Launching Gazebo in background with turtlebot3_gazebo/$WORLD_LAUNCH"
  ros2 launch turtlebot3_gazebo "$WORLD_LAUNCH" &
  GAZEBO_PID=$!
  echo "Gazebo pid=$GAZEBO_PID"
}

resolve_path() {
  local input_path="$1"

  if [[ "$input_path" = /* ]]; then
    printf '%s\n' "$input_path"
    return 0
  fi

  if [[ -e "$input_path" ]]; then
    printf '%s\n' "$(cd "$(dirname "$input_path")" && pwd)/$(basename "$input_path")"
    return 0
  fi

  printf '%s\n' "$REPO_ROOT/$input_path"
}

get_nav_params_file() {
  local nav_prefix
  local ros_distro
  local distro_params
  local legacy_params

  nav_prefix="$(ros2 pkg prefix turtlebot3_navigation2)"
  ros_distro="${ROS_DISTRO:-humble}"
  distro_params="$nav_prefix/share/turtlebot3_navigation2/param/$ros_distro/$TURTLEBOT3_MODEL.yaml"
  legacy_params="$nav_prefix/share/turtlebot3_navigation2/param/$TURTLEBOT3_MODEL.yaml"

  if [[ -f "$distro_params" ]]; then
    printf '%s\n' "$distro_params"
    return 0
  fi

  if [[ -f "$legacy_params" ]]; then
    printf '%s\n' "$legacy_params"
    return 0
  fi

  echo "Nav2 params file not found for model '$TURTLEBOT3_MODEL'." >&2
  exit 1
}

launch_nav2() {
  if [[ "$WITH_RVIZ" == "true" ]]; then
    echo "Launching Nav2 with RViz using map: $MAP_PATH"
    ros2 launch turtlebot3_navigation2 navigation2.launch.py \
      map:="$MAP_PATH" \
      use_sim_time:=True
    return 0
  fi

  local params_file
  params_file="$(get_nav_params_file)"
  echo "Launching Nav2 without RViz using map: $MAP_PATH"
  echo "Using params file: $params_file"
  ros2 launch nav2_bringup bringup_launch.py \
    map:="$MAP_PATH" \
    params_file:="$params_file" \
    use_sim_time:=True
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --world)
      WORLD_LAUNCH="${2:-}"
      shift 2
      ;;
    --map)
      MAP_PATH="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --ros-domain-id)
      DOMAIN_ID="${2:-}"
      shift 2
      ;;
    --rviz)
      WITH_RVIZ="true"
      shift
      ;;
    --no-rviz)
      WITH_RVIZ="false"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

MAP_PATH="$(resolve_path "$MAP_PATH")"

if [[ ! -f "$ROS_SETUP" ]]; then
  echo "ROS 2 setup not found: $ROS_SETUP" >&2
  exit 1
fi

if [[ ! -f "$WS_SETUP" ]]; then
  echo "Workspace setup not found: $WS_SETUP" >&2
  echo "Run robotics/scripts/build_ws.sh first." >&2
  exit 1
fi

if [[ "$MODE" == "nav" || "$MODE" == "nav-only" ]] && [[ ! -f "$MAP_PATH" ]]; then
  echo "Map file not found: $MAP_PATH" >&2
  exit 1
fi

set +u
source "$ROS_SETUP"
source /usr/share/gazebo/setup.sh
source "$WS_SETUP"
set -u

export TURTLEBOT3_MODEL="$MODEL"
export ROS_DOMAIN_ID="$DOMAIN_ID"

echo "REPO_ROOT=$REPO_ROOT"
echo "TURTLEBOT3_MODEL=$TURTLEBOT3_MODEL"
echo "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "MODE=$MODE"
echo "WITH_RVIZ=$WITH_RVIZ"

case "$MODE" in
  gazebo)
    echo "Launching Gazebo with turtlebot3_gazebo/$WORLD_LAUNCH"
    exec ros2 launch turtlebot3_gazebo "$WORLD_LAUNCH"
    ;;
  nav)
    trap cleanup EXIT INT TERM
    launch_gazebo_background
    echo "Waiting for Gazebo to start before launching Nav2..."
    sleep 5
    launch_nav2
    ;;
  nav-only)
    echo "Launching Nav2 against an existing Gazebo runtime."
    launch_nav2
    ;;
  *)
    echo "Invalid mode: $MODE" >&2
    usage >&2
    exit 1
    ;;
esac
