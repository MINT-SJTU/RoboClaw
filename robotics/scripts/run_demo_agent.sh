#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
WS_SETUP="$REPO_ROOT/robotics/ros_ws/install/setup.bash"
DEMO_CONFIG="${ROBOCLAW_DEMO_CONFIG:-$HOME/.roboclaw/config_demo_navigation.json}"
DEMO_WORKSPACE="${ROBOCLAW_DEMO_WORKSPACE:-$HOME/.roboclaw/workspace_demo_navigation}"
DEMO_PROVIDER="${ROBOCLAW_DEMO_PROVIDER:-openrouter}"
DEMO_MODEL="${ROBOCLAW_DEMO_MODEL:-openrouter/openai/gpt-4.1-mini}"

if [ ! -f "$ROS_SETUP" ]; then
  echo "ROS setup not found: $ROS_SETUP"
  exit 1
fi

if [ ! -f "$WS_SETUP" ]; then
  echo "Workspace setup not found: $WS_SETUP"
  echo "Run robotics/scripts/build_ws.sh first."
  exit 1
fi

has_config=0
for arg in "$@"; do
  case "$arg" in
    -c|--config|--config=*)
      has_config=1
      ;;
  esac
done

bootstrap_demo_config() {
  if [ -f "$DEMO_CONFIG" ]; then
    return 0
  fi

  mkdir -p "$(dirname "$DEMO_CONFIG")"
  python - "$DEMO_CONFIG" "$DEMO_WORKSPACE" "$DEMO_PROVIDER" "$DEMO_MODEL" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser()
workspace = sys.argv[2]
provider = sys.argv[3]
model = sys.argv[4]

providers = {
    name: {"apiKey": "", "apiBase": None}
    for name in [
        "custom",
        "azureOpenai",
        "anthropic",
        "openai",
        "openrouter",
        "deepseek",
        "groq",
        "zhipu",
        "dashscope",
        "vllm",
        "ollama",
        "gemini",
        "moonshot",
        "minimax",
        "aihubmix",
        "siliconflow",
        "volcengine",
        "volcengineCodingPlan",
        "byteplus",
        "byteplusCodingPlan",
        "openaiCodex",
        "githubCopilot",
    ]
}

providers.setdefault(provider, {"apiKey": "", "apiBase": None})

data = {
    "agents": {
        "defaults": {
            "workspace": workspace,
            "model": model,
            "provider": provider,
            "maxTokens": 8192,
            "contextWindowTokens": 65536,
            "temperature": 0.1,
            "maxToolIterations": 40,
            "reasoningEffort": None,
        }
    },
    "channels": {
        "sendProgress": True,
        "sendToolHints": False,
    },
    "providers": providers,
    "gateway": {
        "host": "0.0.0.0",
        "port": 18790,
        "heartbeat": {"enabled": True, "intervalS": 1800},
    },
    "tools": {
        "web": {"proxy": None, "search": {"provider": "brave", "apiKey": "", "baseUrl": "", "maxResults": 5}},
        "exec": {"timeout": 60, "pathAppend": ""},
        "restrictToWorkspace": False,
        "mcpServers": {},
    },
}

config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  echo "Created demo config: $DEMO_CONFIG"
}

if [ "$has_config" -eq 0 ]; then
  bootstrap_demo_config
fi

export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-burger}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-2}"

set +u
source "$ROS_SETUP"
source /usr/share/gazebo/setup.sh
source "$WS_SETUP"
set -u

cd "$REPO_ROOT"
if [ "$has_config" -eq 0 ]; then
  python robotics/scripts/run_demo_agent.py --config "$DEMO_CONFIG" "$@"
else
  python robotics/scripts/run_demo_agent.py "$@"
fi
