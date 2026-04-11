"""Static wiring checks for the standalone Phase 5B demo integration."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_main_context_is_restored_without_demo_hooks() -> None:
    content = _read("roboclaw/agent/context.py")

    assert "extra_system_prompt" not in content
    assert "# Extra System Guidance" not in content


def test_demo_context_appends_navigation_prompt() -> None:
    content = _read("roboclaw/agent/context_nav.py")

    assert "NavigationDemoContextBuilder" in content
    assert "DEMO_NAVIGATION_PROMPT" in content
    assert "# Extra System Guidance" in content


def test_demo_loop_registers_demo_tools_without_arm_embodied_tools() -> None:
    content = _read("roboclaw/agent/loop_nav.py")

    assert "class NavigationDemoAgentLoop(AgentLoop)" in content
    assert "register_demo_tools(self.tools)" in content
    assert "create_embodied_tools" not in content
    assert "self.context = NavigationDemoContextBuilder(workspace)" in content


def test_demo_launcher_uses_demo_navigation_flag() -> None:
    content = _read("robotics/scripts/run_demo_agent.sh")

    assert "config_demo_navigation.json" in content
    assert "workspace_demo_navigation" in content
    assert "Created demo config:" in content
    assert "python robotics/scripts/run_demo_agent.py --config \"$DEMO_CONFIG\"" in content
    assert "provider_block[\"apiKey\"]" not in content


def test_demo_python_runner_uses_navigation_demo_loop() -> None:
    content = _read("robotics/scripts/run_demo_agent.py")

    assert "NavigationDemoAgentLoop" in content
    assert "Demo navigation mode" in content
    assert '"--verbose"' in content
    assert "def _should_exit" in content
    assert "re.fullmatch(r\"(?:ok|okay)[,\\s]+(?:exit|quit)\"" in content
    assert "_apply_demo_env_overrides" in content
    assert "_ensure_demo_credentials" in content
    assert "_configure_demo_logging" in content
    assert 'level="INFO" if verbose else "WARNING"' in content
    assert "getpass.getpass" in content
    assert "_cleanup_demo_runtime" in content
    assert 'loop.tools.get("embodied_simulation")' in content
    assert 'execute(action="shutdown")' in content
    assert "except KeyboardInterrupt" in content
