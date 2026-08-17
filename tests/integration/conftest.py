"""Fixtures for PTY-based integration tests.

These tests spawn ``roboclaw agent`` in a pseudo-terminal via pexpect so
that we can exercise the interactive CLI behaviour (spinners, Ctrl-C,
CJK rendering) without a real LLM provider.

A stub LLM provider is injected via ``ROBOCLAW_STUB_LLM``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

pexpect = pytest.importorskip("pexpect")

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# SimulatedAgent — wraps pexpect with setup helpers
# ---------------------------------------------------------------------------


@dataclass
class SimulatedAgent:
    """Manages a pexpect-driven ``roboclaw agent`` process with a stub LLM."""

    env: dict[str, str]
    home: Path
    setup_path: Path
    child: Any = field(default=None, init=False)

    def write_setup(
        self,
        *,
        arms: list[dict[str, Any]] | None = None,
        cameras: list[dict[str, Any]] | None = None,
    ) -> None:
        manifest = self.read_setup()
        manifest["arms"] = arms or []
        manifest["cameras"] = cameras or []
        self.setup_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def read_setup(self) -> dict[str, Any]:
        if not self.setup_path.exists():
            return {}
        return json.loads(self.setup_path.read_text(encoding="utf-8"))

    def start(self) -> "SimulatedAgent":
        if self.child is not None:
            raise RuntimeError("agent already started")
        self.child = pexpect.spawn(
            sys.executable,
            ["-m", "roboclaw.cli.commands", "agent", "--no-markdown"],
            cwd=str(REPO_ROOT),
            env=self.env,
            encoding="utf-8",
            timeout=20,
        )
        self.child.expect("You:", timeout=15)
        return self

    def sendline(self, text: str) -> None:
        self.child.sendline(text)

    def expect(self, pattern: str, timeout: int = 15) -> int:
        return self.child.expect(pattern, timeout=timeout)

    def expect_prompt(self, timeout: int = 30) -> None:
        self.child.expect("You:", timeout=timeout)

    def close(self) -> None:
        if self.child is None:
            return
        if self.child.isalive():
            self.child.sendline("exit")
            try:
                self.child.expect(pexpect.EOF, timeout=10)
            except (pexpect.TIMEOUT, pexpect.EOF):
                self.child.terminate(force=True)
        self.child = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simulated_home(tmp_path: Path) -> Path:
    """Return a temporary ROBOCLAW_HOME directory with a seeded config.

    This fixture is for lightweight lifecycle tests that do NOT need the
    stub LLM provider (e.g. startup/exit, Ctrl-C).
    """
    home = tmp_path / ".roboclaw"
    home.mkdir()
    config = {
        "agents": {"defaults": {"model": "openai/gpt-4o-mini"}},
        "providers": {"openai": {"apiKey": "sk-test-fake-key"}},
    }
    (home / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return home


@pytest.fixture()
def simulated_agent_child(simulated_home: Path):
    """Spawn ``roboclaw agent`` as a raw pexpect child (no stub provider).

    Yields a *pexpect.spawn* child.  Used by lifecycle tests.
    """
    config_path = str(simulated_home / "config.json")
    env = os.environ.copy()
    env["ROBOCLAW_HOME"] = str(simulated_home)
    env["ROBOCLAW_STUB"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"

    child = pexpect.spawn(
        sys.executable,
        ["-m", "roboclaw.cli.commands", "agent", "--config", config_path],
        env=env,
        encoding="utf-8",
        timeout=30,
    )
    yield child

    if child.isalive():
        child.terminate(force=True)


@pytest.fixture()
def simulated_agent(tmp_path: Path):
    """Full SimulatedAgent with a stub LLM provider.

    Initialises workspace via ``roboclaw dev reset``, then yields a
    SimulatedAgent instance.  Call ``agent.start()`` to spawn the process.
    """
    home = tmp_path / "home"
    roboclaw_home = home / ".roboclaw"
    setup_path = roboclaw_home / "workspace" / "embodied" / "manifest.json"

    env = os.environ.copy()
    # Ensure project root is on PYTHONPATH so tests.integration.stub_llm is importable
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), existing_pp) if p
    )
    env.update(
        HOME=str(home),
        PYTHONUNBUFFERED="1",
        ROBOCLAW_HOME=str(roboclaw_home),
        ROBOCLAW_STUB="1",
        ROBOCLAW_STUB_LLM="tests.integration.stub_llm",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        NO_COLOR="1",
    )

    # Bootstrap workspace with dev reset
    result = subprocess.run(
        [sys.executable, "-m", "roboclaw", "dev", "reset", "--yes",
         "--model", "stub-model", "--provider", "custom",
         "--api-base", "http://stub.invalid/v1", "--api-key", "test-key"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dev reset failed (exit {result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    agent = SimulatedAgent(
        env=env,
        home=home,
        setup_path=setup_path,
    )
    try:
        yield agent
    finally:
        agent.close()
