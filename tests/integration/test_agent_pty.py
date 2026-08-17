"""PTY integration tests for ``roboclaw agent``.

These tests require pexpect (``pip install pexpect``).
Mark: ``@pytest.mark.pty`` so they can be selected / skipped via ``-m pty``.

**Lifecycle tests** (simulated_agent_child) validate terminal I/O behaviour
(startup banner, exit, interrupt handling, CJK output) without an LLM.

**Flow tests** (simulated_agent) exercise full agent → tool → stub pipelines
using a stub LLM provider that maps keywords to tool calls.
"""

from __future__ import annotations

import pytest

pexpect = pytest.importorskip("pexpect")


# ===================================================================
# Lifecycle tests — raw pexpect child, no stub provider
# ===================================================================


@pytest.mark.pty
def test_agent_startup_and_exit(simulated_agent_child) -> None:
    """Agent should print a startup banner, accept ``exit``, and quit."""
    child = simulated_agent_child
    idx = child.expect([r"You:", pexpect.TIMEOUT, pexpect.EOF], timeout=15)
    assert idx == 0, "Agent did not reach interactive prompt"

    child.sendline("exit")
    child.expect(pexpect.EOF, timeout=10)


@pytest.mark.pty
def test_agent_quit_command(simulated_agent_child) -> None:
    """``quit`` should also terminate the agent."""
    child = simulated_agent_child
    child.expect(r"You:", timeout=15)
    child.sendline("quit")
    child.expect(pexpect.EOF, timeout=10)


@pytest.mark.pty
def test_agent_ctrl_c(simulated_agent_child) -> None:
    """Ctrl-C at the prompt should not crash the agent."""
    child = simulated_agent_child
    child.expect(r"You:", timeout=15)
    child.sendintr()
    child.expect(r"Received SIGINT, goodbye!", timeout=10)
    child.close(force=True)


@pytest.mark.pty
def test_agent_cjk_input(simulated_agent_child) -> None:
    """CJK characters should not cause encoding crashes."""
    child = simulated_agent_child
    child.expect(r"You:", timeout=15)
    child.sendline("你好")
    idx = child.expect([r"You:", r"Error", pexpect.TIMEOUT, pexpect.EOF], timeout=15)
    assert idx in (0, 1), f"Agent crashed or timed out on CJK input (idx={idx})"


# ===================================================================
# Flow tests — SimulatedAgent with stub LLM provider
# ===================================================================


@pytest.mark.pty
def test_agent_identify_flow_handles_no_hardware(simulated_agent) -> None:
    """Identify should return cleanly when no matching hardware is found."""
    simulated_agent.write_setup(arms=[])
    simulated_agent.start()

    simulated_agent.sendline("identify the moved arm")
    simulated_agent.expect("Select embodiment type:", timeout=30)
    simulated_agent.sendline("1")
    simulated_agent.expect(r"Select model \(1/2\):", timeout=30)
    simulated_agent.sendline("1")
    simulated_agent.expect("No matching hardware found.", timeout=30)
    simulated_agent.expect_prompt()

    arms = simulated_agent.read_setup()["arms"]
    assert arms == []
