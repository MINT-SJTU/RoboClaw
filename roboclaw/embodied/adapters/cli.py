"""CLI adapter — keyboard-driven terminal interaction for teleop/record.

Bridges the agent's tty_handoff mechanism to EmbodiedService, which uses
OperationEngine internally.  The CLI adapter owns:
- TTY handoff lifecycle (start/stop labels)
- Raw terminal mode for key capture
- Interactive polling loop (read keys, print status)
- Routing key presses to service episode-control methods

Everything else (subprocess management, state machine, episode tracking)
stays inside OperationEngine via EmbodiedService.
"""

from __future__ import annotations

import asyncio
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager
from typing import Any


@contextmanager
def _raw_terminal():
    """Switch stdin to raw mode, restore on exit."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _read_key_nonblocking() -> str | None:
    """Read a keypress from stdin without blocking.

    Returns one of: "right", "left", "esc", "ctrl_c", or None.
    """
    fd = sys.stdin.fileno()
    if not select.select([fd], [], [], 0)[0]:
        return None
    ch = os.read(fd, 1)
    if ch == b"\x03":
        return "ctrl_c"
    if ch == b"\x1b":
        # Could be a bare ESC or the start of an arrow sequence
        if not select.select([fd], [], [], 0.05)[0]:
            return "esc"
        ch2 = os.read(fd, 1)
        if ch2 == b"[":
            ch3 = os.read(fd, 1)
            if ch3 == b"C":
                return "right"
            if ch3 == b"D":
                return "left"
        return "esc"
    return None


def _format_status_line(status: dict[str, Any]) -> str:
    """Build a single-line status string from OperationEngine status."""
    state = status.get("state", "idle")
    if state == "idle":
        return "  idle"
    if state == "preparing":
        return "  preparing..."
    if state == "teleoperating":
        elapsed = status.get("elapsed_seconds", 0)
        return f"  teleoperating  | {elapsed:.0f}s"

    # Recording states
    phase = status.get("episode_phase", "")
    current = status.get("current_episode", 0)
    target = status.get("target_episodes", 0)
    saved = status.get("saved_episodes", 0)
    return f"  Episode {current}/{target} | Saved: {saved} | {phase or state}"


async def run_cli_session(
    service: Any,
    action: str,
    manifest: dict[str, Any],
    kwargs: dict[str, Any],
    tty_handoff: Any,
) -> str:
    """Drive a teleop/record session from the terminal via EmbodiedService.

    Parameters
    ----------
    service:
        EmbodiedService instance.
    action:
        "teleoperate" or "record".
    manifest:
        Current hardware manifest dict (unused directly; OperationEngine loads
        its own via load_manifest).
    kwargs:
        Action-specific keyword args (fps, task, num_episodes, etc.).
    tty_handoff:
        Async callable ``(start: bool, label: str) -> None`` that the agent
        loop uses to claim/release the terminal.
    """
    label = f"lerobot-{action}"
    await tty_handoff(start=True, label=label)
    try:
        return await _interactive_loop(service, action, kwargs)
    finally:
        if service.busy:
            await service.stop()
        await tty_handoff(start=False, label=label)


async def _interactive_loop(
    service: Any,
    action: str,
    kwargs: dict[str, Any],
) -> str:
    """Start operation, capture keys, print status until done."""
    if action == "teleoperate":
        await service.start_teleop(fps=kwargs.get("fps", 30))
        print("Teleoperating... Press Ctrl+C to stop.\n")
    else:
        dataset_name = await service.start_recording(
            task=kwargs.get("task", "default_task"),
            num_episodes=kwargs.get("num_episodes", 10),
            fps=kwargs.get("fps", 30),
            episode_time_s=kwargs.get("episode_time_s", 300),
            reset_time_s=kwargs.get("reset_time_s", 10),
        )
        print(f"Recording -> {dataset_name}")
        print("  -> / <- = save / discard | ESC = stop\n")

    with _raw_terminal():
        while service.busy:
            key = _read_key_nonblocking()
            await _handle_key(service, key, action)
            status = service.get_status()
            print(f"\r{_format_status_line(status)}", end="", flush=True)
            await asyncio.sleep(0.05)

    print()  # newline after status line
    return _format_result(service, action)


async def _handle_key(service: Any, key: str | None, action: str) -> None:
    """Route a captured keypress to the appropriate service method."""
    if key is None:
        return
    if key == "ctrl_c" or key == "esc":
        await service.stop()
    elif key == "right" and action == "record":
        await service.save_episode()
    elif key == "left" and action == "record":
        await service.discard_episode()


def _format_result(service: Any, action: str) -> str:
    """Build a human-readable result string after the session ends."""
    status = service.get_status()
    error = status.get("error", "")
    if error:
        return f"{action.capitalize()} failed: {error}"
    if action == "teleoperate":
        return "Teleoperation finished."
    saved = status.get("saved_episodes", 0)
    dataset = status.get("dataset")
    if dataset:
        return f"Recording finished. {saved} episodes saved to {dataset}."
    return f"Recording finished. {saved} episodes saved."
