"""Regression tests for the LeRobot wrapper entrypoint."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from roboclaw.embodied import lerobot_wrapper


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str] | str]:
    state: dict[str, list[str] | str] = {}

    lerobot = ModuleType("lerobot")
    lerobot.__path__ = []
    scripts = ModuleType("lerobot.scripts")
    scripts.__path__ = []
    setattr(lerobot, "scripts", scripts)

    def _make_script(name: str) -> ModuleType:
        module = ModuleType(name)

        def _main() -> None:
            state["module"] = name
            state["argv"] = sys.argv[:]

        module.main = _main
        return module

    modules = {
        "lerobot.scripts.lerobot_record": _make_script("lerobot.scripts.lerobot_record"),
        "lerobot.scripts.lerobot_replay": _make_script("lerobot.scripts.lerobot_replay"),
        "lerobot.scripts.lerobot_teleoperate": _make_script("lerobot.scripts.lerobot_teleoperate"),
        "lerobot.scripts.lerobot_calibrate": _make_script("lerobot.scripts.lerobot_calibrate"),
    }

    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.scripts", scripts)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        setattr(scripts, name.rsplit(".", 1)[-1], module)

    return state


@pytest.mark.parametrize(
    ("action", "module_name", "expect_patch"),
    [
        ("record", "lerobot.scripts.lerobot_record", True),
        ("replay", "lerobot.scripts.lerobot_replay", False),
        ("teleoperate", "lerobot.scripts.lerobot_teleoperate", False),
        ("calibrate", "lerobot.scripts.lerobot_calibrate", False),
    ],
)
def test_run_applies_headless_patch_only_for_record(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    module_name: str,
    expect_patch: bool,
) -> None:
    state = _install_fake_lerobot(monkeypatch)
    patch_calls: list[str] = []
    monkeypatch.setattr(lerobot_wrapper, "apply_headless_patch", lambda: patch_calls.append(action))

    original_argv = sys.argv[:]
    args = ["--flag=value"]

    lerobot_wrapper._run(action, args)

    assert patch_calls == ([action] if expect_patch else [])
    assert state["module"] == module_name
    assert state["argv"] == [f"lerobot-{action}", *args]
    assert sys.argv == original_argv
