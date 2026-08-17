"""Wrapper entrypoint that patches LeRobot before LeRobot CLI actions."""

from __future__ import annotations

import sys

from roboclaw.embodied.command.headless_patch import apply_headless_patch

_HEADLESS_PATCH_ACTIONS = frozenset({"record", "replay"})


def record(argv: list[str] | None = None) -> None:
    _run("record", argv)


def replay(argv: list[str] | None = None) -> None:
    _run("replay", argv)


def teleoperate(argv: list[str] | None = None) -> None:
    _run("teleoperate", argv)


def calibrate(argv: list[str] | None = None) -> None:
    _run("calibrate", argv)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit(
            "Usage: python -m roboclaw.embodied.command.wrapper "
            "<record|replay|teleoperate|calibrate> [args...]"
        )
    action = args[0]
    if action not in {"record", "replay", "teleoperate", "calibrate"}:
        raise SystemExit(f"Unsupported action: {action}")
    _run(action, args[1:])


def _run(action: str, argv: list[str] | None = None) -> None:
    args = list([] if argv is None else argv)
    if action in _HEADLESS_PATCH_ACTIONS:
        apply_headless_patch()
    original_argv = sys.argv[:]
    try:
        sys.argv = [f"lerobot-{action}", *args]
        if action == "record":
            from lerobot.scripts import lerobot_record as module
        elif action == "replay":
            from lerobot.scripts import lerobot_replay as module
        elif action == "teleoperate":
            from lerobot.scripts import lerobot_teleoperate as module
        else:
            from lerobot.scripts import lerobot_calibrate as module
        module.main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
