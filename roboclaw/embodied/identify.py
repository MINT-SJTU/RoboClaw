"""Interactive arm identification — detect which serial port a user is moving.

Run via: python -m roboclaw.embodied.identify <scanned_ports_json>
"""

from __future__ import annotations

import json
import sys

from roboclaw.embodied.embodiment.arm.registry import all_arm_types
from roboclaw.embodied.hardware.discovery import HardwareDiscovery
from roboclaw.embodied.hardware.motion import (
    MOTION_THRESHOLD,
    detect_motion,
    read_positions_for_port,
    resolve_port_by_id,
    resolve_port_path,
)
from roboclaw.embodied.interface.serial import SerialInterface

# Build dynamic menu from all registered arm types.
_ALL_TYPES = all_arm_types()
_ARM_TYPE_CHOICES: dict[str, str] = {}
for _idx, _t in enumerate(_ALL_TYPES, 1):
    _ARM_TYPE_CHOICES[str(_idx)] = _t
    _ARM_TYPE_CHOICES[_t] = _t
_leader = next(t for t in _ALL_TYPES if "leader" in t)
_follower = next(t for t in _ALL_TYPES if "follower" in t)
_ARM_TYPE_CHOICES.update({"leader": _leader, "follower": _follower, "主": _leader, "从": _follower})


def _read_line(prompt: str) -> str:
    """Write a prompt and read one UTF-8 line from stdin without readline."""
    sys.stdout.write(prompt)
    sys.stdout.flush()
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError
    return line.decode("utf-8").rstrip("\r\n")


def _choose_arm_type() -> str:
    """Prompt until a valid arm type is selected."""
    print("Choose arm type:")
    for i, t in enumerate(_ALL_TYPES, 1):
        label = "主臂" if "leader" in t else "从臂"
        print(f"  {i}. {t} ({label})")
    while True:
        choice = _read_line("Select: ").strip().casefold()
        arm_type = _ARM_TYPE_CHOICES.get(choice)
        if arm_type is not None:
            return arm_type
        print(f"Invalid choice. Enter 1-{len(_ALL_TYPES)} or a type name.")


def _choose_alias(existing_aliases: set[str]) -> str:
    """Prompt until a non-empty, unique alias is provided."""
    while True:
        alias = _read_line("Name for this arm: ").strip()
        if not alias:
            print("Alias is required.")
            continue
        if alias in existing_aliases:
            print(f"Alias '{alias}' already exists. Choose a different name.")
            continue
        return alias


def _confirm(prompt: str) -> bool:
    """Read a Y/n style prompt with validation."""
    while True:
        answer = _read_line(prompt).strip().casefold()
        if answer in ("", "y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please enter Y or n.")


def _find_moved_port(
    ports: list[SerialInterface], baselines: dict[str, dict[int, int]],
) -> SerialInterface | None:
    """Read current positions, find the port with largest motion above threshold."""
    from roboclaw.embodied.stub import is_stub_mode, stub_moved_port

    if is_stub_mode():
        return stub_moved_port(ports)

    best_port: SerialInterface | None = None
    best_delta = 0
    for port in ports:
        path = resolve_port_path(port)
        current = read_positions_for_port(port)
        delta = detect_motion(baselines[path], current)
        if delta > MOTION_THRESHOLD and delta > best_delta:
            best_delta = delta
            best_port = port
    return best_port


def _save_arm(alias: str, arm_type: str, port: SerialInterface) -> None:
    """Save arm to manifest via set_arm."""
    from roboclaw.embodied.manifest.helpers import set_arm

    port_id = resolve_port_by_id(port)
    set_arm(alias, arm_type, port_id)
    print(f"Saved: {alias} ({arm_type}) on {port_id}")


def _identify_one_arm(
    ports: list[SerialInterface], existing_aliases: set[str],
) -> dict | None:
    """Run one round of identification."""
    baselines = {resolve_port_path(p): read_positions_for_port(p) for p in ports}
    _read_line("\nMove one arm, then press Enter.")
    moved = _find_moved_port(ports, baselines)
    if moved is None:
        print("No movement detected. Try again.")
        return None
    port_id = resolve_port_by_id(moved)
    print(f"\nDetected movement on: {port_id}")

    while True:
        arm_type = _choose_arm_type()
        alias = _choose_alias(existing_aliases)
        print(f"Arm: {alias} ({arm_type}) on {port_id}")
        if _confirm("OK? (Y/n): "):
            return {
                "alias": alias,
                "arm_type": arm_type,
                "port": moved,
                "port_id": port_id,
            }
        print("Redoing this arm.")


def main() -> None:
    """Interactive identify loop. Expects scanned_ports JSON as argv[1]."""
    if len(sys.argv) < 2:
        print("Usage: python -m roboclaw.embodied.identify <scanned_ports_json>")
        sys.exit(1)

    try:
        raw_ports = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(f"Invalid scanned_ports JSON: {exc}")
        sys.exit(1)
    if not isinstance(raw_ports, list):
        print("scanned_ports_json must decode to a list.")
        sys.exit(1)
    if not raw_ports:
        print("No serial ports provided.")
        sys.exit(1)

    print("Probing ports for motors...")
    discovery = HardwareDiscovery()
    ports = discovery.discover_all()
    if not ports:
        print("No motors found on any port.")
        sys.exit(1)

    print(f"Found {len(ports)} port(s) with motors.")
    batch_aliases: set[str] = set()
    staged: list[dict] = []

    try:
        while ports:
            identified = _identify_one_arm(ports, batch_aliases)
            if identified is None:
                continue
            ports.remove(identified["port"])
            staged.append(identified)
            batch_aliases.add(identified["alias"])
            if not ports:
                break
            if not _confirm("Continue? (Y/n): "):
                break
    except EOFError:
        print("\nInput closed.")

    for arm in staged:
        _save_arm(arm["alias"], arm["arm_type"], arm["port"])

    print(f"\nDone. Identified {len(staged)} arm(s).")
    for arm in staged:
        print(f"  - {arm['alias']} ({arm['arm_type']}) on {arm['port_id']}")


if __name__ == "__main__":
    main()
