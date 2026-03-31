"""Interactive arm identification — detect which serial port a user is moving.

Run via: python -m roboclaw.embodied.identify <scanned_ports_json>
"""

from __future__ import annotations

import json
import sys

from roboclaw.embodied.scan import restore_stderr, suppress_stderr
from roboclaw.embodied.setup import _ARM_TYPES

PRESENT_POS_ADDR = 56
PRESENT_POS_LEN = 2
MOTOR_IDS = list(range(1, 7))
DEFAULT_BAUDRATE = 1_000_000
MOTION_THRESHOLD = 50

# Derive the menu from the canonical _ARM_TYPES tuple in setup.py.
_ARM_TYPE_CHOICES: dict[str, str] = {}
for _idx, _t in enumerate(_ARM_TYPES, 1):
    _ARM_TYPE_CHOICES[str(_idx)] = _t
    _ARM_TYPE_CHOICES[_t] = _t
_leader = next(t for t in _ARM_TYPES if "leader" in t)
_follower = next(t for t in _ARM_TYPES if "follower" in t)
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
    for i, t in enumerate(_ARM_TYPES, 1):
        label = "主臂" if "leader" in t else "从臂"
        print(f"  {i}. {t} ({label})")
    while True:
        choice = _read_line("Select: ").strip().casefold()
        arm_type = _ARM_TYPE_CHOICES.get(choice)
        if arm_type is not None:
            return arm_type
        print(f"Invalid choice. Enter 1-{len(_ARM_TYPES)} or a type name.")


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


def probe_port(port_path: str, baudrate: int = DEFAULT_BAUDRATE) -> list[int]:
    """Try reading Present_Position for motor IDs 1-6. Return responding IDs."""
    from roboclaw.embodied.stub import is_stub_mode, stub_motor_ids

    if is_stub_mode():
        return stub_motor_ids(port_path)
    import scservo_sdk as scs

    handler = scs.PortHandler(port_path)
    if not handler.openPort():
        return []
    handler.setBaudRate(baudrate)
    packet = scs.PacketHandler(0)
    found = []
    for mid in MOTOR_IDS:
        val, result, _ = packet.read2ByteTxRx(handler, mid, PRESENT_POS_ADDR)
        if result == scs.COMM_SUCCESS:
            found.append(mid)
    handler.closePort()
    return found


def read_positions(
    port_path: str, motor_ids: list[int], baudrate: int = DEFAULT_BAUDRATE,
) -> dict[int, int]:
    """Read Present_Position (addr=56, len=2) for each motor ID."""
    from roboclaw.embodied.stub import is_stub_mode

    if is_stub_mode():
        return {mid: 0 for mid in motor_ids}
    import scservo_sdk as scs

    handler = scs.PortHandler(port_path)
    if not handler.openPort():
        return {}
    handler.setBaudRate(baudrate)
    packet = scs.PacketHandler(0)
    positions: dict[int, int] = {}
    for mid in motor_ids:
        val, result, _ = packet.read2ByteTxRx(handler, mid, PRESENT_POS_ADDR)
        if result == scs.COMM_SUCCESS:
            positions[mid] = val
    handler.closePort()
    return positions


def read_positions_dynamixel(
    port_path: str, motor_ids: list[int], baudrate: int = DEFAULT_BAUDRATE,
) -> dict[int, int]:
    """Read Dynamixel Present_Position (addr=132, len=4) for each motor ID."""
    from roboclaw.embodied.stub import is_stub_mode

    if is_stub_mode():
        return {mid: 0 for mid in motor_ids}
    import dynamixel_sdk as dxl

    handler = dxl.PortHandler(port_path)
    if not handler.openPort():
        return {}
    handler.setBaudRate(baudrate)
    packet = dxl.PacketHandler(2.0)
    positions: dict[int, int] = {}
    for mid in motor_ids:
        val, result, _ = packet.read4ByteTxRx(handler, mid, 132)
        if result == dxl.COMM_SUCCESS:
            positions[mid] = val
    handler.closePort()
    return positions


def detect_motion(baseline: dict[int, int], current: dict[int, int]) -> int:
    """Compute total absolute delta between baseline and current positions."""
    total = 0
    for mid, base_val in baseline.items():
        cur_val = current.get(mid)
        if cur_val is None:
            continue
        total += abs(cur_val - base_val)
    return total


def _resolve_port_path(port: dict) -> str:
    """Pick the best device path from a scanned port entry."""
    return port.get("dev") or port.get("by_id") or port.get("by_path", "")


def _resolve_port_by_id(port: dict) -> str:
    """Pick a stable identifier for set_arm (prefer by_id)."""
    return port.get("by_id") or port.get("dev") or port.get("by_path", "")


def probe_port_dynamixel(port_path: str, baudrate: int = DEFAULT_BAUDRATE) -> list[int]:
    """Try reading Present_Position for Dynamixel motor IDs 1-6. Return responding IDs."""
    from roboclaw.embodied.stub import is_stub_mode, stub_motor_ids

    if is_stub_mode():
        return stub_motor_ids(port_path)
    import dynamixel_sdk as dxl

    handler = dxl.PortHandler(port_path)
    if not handler.openPort():
        return []
    handler.setBaudRate(baudrate)
    packet = dxl.PacketHandler(2.0)
    found = []
    for mid in MOTOR_IDS:
        val, result, _ = packet.read4ByteTxRx(handler, mid, 132)
        if result == dxl.COMM_SUCCESS:
            found.append(mid)
    handler.closePort()
    return found


def _probe_single_port(port: dict) -> dict | None:
    """Probe one port for Feetech motors. Returns enriched dict or None."""
    path = _resolve_port_path(port)
    if not path:
        return None
    ids = probe_port(path)
    if not ids:
        return None
    return {**port, "motor_ids": ids, "bus_type": "feetech"}


def _probe_single_port_dynamixel(port: dict) -> dict | None:
    """Probe one port for Dynamixel motors. Returns enriched dict or None."""
    path = _resolve_port_path(port)
    if not path:
        return None
    ids = probe_port_dynamixel(path)
    if not ids:
        return None
    return {**port, "motor_ids": ids, "bus_type": "dynamixel"}


def _filter_motor_ports(scanned_ports: list[dict]) -> list[dict]:
    """Probe each port for Feetech then Dynamixel motors. Attach motor_ids and bus_type."""
    saved = suppress_stderr()
    try:
        feetech = [_probe_single_port(p) for p in scanned_ports]
        feetech = [r for r in feetech if r is not None]
        if feetech:
            return feetech
        dxl = [_probe_single_port_dynamixel(p) for p in scanned_ports]
        return [r for r in dxl if r is not None]
    finally:
        restore_stderr(saved)


def _read_all_baselines(ports: list[dict]) -> dict[str, dict[int, int]]:
    """Read positions for all ports. Returns {dev_path: {motor_id: position}}."""
    baselines: dict[str, dict[int, int]] = {}
    for port in ports:
        path = _resolve_port_path(port)
        if port.get("bus_type") == "dynamixel":
            baselines[path] = read_positions_dynamixel(path, port["motor_ids"])
        else:
            baselines[path] = read_positions(path, port["motor_ids"])
    return baselines


def _find_moved_port(ports: list[dict], baselines: dict[str, dict[int, int]]) -> dict | None:
    """Read current positions, find the port with largest motion above threshold."""
    from roboclaw.embodied.stub import is_stub_mode, stub_moved_port

    if is_stub_mode():
        return stub_moved_port(ports)

    best_port = None
    best_delta = 0
    for port in ports:
        path = _resolve_port_path(port)
        if port.get("bus_type") == "dynamixel":
            current = read_positions_dynamixel(path, port["motor_ids"])
        else:
            current = read_positions(path, port["motor_ids"])
        delta = detect_motion(baselines[path], current)
        if delta > MOTION_THRESHOLD and delta > best_delta:
            best_delta = delta
            best_port = port
    return best_port


def _save_arm(alias: str, arm_type: str, port: dict) -> None:
    """Save arm to setup.json via set_arm."""
    from roboclaw.embodied.setup import set_arm

    port_id = _resolve_port_by_id(port)
    set_arm(alias, arm_type, port_id)
    print(f"Saved: {alias} ({arm_type}) on {port_id}")


def _identify_one_arm(ports: list[dict], existing_aliases: set[str]) -> dict | None:
    """Run one round of identification. Returns staged arm data or None."""
    baselines = _read_all_baselines(ports)
    _read_line("\nMove one arm, then press Enter.")
    moved = _find_moved_port(ports, baselines)
    if moved is None:
        print("No movement detected. Try again.")
        return None
    port_id = _resolve_port_by_id(moved)
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
        scanned_ports = json.loads(sys.argv[1])
    except json.JSONDecodeError as exc:
        print(f"Invalid scanned_ports JSON: {exc}")
        sys.exit(1)
    if not isinstance(scanned_ports, list):
        print("scanned_ports_json must decode to a list.")
        sys.exit(1)
    if not scanned_ports:
        print("No serial ports provided.")
        sys.exit(1)

    print("Probing ports for motors...")
    ports = _filter_motor_ports(scanned_ports)
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
