"""Interactive arm identification — detect which serial port a user is moving.

Run via: python -m roboclaw.embodied.identify <scanned_ports_json>
"""

from __future__ import annotations

import json
import os
import sys

from roboclaw.embodied.scan import restore_stderr, suppress_stderr

PRESENT_POS_ADDR = 56
PRESENT_POS_LEN = 2
MOTOR_IDS = list(range(1, 7))
DEFAULT_BAUDRATE = 1_000_000
MOTION_THRESHOLD = 50


def probe_port(port_path: str, baudrate: int = DEFAULT_BAUDRATE) -> list[int]:
    """Try reading Present_Position for motor IDs 1-6. Return responding IDs."""
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


def _probe_single_port(port: dict) -> dict | None:
    """Probe one port for Feetech motors. Returns enriched dict or None."""
    path = _resolve_port_path(port)
    if not path:
        return None
    ids = probe_port(path)
    if not ids:
        return None
    return {**port, "motor_ids": ids}


def _filter_feetech_ports(scanned_ports: list[dict]) -> list[dict]:
    """Probe each port, keep only those with Feetech motors. Attach motor_ids."""
    saved = suppress_stderr()
    try:
        results = [_probe_single_port(p) for p in scanned_ports]
    finally:
        restore_stderr(saved)
    return [r for r in results if r is not None]


def _read_all_baselines(ports: list[dict]) -> dict[str, dict[int, int]]:
    """Read positions for all ports. Returns {dev_path: {motor_id: position}}."""
    baselines: dict[str, dict[int, int]] = {}
    for port in ports:
        path = _resolve_port_path(port)
        baselines[path] = read_positions(path, port["motor_ids"])
    return baselines


def _find_moved_port(ports: list[dict], baselines: dict[str, dict[int, int]]) -> dict | None:
    """Read current positions, find the port with largest motion above threshold."""
    best_port = None
    best_delta = 0
    for port in ports:
        path = _resolve_port_path(port)
        current = read_positions(path, port["motor_ids"])
        delta = detect_motion(baselines[path], current)
        if delta > MOTION_THRESHOLD and delta > best_delta:
            best_delta = delta
            best_port = port
    return best_port


def _ask_arm_details(port: dict) -> tuple[str, str]:
    """Ask user for arm type and alias. Returns (arm_type, alias)."""
    port_id = _resolve_port_by_id(port)
    print(f"\nDetected movement on: {port_id}")
    arm_type = input("Type (so101_follower/so101_leader): ").strip()
    alias = input("Name for this arm: ").strip()
    return arm_type, alias


def _save_arm(alias: str, arm_type: str, port: dict) -> None:
    """Save arm to setup.json via set_arm."""
    from roboclaw.embodied.setup import set_arm

    port_id = _resolve_port_by_id(port)
    set_arm(alias, arm_type, port_id)
    print(f"Saved: {alias} ({arm_type}) on {port_id}")


def _identify_one_arm(ports: list[dict]) -> dict | None:
    """Run one round of the identify loop. Returns the identified port or None."""
    baselines = _read_all_baselines(ports)
    input("\nMove one arm, then press Enter.")
    moved = _find_moved_port(ports, baselines)
    if moved is None:
        print("No movement detected. Try again.")
        return None
    arm_type, alias = _ask_arm_details(moved)
    if not alias or not arm_type:
        print("Skipped (empty alias or type).")
        return moved
    _save_arm(alias, arm_type, moved)
    return moved


def main() -> None:
    """Interactive identify loop. Expects scanned_ports JSON as argv[1]."""
    if len(sys.argv) < 2:
        print("Usage: python -m roboclaw.embodied.identify <scanned_ports_json>")
        sys.exit(1)

    scanned_ports: list[dict] = json.loads(sys.argv[1])
    if not scanned_ports:
        print("No serial ports provided.")
        sys.exit(1)

    print("Probing ports for Feetech motors...")
    ports = _filter_feetech_ports(scanned_ports)
    if not ports:
        print("No Feetech motors found on any port.")
        sys.exit(1)

    print(f"Found {len(ports)} port(s) with motors.")
    identified: list[str] = []

    while ports:
        moved = _identify_one_arm(ports)
        if moved is None:
            continue
        ports.remove(moved)
        identified.append(_resolve_port_by_id(moved))
        if not ports:
            break
        cont = input("Continue identifying? (Y/n): ").strip().lower()
        if cont == "n":
            break

    print(f"\nDone. Identified {len(identified)} arm(s).")
    for port_id in identified:
        print(f"  - {port_id}")


if __name__ == "__main__":
    main()
