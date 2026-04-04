"""Hardware discovery orchestrator."""
from __future__ import annotations

from dataclasses import replace

from roboclaw.embodied.hardware.motion import (
    MOTION_THRESHOLD,
    detect_motion,
    read_positions_for_port,
    resolve_port_by_id,
    resolve_port_path,
)
from roboclaw.embodied.hardware.probers import _REGISTRY, get_prober
from roboclaw.embodied.hardware.scan import (
    capture_camera_frames,
    scan_cameras,
    scan_serial_ports,
    suppress_stderr,
    restore_stderr,
)
from roboclaw.embodied.interface.serial import SerialInterface
from roboclaw.embodied.interface.video import VideoInterface


class HardwareDiscovery:
    """Stateful hardware discovery workflow."""

    def __init__(self) -> None:
        self._scanned_ports: list[SerialInterface] = []
        self._scanned_cameras: list[VideoInterface] = []
        self._baselines: dict[str, dict[int, int]] = {}
        self._motion_active: bool = False

    @property
    def scanned_ports(self) -> list[SerialInterface]:
        return self._scanned_ports

    @property
    def scanned_cameras(self) -> list[VideoInterface]:
        return self._scanned_cameras

    @property
    def motion_active(self) -> bool:
        return self._motion_active

    def discover(self, model: str) -> list[SerialInterface]:
        """Probe ports using the protocol for the given model."""
        from roboclaw.embodied.embodiment.arm.registry import get_arm_spec_by_name

        spec = get_arm_spec_by_name(model)
        prober = get_prober(spec.probe_protocol)
        ports = scan_serial_ports()
        result = self._probe_ports(ports, prober, spec.probe_protocol)
        self._scanned_ports = result
        return result

    def discover_all(self) -> list[SerialInterface]:
        """Probe all ports with all registered probers (per-port, per-prober).

        Each port is tried with each prober independently, fixing the
        mixed-protocol bug where Koch arms were lost if SO101 was found first.
        """
        ports = scan_serial_ports()
        result: list[SerialInterface] = []
        for protocol, prober_cls in _REGISTRY.items():
            prober = prober_cls()
            matched = self._probe_ports(ports, prober, protocol)
            matched_devs = {p.dev for p in matched}
            result.extend(matched)
            ports = [p for p in ports if p.dev not in matched_devs]
            if not ports:
                break
        self._scanned_ports = result
        return result

    def discover_cameras(self) -> list[VideoInterface]:
        """Scan connected cameras."""
        result = scan_cameras()
        self._scanned_cameras = result
        return result

    def capture_camera_previews(self, output_dir: str) -> list[dict]:
        """Capture one preview frame per camera."""
        if not self._scanned_cameras:
            raise RuntimeError("No cameras scanned. Run discover_cameras first.")
        return capture_camera_frames(self._scanned_cameras, output_dir)

    def start_motion_detection(self) -> int:
        """Read baselines for all scanned ports. Returns port count."""
        if not self._scanned_ports:
            raise RuntimeError("No scanned ports. Run discover first.")
        self._baselines = {}
        saved = suppress_stderr()
        try:
            for port in self._scanned_ports:
                path = resolve_port_path(port)
                self._baselines[path] = read_positions_for_port(port)
        finally:
            restore_stderr(saved)
        self._motion_active = True
        return len(self._baselines)

    def poll_motion(self) -> list[dict]:
        """Read current positions and compute motion delta for each port."""
        if not self._motion_active:
            raise RuntimeError("Motion detection not started.")
        results = []
        saved = suppress_stderr()
        try:
            for port in self._scanned_ports:
                path = resolve_port_path(port)
                current = read_positions_for_port(port)
                baseline = self._baselines.get(path, {})
                delta = detect_motion(baseline, current)
                results.append({
                    "port_id": resolve_port_by_id(port),
                    "dev": port.dev,
                    "by_id": port.by_id,
                    "motor_ids": list(port.motor_ids),
                    "delta": delta,
                    "moved": delta > MOTION_THRESHOLD,
                })
        finally:
            restore_stderr(saved)
        return results

    def stop_motion_detection(self) -> None:
        """Clear baselines and stop motion detection."""
        self._motion_active = False
        self._baselines = {}

    def _probe_ports(
        self, ports: list[SerialInterface], prober, protocol: str = "",
    ) -> list[SerialInterface]:
        """Probe ports with a single prober, handling permission errors."""
        from roboclaw.embodied.hardware.scan import fix_serial_permissions

        saved = suppress_stderr()
        try:
            try:
                return self._do_probe(ports, prober, protocol)
            except Exception as exc:
                if "Permission denied" not in str(exc) and "Errno 13" not in str(exc):
                    raise
                if fix_serial_permissions():
                    return self._do_probe(ports, prober, protocol)
                raise PermissionError(
                    "Serial port permission denied. Run: bash scripts/setup-udev.sh"
                ) from exc
        finally:
            restore_stderr(saved)

    @staticmethod
    def _do_probe(
        ports: list[SerialInterface], prober, protocol: str = "",
    ) -> list[SerialInterface]:
        """Run the prober on each port, return those with motors."""
        result: list[SerialInterface] = []
        for port in ports:
            path = resolve_port_path(port)
            if not path:
                continue
            ids = prober.probe(path)
            if ids:
                result.append(replace(port, motor_ids=tuple(ids), bus_type=protocol))
        return result
