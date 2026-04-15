"""Hardware scanning — detect serial ports and cameras."""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path

from loguru import logger

from roboclaw.embodied.embodiment.interface.serial import SerialInterface
from roboclaw.embodied.embodiment.interface.video import VideoInterface


def _read_symlink_map(directory: str) -> dict[str, str]:
    """Read a directory of symlinks, return {resolved_target: symlink_path}."""
    d = Path(directory)
    if not d.exists():
        return {}
    result = {}
    for entry in d.iterdir():
        if entry.is_symlink():
            target = os.path.realpath(str(entry))
            result[target] = str(entry)
    return result


def serial_patterns_for_platform() -> tuple[str, ...]:
    """Return the correct serial device glob patterns for the current OS.

    On macOS, /dev/cu.* (user-space callout) and /dev/tty.* (BSD modem-control)
    point to the same physical USB serial adapter. Only cu.* is the correct
    endpoint for serial communication — tty.* blocks on DCD and causes duplicate
    detection during setup-identify. We scan cu.* exclusively.
    """
    if sys.platform == "darwin":
        return ("cu.usb*",)
    return ("ttyACM*", "ttyUSB*")


def _list_serial_ports(device_patterns: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """Return hardware serial ports, optionally filtered by patterns from a spec.

    Only scans actual USB/hardware serial devices — never virtual consoles
    (/dev/tty, /dev/tty0-63) or pseudo-terminals, since opening those with
    pyserial corrupts the controlling terminal's termios flags (OPOST, ECHO).
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        list_ports = None

    if os.name == "nt":
        if list_ports is None:
            return []
        return sorted(
            port.device
            for port in list_ports.comports()
            if getattr(port, "device", "")
        )

    if device_patterns:
        platform_key = "darwin" if sys.platform == "darwin" else "linux"
        patterns = device_patterns.get(platform_key, ())
    else:
        patterns = serial_patterns_for_platform()

    return sorted({str(p) for pat in patterns for p in Path("/dev").glob(pat)})


def scan_serial_ports(device_patterns: dict[str, tuple[str, ...]] | None = None) -> list[SerialInterface]:
    """Scan serial devices, return list of SerialInterface objects.

    Discovery scope is intentionally aligned with `lerobot-find-port`, while
    Linux symlink trees are still attached as stable `/dev/serial/by-*`
    aliases when available.
    """
    by_path = _read_symlink_map("/dev/serial/by-path")
    by_id = _read_symlink_map("/dev/serial/by-id")
    all_devs = set(_list_serial_ports(device_patterns)) | set(by_path.keys()) | set(by_id.keys())
    ports: list[SerialInterface] = []
    for dev in sorted(all_devs):
        if not os.path.exists(dev):
            continue
        ports.append(SerialInterface(
            by_path=by_path.get(dev, ""),
            by_id=by_id.get(dev, ""),
            dev=dev,
        ))
    return ports


def list_serial_device_paths() -> list[str]:
    """Return USB serial device paths using platform-appropriate patterns.

    Scoped to actual hardware serial ports only — NOT virtual consoles,
    pseudo-terminals, or other /dev/tty* entries. Used by permission
    checks and udev rule installation.
    """
    return sorted(
        path
        for pat in serial_patterns_for_platform()
        for path in glob.glob(f"/dev/{pat}")
    )


def list_video_device_paths() -> list[str]:
    """Return /dev/videoN paths present on the system."""
    return sorted(glob.glob("/dev/video*"))


def check_device_permissions() -> dict[str, dict[str, object]]:
    """Check R/W access for serial and camera devices.

    Returns ``{serial: {ok, count}, camera: {ok, count}, platform}``.
    On non-Linux platforms serial/camera are always reported as ok.
    """
    if sys.platform != "linux":
        return {
            "serial": {"ok": True, "count": 0},
            "camera": {"ok": True, "count": 0},
            "platform": sys.platform,
        }

    serial_devs = list_serial_device_paths()
    video_devs = list_video_device_paths()

    serial_ok = all(os.access(d, os.R_OK | os.W_OK) for d in serial_devs) if serial_devs else True
    camera_ok = all(os.access(d, os.R_OK | os.W_OK) for d in video_devs) if video_devs else True

    return {
        "serial": {"ok": serial_ok, "count": len(serial_devs)},
        "camera": {"ok": camera_ok, "count": len(video_devs)},
        "platform": "linux",
    }



def suppress_stderr() -> int:
    """Redirect stderr to /dev/null. Returns saved fd for restore_stderr."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    return saved


def restore_stderr(saved: int) -> None:
    """Restore stderr from saved fd."""
    os.dup2(saved, 2)
    os.close(saved)


def scan_cameras() -> list[VideoInterface]:
    """Scan cameras, return list of VideoInterface objects."""
    try:
        import cv2
    except ImportError:
        return []

    saved = suppress_stderr()
    try:
        by_path = _read_symlink_map("/dev/v4l/by-path")
        by_id = _read_symlink_map("/dev/v4l/by-id")
        return _probe_cameras(cv2, by_path, by_id)
    finally:
        restore_stderr(saved)


def capture_camera_frames(
    scanned_cameras: list[VideoInterface], output_dir: str | Path,
) -> list[dict[str, str]]:
    """Capture one JPEG preview for each scanned camera."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera previews.") from exc

    previews: list[dict[str, str]] = []
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    saved = suppress_stderr()
    try:
        for index, camera in enumerate(scanned_cameras):
            preview = _capture_camera_frame(cv2, camera, target_dir, index)
            if preview is not None:
                previews.append(preview)
        return previews
    finally:
        restore_stderr(saved)


def _probe_cameras(cv2, by_path: dict, by_id: dict) -> list[VideoInterface]:
    """Try opening each /dev/videoN, return one per physical USB device."""
    raw: list[VideoInterface] = []
    for dev in sorted(glob.glob("/dev/video*")):
        m = re.match(r"/dev/video(\d+)$", dev)
        if not m:
            continue
        info = _try_open_camera(cv2, int(m.group(1)), dev, by_path, by_id)
        if info:
            raw.append(info)
    return _dedupe_by_usb_device(raw)


def _usb_device_key(by_path_str: str) -> str:
    """Extract physical USB device from by-path.

    e.g. "pci-0000:00:14.0-usb-0:3:1.0-video-index0" → "usb-0:3"
    e.g. "pci-0000:00:14.0-usb-0:8.2:1.0-video-index0" → "usb-0:8.2"
    Different interfaces (1.0, 1.3) on the same port are the same device,
    but different hub sub-ports (8.1, 8.2, 8.3) are different devices.
    """
    m = re.search(r"(usb-\d+:\d+(?:\.\d+)*)", by_path_str)
    return m.group(1) if m else ""


def _interface_sort_key(cam: VideoInterface) -> tuple[tuple[int, int], str]:
    """Sort key: prefer higher interface number (RealSense RGB = 1.3), then lowest video index."""
    bp = cam.by_path
    # Extract interface e.g. "1.3" from "usb-0:2:1.3-video-index0"
    m = re.search(r"usb-\d+:\d+:(\d+)\.(\d+)", bp)
    iface = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return (iface, cam.dev)


def _dedupe_by_usb_device(cameras: list[VideoInterface]) -> list[VideoInterface]:
    """Keep one camera per physical USB device.

    For multi-stream devices (e.g. RealSense), prefer the highest interface
    number — on RealSense D435 interface 1.3 is RGB, 1.0 is depth/IR.
    """
    groups: dict[str, list[VideoInterface]] = {}
    ungrouped: list[VideoInterface] = []
    for cam in cameras:
        key = _usb_device_key(cam.by_path)
        if key:
            groups.setdefault(key, []).append(cam)
        else:
            ungrouped.append(cam)
    result = list(ungrouped)
    for cams in groups.values():
        # Prefer highest interface (RGB on RealSense), then lowest video index
        result.append(max(cams, key=_interface_sort_key))
    return sorted(result, key=lambda c: c.dev)


def _try_open_camera(cv2, index: int, dev: str, by_path: dict, by_id: dict) -> VideoInterface | None:
    """Open a single camera by index, return VideoInterface or None."""
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real = os.path.realpath(dev)
        fourcc = ""
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 30:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
            if cap.get(cv2.CAP_PROP_FPS) >= 30:
                fourcc = "MJPG"
                fps = 30
        return VideoInterface(
            by_path=by_path.get(real, ""),
            by_id=by_id.get(real, ""),
            dev=dev,
            width=w,
            height=h,
            fps=int(fps),
            fourcc=fourcc,
        )
    finally:
        cap.release()


def _capture_camera_frame(
    cv2, camera: VideoInterface, output_dir: Path, index: int,
) -> dict[str, str] | None:
    source = camera.address
    label = source
    if not source:
        return None

    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            return None
        # Skip initial frames — some cameras (e.g. RealSense) produce
        # garbage on the first few reads while the sensor initialises.
        for _ in range(30):
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        image_path = output_dir / f"{index:02d}_{Path(label).name}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Failed to write camera preview to {image_path}")
        return {
            "camera": label,
            "image_path": str(image_path),
        }
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Serial permission helpers
# ---------------------------------------------------------------------------


def fix_serial_permissions() -> bool:
    """Install udev rules for serial device access. Returns True on success."""
    udev_rule = (
        'KERNEL=="ttyACM[0-9]*", MODE="0666"\n'
        'KERNEL=="ttyUSB[0-9]*", MODE="0666"\n'
        'SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", MODE="0666"\n'
        'SUBSYSTEM=="video4linux", MODE="0666"\n'
    )
    try:
        result = subprocess.run(
            ["sudo", "-n", "tee", "/etc/udev/rules.d/99-roboclaw.rules"],
            input=udev_rule.encode(), capture_output=True, timeout=5,
        )
        if result.returncode != 0:
            logger.warning("Passwordless sudo not available for udev rules")
            return _try_chmod_devices()
        subprocess.run(
            ["sudo", "-n", "udevadm", "control", "--reload-rules"],
            capture_output=True, timeout=5,
        )
        subprocess.run(
            ["sudo", "-n", "udevadm", "trigger"],
            capture_output=True, timeout=5,
        )
        logger.info("Installed udev rules for serial device access")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return _try_chmod_devices()


def _try_chmod_devices() -> bool:
    """Fallback: chmod individual device files."""
    devices = list_serial_device_paths()
    if not devices:
        return False
    for dev in devices:
        try:
            os.chmod(dev, 0o666)
        except PermissionError:
            result = subprocess.run(
                ["sudo", "-n", "chmod", "666", dev],
                capture_output=True, timeout=5,
            )
            if result.returncode != 0:
                logger.warning("Cannot chmod {}: no passwordless sudo", dev)
                return False
    logger.info("Fixed serial device permissions via chmod")
    return True
