from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ONBOARDING_DIR = _ROOT / "roboclaw" / "embodied" / "onboarding"


def _load_onboarding_module(module_name: str, file_name: str):
    package_name = "roboclaw.embodied.onboarding"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(_ONBOARDING_DIR)]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, _ONBOARDING_DIR / file_name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_model = _load_onboarding_module("roboclaw.embodied.onboarding.model", "model.py")
_helpers = _load_onboarding_module("roboclaw.embodied.onboarding.helpers", "helpers.py")

SetupOnboardingState = _model.SetupOnboardingState
canonical_ids = _helpers.canonical_ids
component_summary = _helpers.component_summary
mount_frame = _helpers.mount_frame
normalize_serial_device_by_id = _helpers.normalize_serial_device_by_id
select_serial_device_by_id = _helpers.select_serial_device_by_id


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("", None),
        ("random\n/dev/ttyUSB0\n", None),
        ("/dev/serial/by-id/usb-robot\n", "/dev/serial/by-id/usb-robot"),
        (
            "  /dev/serial/by-id/usb-robot -> ../../ttyUSB0\n",
            "/dev/serial/by-id/usb-robot",
        ),
        (
            "noise\n/dev/serial/by-id/first\n/dev/serial/by-id/second\n",
            "/dev/serial/by-id/first",
        ),
    ],
)
def test_select_serial_device_by_id(output: str, expected: str | None) -> None:
    assert select_serial_device_by_id(output) == expected


def test_normalize_serial_device_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _helpers,
        "resolve_serial_by_id_path",
        lambda device_path: Path("/dev/serial/by-id/usb-robot") if device_path.strip() == "/dev/ttyUSB0" else None,
    )

    assert normalize_serial_device_by_id(" /dev/ttyUSB0 ") == "/dev/serial/by-id/usb-robot"
    assert normalize_serial_device_by_id(" ") is None
    assert normalize_serial_device_by_id("/dev/ttyACM0") is None


@pytest.mark.parametrize(
    ("current_setup_id", "robots", "expected"),
    [
        (
            "embodied_setup",
            [{"robot_id": "so100"}],
            ("so100_setup", "so100_setup", "so100_setup", "so100_setup_real_local", "so100_setup_ros2_local"),
        ),
        (
            "custom_setup",
            [{"robot_id": "so100"}],
            ("custom_setup", "custom_setup", "custom_setup", "custom_setup_real_local", "custom_setup_ros2_local"),
        ),
        (
            "embodied_setup",
            [],
            ("embodied_setup", "embodied_setup", "embodied_setup", "embodied_setup_real_local", "embodied_setup_ros2_local"),
        ),
    ],
)
def test_canonical_ids(
    current_setup_id: str,
    robots: list[dict[str, str]],
    expected: tuple[str, str, str, str, str],
) -> None:
    assert canonical_ids(current_setup_id, robots) == expected


def test_component_summary() -> None:
    state = SetupOnboardingState(
        setup_id="setup",
        intake_slug="setup",
        assembly_id="setup",
        deployment_id="setup_real_local",
        adapter_id="setup_ros2_local",
        robot_attachments=[
            {"attachment_id": "arm", "robot_id": "so100", "role": "primary"},
            {"attachment_id": "hand", "robot_id": "dexhand", "role": "tool"},
        ],
        sensor_attachments=[
            {"attachment_id": "wrist_camera", "sensor_id": "rgb_camera", "mount": "wrist"},
            {"attachment_id": "overhead_camera", "sensor_id": "rgb_camera", "mount": "overhead"},
        ],
    )

    assert component_summary(state) == "robots=[so100, dexhand] sensors=[rgb_camera@wrist, rgb_camera@overhead]"


@pytest.mark.parametrize(
    ("mount", "expected"),
    [
        ("wrist", "tool0"),
        ("overhead", "world"),
        ("external", "world"),
        ("custom", "world"),
    ],
)
def test_mount_frame(mount: str, expected: str) -> None:
    assert mount_frame(mount) == expected
