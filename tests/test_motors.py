from __future__ import annotations

from types import SimpleNamespace

import pytest

from roboclaw.embodied.embodiment.arm.registry import get_runtime_spec
from roboclaw.embodied.embodiment.hardware import motors
from roboclaw.embodied.embodiment.manifest.binding import load_binding


class FakeMotor:
    def __init__(self, id: int, model: str, norm_mode: str) -> None:
        self.id = id
        self.model = model
        self.norm_mode = norm_mode


class _BaseFakeBus:
    instances: list["_BaseFakeBus"] = []

    def __init__(self, port: str, motors: dict[str, FakeMotor]) -> None:
        self.port = port
        self.motors = motors
        self.connected = False
        self.disconnected = False
        type(self).instances.append(self)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.disconnected = True

    def read(self, register: str, name: str, normalize: bool = False) -> int:
        if register == "Present_Position":
            return self.motors[name].id * 100
        if register == "Present_Temperature":
            return self.motors[name].id + 20
        raise ValueError(register)


class FakeFeetechBus(_BaseFakeBus):
    instances: list["FakeFeetechBus"] = []


class FakeDynamixelBus(_BaseFakeBus):
    instances: list["FakeDynamixelBus"] = []


def _arm(alias: str, arm_type: str, port: str = "/dev/ttyUSB0"):
    return load_binding(
        {
            "alias": alias,
            "type": arm_type,
            "port": port,
            "calibration_dir": f"/tmp/{alias}",
            "calibrated": False,
        },
        "arm",
        {},
    )


@pytest.fixture(autouse=True)
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeFeetechBus.instances.clear()
    FakeDynamixelBus.instances.clear()

    def fake_import_module(name: str):
        if name == "lerobot.motors.motors_bus":
            return SimpleNamespace(
                Motor=FakeMotor,
                MotorNormMode=SimpleNamespace(RANGE_M100_100="range"),
            )
        if name == "lerobot.motors.feetech":
            return SimpleNamespace(FeetechMotorsBus=FakeFeetechBus)
        if name == "lerobot.motors.dynamixel":
            return SimpleNamespace(DynamixelMotorsBus=FakeDynamixelBus)
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(motors.importlib, "import_module", fake_import_module)


def test_read_servo_positions_dispatches_per_arm_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(motors, "load_calibration", lambda arm: {})

    result = motors.read_servo_positions([
        _arm("left_so101", "so101_follower", "/dev/ttyUSB0"),
        _arm("right_koch", "koch_leader", "/dev/ttyUSB1"),
    ])

    assert result["error"] is None
    assert len(FakeFeetechBus.instances) == 1
    assert len(FakeDynamixelBus.instances) == 1
    assert set(FakeFeetechBus.instances[0].motors) == set(get_runtime_spec("so101").default_joint_names)
    assert set(FakeDynamixelBus.instances[0].motors) == set(get_runtime_spec("koch").default_joint_names)


def test_read_servo_positions_prefers_calibration_names_and_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        motors,
        "load_calibration",
        lambda arm: {
            "joint_a": {"id": 7},
            "joint_b": {"id": 9},
        },
    )

    result = motors.read_servo_positions([
        _arm("calibrated_koch", "koch_follower"),
    ])

    bus = FakeDynamixelBus.instances[0]
    assert list(bus.motors) == ["joint_a", "joint_b"]
    assert bus.motors["joint_a"].id == 7
    assert bus.motors["joint_b"].model == get_runtime_spec("koch").default_motor
    assert result["arms"]["calibrated_koch"]["positions"] == {"joint_a": 700, "joint_b": 900}


def test_read_servo_positions_uses_model_defaults_without_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(motors, "load_calibration", lambda arm: {})

    result = motors.read_servo_positions([
        _arm("default_so101", "so101_leader"),
    ])

    spec = get_runtime_spec("so101")
    bus = FakeFeetechBus.instances[0]
    assert list(bus.motors) == list(spec.default_joint_names)
    assert bus.motors["shoulder_pan"].id == 1
    assert bus.motors["gripper"].model == spec.default_motor
    assert result["arms"]["default_so101"]["temperatures"]["gripper"] == 26


def test_read_servo_positions_fails_fast_on_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown arm model: mystery"):
        motors.read_servo_positions([
            _arm("unknown", "mystery_follower"),
        ])
