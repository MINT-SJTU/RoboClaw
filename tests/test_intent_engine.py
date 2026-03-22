from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from roboclaw.embodied.intent import IntentClassifier, UserIntent

import roboclaw.embodied


def _load_onboarding_submodule(name: str):
    repo_root = Path(__file__).resolve().parents[1]
    onboarding_dir = repo_root / "roboclaw" / "embodied" / "onboarding"
    package_name = "roboclaw.embodied.onboarding"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(onboarding_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        setattr(roboclaw.embodied, "onboarding", package)

    module_name = f"{package_name}.{name}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(module_name, onboarding_dir / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


OnboardingIntent = _load_onboarding_submodule("model").OnboardingIntent
_load_onboarding_submodule("ros2_install")
IntentEngine = _load_onboarding_submodule("intent_engine").IntentEngine


def _engine(llm_caller=None) -> IntentEngine:
    robot_aliases = {
        "so101": ("so101", "so-101", "leader arm"),
        "demo_bot": ("demo bot", "demo arm"),
        "piperx": ("piperx", "piper x"),
    }
    classifier = IntentClassifier(
        llm_caller=llm_caller,
        known_robots=tuple(robot_aliases),
        robot_aliases=robot_aliases,
    )
    return IntentEngine(classifier, robot_aliases)


def test_looks_like_setup_start_with_various_inputs() -> None:
    engine = _engine()

    assert engine.looks_like_setup_start("connect my real robot") is True
    assert engine.looks_like_setup_start("我想连接真实机器人") is True
    assert engine.looks_like_setup_start("I have no robot, I want to try simulation") is True
    assert engine.looks_like_setup_start("just saying hello") is False


def test_looks_like_sim_request_in_english_and_chinese() -> None:
    engine = _engine()

    assert engine.looks_like_sim_request("I have no robot, I want to try simulation") is True
    assert engine.looks_like_sim_request("我没有机器人，想试试仿真") is True
    assert engine.looks_like_sim_request("please connect my robot") is False


def test_extract_robot_ids_with_aliases() -> None:
    engine = _engine()

    assert engine.extract_robot_ids("please connect demo arm and piper x") == ["demo_bot", "piperx"]


def test_extract_connected_state() -> None:
    engine = _engine()

    assert engine.extract_connected_state("Everything is connected") is True
    assert engine.extract_connected_state("机器人还没连") is False
    assert engine.extract_connected_state("what can you do") is None


def test_heuristic_intent() -> None:
    engine = _engine()

    intent = engine.heuristic_intent("我没有机器人，想试试仿真")

    assert intent == OnboardingIntent(
        robot_ids=(),
        simulation_requested=True,
        sensor_changes=(),
        connected=None,
        serial_path=None,
        ros2_install_profile=None,
        ros2_state=None,
        ros2_install_requested=False,
        ros2_step_advance=False,
        calibration_requested=False,
        preferred_language="zh",
    )


def test_merge_intents() -> None:
    primary = OnboardingIntent(
        robot_ids=("so101",),
        simulation_requested=False,
        connected=True,
        serial_path="/dev/ttyACM0",
        calibration_requested=False,
    )
    secondary = OnboardingIntent(
        simulation_requested=True,
        connected=None,
        calibration_requested=True,
        preferred_language="zh",
    )

    merged = IntentEngine.merge_intents(primary, secondary)

    assert merged.robot_ids == ("so101",)
    assert merged.simulation_requested is True
    assert merged.connected is True
    assert merged.serial_path == "/dev/ttyACM0"
    assert merged.calibration_requested is True
    assert merged.preferred_language == "zh"


def test_heuristic_intent_uses_classified_fields_when_llm_is_present() -> None:
    async def llm(system_prompt: str, user_message: str) -> str:
        return '{"wants_setup": true, "robot_id": "so101", "connection_confirmed": true, "wants_calibration": true, "serial_path": "/dev/serial/by-id/demo", "is_embodied": true}'

    engine = _engine(llm_caller=llm)

    intent = engine.heuristic_intent(
        "please connect my robot",
        user_intent=UserIntent(
            wants_setup=True,
            robot_id="so101",
            connection_confirmed=True,
            wants_calibration=True,
            serial_path="/dev/serial/by-id/demo",
            is_embodied=True,
        ),
    )

    assert intent.robot_ids == ("so101",)
    assert intent.connected is True
    assert intent.serial_path == "/dev/serial/by-id/demo"
    assert intent.calibration_requested is True
