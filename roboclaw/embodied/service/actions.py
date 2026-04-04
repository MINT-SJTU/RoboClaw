"""Stateless action dispatch functions for embodied operations.

Each function follows the signature (manifest, kwargs, tty_handoff) -> str
and is called by EmbodiedService methods.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from roboclaw.embodied.engine.command_builder import builder_for_arms
from roboclaw.embodied.engine.helpers import (
    _BIMANUAL_ID,
    _DEFAULT_REPLAY_ROOT,
    _NO_TTY_MSG,
    _arm_id,
    _format_tty_failure,
    _is_interrupted,
    _logs_dir,
    _resolve_action_arms,
    _run,
    _run_tty,
    _validate_dataset_name,
    dataset_path,
    group_arms,
)
from roboclaw.embodied.sensor.camera import resolve_cameras


async def do_doctor(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.engine.command_builder import ArmCommandBuilder
    from roboclaw.embodied.runner import LocalLeRobotRunner

    result = await _run(LocalLeRobotRunner(), ArmCommandBuilder().doctor())
    return result + f"\n\nCurrent setup:\n{json.dumps(manifest, indent=2, ensure_ascii=False)}"


async def do_identify(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.hardware.scan import scan_serial_ports

    if not tty_handoff:
        return _NO_TTY_MSG
    ports = scan_serial_ports()
    if not ports:
        return "No serial ports detected."
    ports_json = json.dumps([p.to_dict() for p in ports])
    argv = [sys.executable, "-m", "roboclaw.embodied.identify", ports_json]
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "identify-arms")
    if rc == 0:
        return "Arm identification complete."
    return _format_tty_failure("Arm identification failed", rc, stderr_text)


def _sync_calibration_to_motors(arm: dict[str, Any]) -> None:
    """Write calibration values from file to motor EEPROM.

    Best-effort: failures are logged but never block the calibration result.
    Uses the arm spec registry to pick the correct motor bus and models.
    """
    cal_dir = arm.get("calibration_dir", "")
    serial = Path(cal_dir).name
    cal_path = Path(cal_dir) / f"{serial}.json"
    if not cal_path.exists():
        return
    try:
        from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
    except ImportError:
        logger.debug("lerobot.motors not installed, skipping EEPROM sync")
        return

    from roboclaw.embodied.embodiment.arm.registry import get_arm_spec, get_role

    spec = get_arm_spec(arm["type"])
    role = get_role(arm["type"])
    model_map = spec.motor_models(role)
    cal = json.loads(cal_path.read_text())

    motors, calibration = {}, {}
    for name, cfg in cal.items():
        model = model_map.get(name, list(model_map.values())[0])
        motors[name] = Motor(id=cfg["id"], model=model, norm_mode=MotorNormMode.DEGREES)
        calibration[name] = MotorCalibration(
            id=cfg["id"], drive_mode=cfg["drive_mode"],
            homing_offset=cfg["homing_offset"],
            range_min=cfg["range_min"], range_max=cfg["range_max"],
        )

    import importlib
    mod = importlib.import_module(spec.motor_bus_module)
    BusClass = getattr(mod, spec.motor_bus_class)

    bus = BusClass(port=arm["port"], motors=motors, calibration=calibration)
    try:
        bus.connect()
        for name, cfg in cal.items():
            bus.write("Homing_Offset", name, cfg["homing_offset"], normalize=False)
            bus.write("Min_Position_Limit", name, cfg["range_min"], normalize=False)
            bus.write("Max_Position_Limit", name, cfg["range_max"], normalize=False)
    except (OSError, ConnectionError):
        logger.debug("Motor EEPROM sync failed for %s", arm.get("alias", "?"))
    finally:
        bus.disconnect()


async def do_calibrate(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.manifest.helpers import arm_display_name
    from roboclaw.embodied.manifest.helpers import mark_arm_calibrated
    from roboclaw.embodied.runner import LocalLeRobotRunner

    configured = manifest.get("arms", [])
    if not configured:
        return "No arms configured."
    selected = _resolve_action_arms(manifest, kwargs)
    targets = selected if kwargs.get("arms", "") else [a for a in selected if not a.get("calibrated")]
    if not targets:
        return "All arms are already calibrated."
    if not tty_handoff:
        return _NO_TTY_MSG
    controller = builder_for_arms(targets)
    runner = LocalLeRobotRunner()
    succeeded = 0
    failed = 0
    results: list[str] = []
    for arm in targets:
        display = arm_display_name(arm)
        argv = controller.calibrate(
            arm["type"], arm["port"], arm.get("calibration_dir", ""), _arm_id(arm),
        )
        rc, stderr_text = await _run_tty(tty_handoff, runner, argv, f"Calibrating: {display}")
        if _is_interrupted(rc):
            return "interrupted"
        if rc == 0:
            succeeded += 1
            mark_arm_calibrated(arm["alias"])
            _sync_calibration_to_motors(arm)
            results.append(f"{display}: OK")
            continue
        failed += 1
        results.append(_format_tty_failure(f"{display}: FAILED", rc, stderr_text))
    return (
        f"{succeeded} succeeded, {failed} failed.\n"
        + "\n".join(results)
        + "\nNote: wrist_roll is auto-calibrated by LeRobot (expected)."
    )


def _resolve_dataset_name(
    kwargs: dict[str, Any], prefix: str,
) -> tuple[str, bool] | str:
    from datetime import datetime

    user_specified = "dataset_name" in kwargs
    name = kwargs["dataset_name"] if user_specified else f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    error = _validate_dataset_name(name)
    if error:
        return error
    return name, user_specified


def _should_resume(user_specified: bool, dataset_root: Path) -> bool:
    return user_specified and dataset_root.exists()


async def do_run_policy(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.learning.act import ACTPipeline
    from roboclaw.embodied.runner import LocalLeRobotRunner

    grouped = group_arms(_resolve_action_arms(manifest, kwargs))
    followers = grouped["followers"]
    if not followers:
        return "No follower arm configured."
    if len(followers) not in {1, 2}:
        return f"Unsupported follower arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
    cameras = {} if kwargs.get("use_cameras") is False else resolve_cameras(manifest)
    policies_root = manifest.get("policies", {}).get("root", "")
    checkpoint = kwargs.get("checkpoint_path")
    if not checkpoint:
        source_dataset = kwargs.get("source_dataset", kwargs.get("dataset_name", ""))
        if source_dataset:
            checkpoint = ACTPipeline().checkpoint_path(str(Path(policies_root) / source_dataset))
        else:
            checkpoint = ACTPipeline().checkpoint_path(policies_root)
    result = _resolve_dataset_name(kwargs, "eval")
    if isinstance(result, str):
        return result
    dataset_name, user_specified = result
    if user_specified and not dataset_name.startswith("eval_"):
        dataset_name = f"eval_{dataset_name}"
    ds_root = dataset_path(manifest, dataset_name)
    resume = _should_resume(user_specified, ds_root)
    controller = builder_for_arms(followers)
    policy_kwargs = {
        "cameras": cameras,
        "policy_path": checkpoint,
        "repo_id": f"local/{dataset_name}",
        "dataset_root": str(ds_root),
        "task": kwargs.get("task", "eval"),
        "num_episodes": kwargs.get("num_episodes", 1),
        "resume": resume,
    }
    if len(followers) == 1:
        follower = followers[0]
        argv = controller.run_policy(
            robot_type=follower["type"],
            robot_port=follower["port"],
            robot_cal_dir=follower["calibration_dir"],
            robot_id=_arm_id(follower),
            **policy_kwargs,
        )
        return await _run(LocalLeRobotRunner(), argv)
    from roboclaw.embodied.manifest.helpers import ensure_bimanual_cal_dir
    robot_dir = ensure_bimanual_cal_dir(followers[0], followers[1], "followers")
    argv = controller.run_policy_bimanual(
        robot_id=_BIMANUAL_ID,
        robot_cal_dir=robot_dir,
        left_robot=followers[0],
        right_robot=followers[1],
        **policy_kwargs,
    )
    return await _run(LocalLeRobotRunner(), argv)


async def do_replay(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    if not tty_handoff:
        return _NO_TTY_MSG
    selected = _resolve_action_arms(manifest, kwargs)
    grouped = group_arms(selected)
    if kwargs.get("arms", "") and grouped["leaders"]:
        return "Replay only supports follower arms. Remove leader arm ports from arms."
    followers = grouped["followers"]
    if not followers:
        return "No follower arm configured."
    if len(followers) not in {1, 2}:
        return f"Unsupported follower arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
    dataset_name = kwargs.get("dataset_name", "default")
    error = _validate_dataset_name(dataset_name)
    if error:
        return error
    ds_root = dataset_path(manifest, dataset_name, fallback=_DEFAULT_REPLAY_ROOT)
    episode = kwargs.get("episode", 0)
    fps = kwargs.get("fps", 30)
    controller = builder_for_arms(followers)
    if len(followers) == 1:
        return await _replay_single(controller, followers[0], dataset_name, ds_root, episode, fps, tty_handoff)
    return await _replay_bimanual(controller, followers, dataset_name, ds_root, episode, fps, tty_handoff)


async def _replay_single(
    controller: Any, follower: dict,
    dataset_name: str, dataset_root: Path, episode: int, fps: int,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    argv = controller.replay(
        robot_type=follower["type"],
        robot_port=follower["port"],
        robot_cal_dir=follower["calibration_dir"],
        robot_id=_arm_id(follower),
        repo_id=f"local/{dataset_name}",
        dataset_root=str(dataset_root),
        episode=episode,
        fps=fps,
    )
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "lerobot-replay")
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Replay finished."
    return _format_tty_failure("Replay failed", rc, stderr_text)


async def _replay_bimanual(
    controller: Any, followers: list[dict],
    dataset_name: str, dataset_root: Path, episode: int, fps: int,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.manifest.helpers import ensure_bimanual_cal_dir
    from roboclaw.embodied.runner import LocalLeRobotRunner

    robot_dir = ensure_bimanual_cal_dir(followers[0], followers[1], "followers")
    argv = controller.replay_bimanual(
        robot_id=_BIMANUAL_ID,
        robot_cal_dir=robot_dir,
        left_robot=followers[0],
        right_robot=followers[1],
        repo_id=f"local/{dataset_name}",
        dataset_root=str(dataset_root),
        episode=episode,
        fps=fps,
    )
    rc, stderr_text = await _run_tty(
        tty_handoff, LocalLeRobotRunner(), argv, "lerobot-replay (bimanual)",
    )
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Replay finished."
    return _format_tty_failure("Replay failed", rc, stderr_text)


async def do_train(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.learning.act import ACTPipeline
    from roboclaw.embodied.runner import LocalLeRobotRunner

    dataset_name = kwargs.get("dataset_name", "default")
    error = _validate_dataset_name(dataset_name)
    if error:
        return error
    ds_root = dataset_path(manifest, dataset_name)
    policies_root = manifest.get("policies", {}).get("root", "")
    output_dir = Path(policies_root).expanduser() / dataset_name
    resume = output_dir.is_dir()
    argv = ACTPipeline().train(
        repo_id=f"local/{dataset_name}",
        dataset_root=str(ds_root),
        output_dir=str(output_dir),
        steps=kwargs.get("steps", 100_000),
        device=kwargs.get("device", "cuda"),
        resume=resume,
    )
    job_id = await LocalLeRobotRunner().run_detached(argv=argv, log_dir=_logs_dir())
    return f"Training started. Job ID: {job_id}"


async def do_job_status(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    job_id = kwargs.get("job_id", "")
    status = await LocalLeRobotRunner().job_status(job_id=job_id, log_dir=_logs_dir())
    return "\n".join(f"{key}: {value}" for key, value in status.items())


def _resolve_hand(manifest: dict[str, Any], hand_name: str) -> dict:
    from roboclaw.embodied.engine.helpers import ActionError
    from roboclaw.embodied.manifest.helpers import find_hand

    hands = manifest.get("hands", [])
    if not hands:
        raise ActionError("No hand configured. Use set_hand to add one.")
    if not hand_name:
        return hands[0]
    hand = find_hand(hands, hand_name)
    if hand is None:
        raise ActionError(f"No hand named '{hand_name}' in manifest.")
    return hand


def _get_hand_controller(hand_type: str):
    import importlib

    from roboclaw.embodied.embodiment.hand.registry import get_hand_spec

    spec = get_hand_spec(hand_type)
    mod = importlib.import_module(spec.controller_module)
    return getattr(mod, spec.controller_class)()


async def _run_hand_method(method_name: str, manifest: dict[str, Any], kwargs: dict[str, Any], extra_args=()):
    hand = _resolve_hand(manifest, kwargs.get("hand_name", ""))
    slave_id = hand["slave_id"]
    controller = _get_hand_controller(hand["type"])
    method = getattr(controller, method_name)
    if asyncio.iscoroutinefunction(method):
        return await method(hand["port"], *extra_args, slave_id)
    return await asyncio.to_thread(method, hand["port"], *extra_args, slave_id)


async def do_hand_open(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    return await _run_hand_method("open_hand", manifest, kwargs)


async def do_hand_close(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    return await _run_hand_method("close_hand", manifest, kwargs)


async def do_hand_pose(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    positions = kwargs.get("positions")
    if not positions:
        return "hand_pose requires positions (6 integers 0-1000)."
    return await _run_hand_method("set_pose", manifest, kwargs, extra_args=(positions,))


async def do_hand_status(manifest: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    return await _run_hand_method("get_status", manifest, kwargs)
