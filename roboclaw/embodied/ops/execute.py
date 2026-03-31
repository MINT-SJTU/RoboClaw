"""Asynchronous embodied action execution."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from roboclaw.embodied.hand_actions import (
    _do_hand_close,
    _do_hand_open,
    _do_hand_pose,
    _do_hand_status,
)
from roboclaw.embodied.ops.helpers import (
    _BIMANUAL_ID,
    _DEFAULT_REPLAY_ROOT,
    _NO_TTY_MSG,
    _arm_id,
    _bimanual_cal_dirs,
    _dataset_path,
    _format_tty_failure,
    _group_arms,
    _is_interrupted,
    _logs_dir,
    _resolve_action_arms,
    _run,
    _run_tty,
    _validate_dataset_name,
    _validate_pairing,
)
from roboclaw.embodied.sensor.camera import resolve_cameras

_BIMANUAL_FAMILIES: frozenset[str] = frozenset({"so101"})


def _arm_family(arms: list[dict]) -> str:
    """Extract arm family (e.g. 'so101', 'koch') and validate no mixing."""
    families = {a["type"].rsplit("_", 1)[0] for a in arms}
    if len(families) > 1:
        from roboclaw.embodied.ops.helpers import ActionError
        raise ActionError(f"Cannot mix arm families: {families}")
    return families.pop()


async def _do_doctor(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder
    from roboclaw.embodied.runner import LocalLeRobotRunner

    result = await _run(LocalLeRobotRunner(), ArmCommandBuilder().doctor())
    return result + f"\n\nCurrent setup:\n{json.dumps(setup, indent=2, ensure_ascii=False)}"


async def _do_identify(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.scan import scan_serial_ports

    if not tty_handoff:
        return _NO_TTY_MSG
    ports = scan_serial_ports()
    if not ports:
        return "No serial ports detected."
    argv = [sys.executable, "-m", "roboclaw.embodied.identify", json.dumps(ports)]
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "identify-arms")
    if rc == 0:
        return "Arm identification complete."
    return _format_tty_failure("Arm identification failed", rc, stderr_text)


def _sync_calibration_to_motors(arm: dict[str, Any]) -> None:
    """Write calibration values from file to motor EEPROM.

    Best-effort: failures are logged but never block the calibration result.
    """
    cal_dir = arm.get("calibration_dir", "")
    serial = Path(cal_dir).name
    cal_path = Path(cal_dir) / f"{serial}.json"
    if not cal_path.exists():
        return
    cal = json.loads(cal_path.read_text())

    if arm["type"].startswith("koch_"):
        _sync_calibration_dynamixel(arm, cal)
    else:
        _sync_calibration_feetech(arm, cal)


def _sync_calibration_feetech(arm: dict, cal: dict) -> None:
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
    except ImportError:
        return
    motors, calibration = {}, {}
    for name, cfg in cal.items():
        motors[name] = Motor(id=cfg["id"], model="sts3215", norm_mode=MotorNormMode.DEGREES)
        calibration[name] = MotorCalibration(
            id=cfg["id"], drive_mode=cfg["drive_mode"],
            homing_offset=cfg["homing_offset"],
            range_min=cfg["range_min"], range_max=cfg["range_max"],
        )
    bus = FeetechMotorsBus(port=arm["port"], motors=motors, calibration=calibration)
    try:
        bus.connect()
        for name, cfg in cal.items():
            bus.write("Homing_Offset", name, cfg["homing_offset"], normalize=False)
            bus.write("Min_Position_Limit", name, cfg["range_min"], normalize=False)
            bus.write("Max_Position_Limit", name, cfg["range_max"], normalize=False)
    except (OSError, ConnectionError):
        pass
    finally:
        bus.disconnect()


def _sync_calibration_dynamixel(arm: dict, cal: dict) -> None:
    try:
        from lerobot.motors.dynamixel import DynamixelMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode
    except ImportError:
        return
    is_leader = "leader" in arm["type"]
    model = "xl330-m077" if is_leader else None
    _FOLLOWER_MODELS = {
        "shoulder_pan": "xl430-w250", "shoulder_lift": "xl430-w250",
        "elbow_flex": "xl330-m288", "wrist_flex": "xl330-m288",
        "wrist_roll": "xl330-m288", "gripper": "xl330-m288",
    }
    motors, calibration = {}, {}
    for name, cfg in cal.items():
        m = model if is_leader else _FOLLOWER_MODELS.get(name, "xl330-m288")
        motors[name] = Motor(id=cfg["id"], model=m, norm_mode=MotorNormMode.DEGREES)
        calibration[name] = MotorCalibration(
            id=cfg["id"], drive_mode=cfg["drive_mode"],
            homing_offset=cfg["homing_offset"],
            range_min=cfg["range_min"], range_max=cfg["range_max"],
        )
    bus = DynamixelMotorsBus(port=arm["port"], motors=motors, calibration=calibration)
    try:
        bus.connect()
        for name, cfg in cal.items():
            bus.write("Homing_Offset", name, cfg["homing_offset"], normalize=False)
            bus.write("Min_Position_Limit", name, cfg["range_min"], normalize=False)
            bus.write("Max_Position_Limit", name, cfg["range_max"], normalize=False)
    except (OSError, ConnectionError):
        pass
    finally:
        bus.disconnect()


async def _do_calibrate(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.setup import arm_display_name, mark_arm_calibrated

    configured = setup.get("arms", [])
    if not configured:
        return "No arms configured."
    selected = _resolve_action_arms(setup, kwargs)
    targets = selected if kwargs.get("arms", "") else [a for a in selected if not a.get("calibrated")]
    if not targets:
        return "All arms are already calibrated."
    if not tty_handoff:
        return _NO_TTY_MSG
    controller = ArmCommandBuilder()
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


async def _do_teleoperate(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder

    if not tty_handoff:
        return _NO_TTY_MSG
    grouped = _group_arms(_resolve_action_arms(setup, kwargs))
    error = _validate_pairing(grouped["followers"], grouped["leaders"])
    if error:
        return error
    followers = grouped["followers"]
    leaders = grouped["leaders"]
    family = _arm_family(followers + leaders)
    controller = ArmCommandBuilder()
    cameras = resolve_cameras(setup)
    if len(followers) == 1:
        return await _teleoperate_single(controller, followers[0], leaders[0], cameras, tty_handoff)
    if family not in _BIMANUAL_FAMILIES:
        return f"{family} arms do not support bimanual mode."
    return await _teleoperate_bimanual(controller, followers, leaders, cameras, tty_handoff)


async def _teleoperate_single(
    controller: Any, follower: dict, leader: dict, cameras: dict, tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner
    from roboclaw.embodied.setup import arm_display_name

    argv = controller.teleoperate(
        robot_type=follower["type"],
        robot_port=follower["port"],
        robot_cal_dir=follower["calibration_dir"],
        robot_id=_arm_id(follower),
        teleop_type=leader["type"],
        teleop_port=leader["port"],
        teleop_cal_dir=leader["calibration_dir"],
        teleop_id=_arm_id(leader),
        cameras=cameras,
    )
    label = f"lerobot-teleoperate ({arm_display_name(follower)} + {arm_display_name(leader)})"
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, label)
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Teleoperation finished."
    return _format_tty_failure("Teleoperation failed", rc, stderr_text)


async def _teleoperate_bimanual(
    controller: Any,
    followers: list[dict],
    leaders: list[dict],
    cameras: dict,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    with _bimanual_cal_dirs(followers, leaders) as (robot_dir, teleop_dir):
        argv = controller.teleoperate_bimanual(
            robot_id=_BIMANUAL_ID,
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            teleop_id=_BIMANUAL_ID,
            teleop_cal_dir=teleop_dir,
            left_teleop=leaders[0],
            right_teleop=leaders[1],
            cameras=cameras,
        )
        rc, stderr_text = await _run_tty(
            tty_handoff, LocalLeRobotRunner(), argv, "lerobot-teleoperate (bimanual)",
        )
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Teleoperation finished."
    return _format_tty_failure("Teleoperation failed", rc, stderr_text)


def _resolve_dataset_name(
    kwargs: dict[str, Any], prefix: str,
) -> tuple[str, bool] | str:
    """Resolve dataset name and whether to resume.

    Returns (dataset_name, user_specified) or an error string.
    """
    user_specified = "dataset_name" in kwargs
    if user_specified:
        name = kwargs["dataset_name"]
    else:
        name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    error = _validate_dataset_name(name)
    if error:
        return error
    return name, user_specified


def _should_resume(
    user_specified: bool, dataset_root: Path,
) -> bool:
    """Resume only when user explicitly named a dataset that already exists."""
    return user_specified and dataset_root.exists()


async def _do_record(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    if kwargs.get("checkpoint_path"):
        return await _do_run_policy(setup, kwargs, tty_handoff)

    from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder

    if not tty_handoff:
        return _NO_TTY_MSG
    grouped = _group_arms(_resolve_action_arms(setup, kwargs))
    error = _validate_pairing(grouped["followers"], grouped["leaders"])
    if error:
        return error
    result = _resolve_dataset_name(kwargs, "rec")
    if isinstance(result, str):
        return result
    dataset_name, user_specified = result
    followers = grouped["followers"]
    leaders = grouped["leaders"]
    family = _arm_family(followers + leaders)
    controller = ArmCommandBuilder()
    cameras = {} if kwargs.get("use_cameras") is False else resolve_cameras(setup)
    record_kwargs = _build_record_kwargs(setup, kwargs, cameras, dataset_name)
    dataset_root = Path(record_kwargs["dataset_root"])
    if _should_resume(user_specified, dataset_root):
        record_kwargs["resume"] = True
    if len(followers) == 1:
        return await _record_single(controller, followers[0], leaders[0], record_kwargs, tty_handoff)
    if family not in _BIMANUAL_FAMILIES:
        return f"{family} arms do not support bimanual mode."
    return await _record_bimanual(controller, followers, leaders, record_kwargs, tty_handoff)


def _build_record_kwargs(
    setup: dict[str, Any], kwargs: dict[str, Any], cameras: dict, dataset_name: str,
) -> dict[str, Any]:
    result = {
        "cameras": cameras,
        "repo_id": f"local/{dataset_name}",
        "task": kwargs.get("task", "default_task"),
        "dataset_root": str(_dataset_path(setup, dataset_name)),
        "push_to_hub": False,
        "fps": kwargs.get("fps", 30),
        "num_episodes": kwargs.get("num_episodes", 10),
    }
    episode_time_s = kwargs.get("episode_time_s")
    if episode_time_s is not None:
        if episode_time_s <= 0:
            raise ValueError("episode_time_s must be positive.")
        result["episode_time_s"] = episode_time_s
    reset_time_s = kwargs.get("reset_time_s")
    if reset_time_s is not None:
        if reset_time_s < 0:
            raise ValueError("reset_time_s must be non-negative.")
        result["reset_time_s"] = reset_time_s
    return result


async def _record_single(
    controller: Any, follower: dict, leader: dict, record_kwargs: dict, tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    argv = controller.record(
        robot_type=follower["type"],
        robot_port=follower["port"],
        robot_cal_dir=follower["calibration_dir"],
        robot_id=_arm_id(follower),
        teleop_type=leader["type"],
        teleop_port=leader["port"],
        teleop_cal_dir=leader["calibration_dir"],
        teleop_id=_arm_id(leader),
        **record_kwargs,
    )
    rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "lerobot-record")
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Recording finished."
    return _format_tty_failure("Recording failed", rc, stderr_text)


async def _record_bimanual(
    controller: Any,
    followers: list[dict],
    leaders: list[dict],
    record_kwargs: dict,
    tty_handoff: Any,
) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    with _bimanual_cal_dirs(followers, leaders) as (robot_dir, teleop_dir):
        argv = controller.record_bimanual(
            robot_id=_BIMANUAL_ID,
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            teleop_id=_BIMANUAL_ID,
            teleop_cal_dir=teleop_dir,
            left_teleop=leaders[0],
            right_teleop=leaders[1],
            **record_kwargs,
        )
        rc, stderr_text = await _run_tty(tty_handoff, LocalLeRobotRunner(), argv, "lerobot-record")
    if _is_interrupted(rc):
        return "interrupted"
    if rc == 0:
        return "Recording finished."
    return _format_tty_failure("Recording failed", rc, stderr_text)


async def _do_run_policy(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    """Run a trained policy - called from record when checkpoint_path is set."""
    from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder
    from roboclaw.embodied.learning.act import ACTPipeline
    from roboclaw.embodied.runner import LocalLeRobotRunner

    grouped = _group_arms(_resolve_action_arms(setup, kwargs))
    followers = grouped["followers"]
    if not followers:
        return "No follower arm configured."
    if len(followers) not in {1, 2}:
        return f"Unsupported follower arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
    cameras = {} if kwargs.get("use_cameras") is False else resolve_cameras(setup)
    policies_root = setup.get("policies", {}).get("root", "")
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
    dataset_root = _dataset_path(setup, dataset_name)
    resume = _should_resume(user_specified, dataset_root)
    family = _arm_family(followers)
    controller = ArmCommandBuilder()
    policy_kwargs = {
        "cameras": cameras,
        "policy_path": checkpoint,
        "repo_id": f"local/{dataset_name}",
        "dataset_root": str(dataset_root),
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
    if family not in _BIMANUAL_FAMILIES:
        return f"{family} arms do not support bimanual mode."
    with _bimanual_cal_dirs(followers, []) as (robot_dir, _):
        argv = controller.run_policy_bimanual(
            robot_id=_BIMANUAL_ID,
            robot_cal_dir=robot_dir,
            left_robot=followers[0],
            right_robot=followers[1],
            **policy_kwargs,
        )
        return await _run(LocalLeRobotRunner(), argv)


async def _do_replay(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder

    if not tty_handoff:
        return _NO_TTY_MSG
    selected = _resolve_action_arms(setup, kwargs)
    grouped = _group_arms(selected)
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
    dataset_root = _dataset_path(setup, dataset_name, fallback=_DEFAULT_REPLAY_ROOT)
    episode = kwargs.get("episode", 0)
    fps = kwargs.get("fps", 30)
    family = _arm_family(followers)
    controller = ArmCommandBuilder()
    if len(followers) == 1:
        return await _replay_single(controller, followers[0], dataset_name, dataset_root, episode, fps, tty_handoff)
    if family not in _BIMANUAL_FAMILIES:
        return f"{family} arms do not support bimanual mode."
    return await _replay_bimanual(controller, followers, dataset_name, dataset_root, episode, fps, tty_handoff)


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
    from roboclaw.embodied.runner import LocalLeRobotRunner

    with _bimanual_cal_dirs(followers, []) as (robot_dir, _):
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


async def _do_train(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.learning.act import ACTPipeline
    from roboclaw.embodied.runner import LocalLeRobotRunner

    dataset_name = kwargs.get("dataset_name", "default")
    error = _validate_dataset_name(dataset_name)
    if error:
        return error
    dataset_root = _dataset_path(setup, dataset_name)
    policies_root = setup.get("policies", {}).get("root", "")
    output_dir = Path(policies_root).expanduser() / dataset_name
    resume = output_dir.is_dir()
    argv = ACTPipeline().train(
        repo_id=f"local/{dataset_name}",
        dataset_root=str(dataset_root),
        output_dir=str(output_dir),
        steps=kwargs.get("steps", 100_000),
        device=kwargs.get("device", "cuda"),
        resume=resume,
    )
    job_id = await LocalLeRobotRunner().run_detached(argv=argv, log_dir=_logs_dir())
    return f"Training started. Job ID: {job_id}"


async def _do_job_status(setup: dict[str, Any], kwargs: dict[str, Any], tty_handoff: Any) -> str:
    from roboclaw.embodied.runner import LocalLeRobotRunner

    job_id = kwargs.get("job_id", "")
    status = await LocalLeRobotRunner().job_status(job_id=job_id, log_dir=_logs_dir())
    return "\n".join(f"{key}: {value}" for key, value in status.items())


ASYNC_DISPATCH: dict[str, Any] = {
    "doctor": _do_doctor,
    "identify": _do_identify,
    "calibrate": _do_calibrate,
    "teleoperate": _do_teleoperate,
    "record": _do_record,
    "replay": _do_replay,
    "train": _do_train,
    "job_status": _do_job_status,
    "hand_open": _do_hand_open,
    "hand_close": _do_hand_close,
    "hand_pose": _do_hand_pose,
    "hand_status": _do_hand_status,
}
