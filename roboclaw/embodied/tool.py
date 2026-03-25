"""Embodied tool — bridges agent to the embodied robotics layer."""

import json
import sys
from pathlib import Path
from typing import Any

from roboclaw.agent.tools.base import Tool

_ACTIONS = [
    "doctor",
    "identify",
    "calibrate",
    "teleoperate",
    "record",
    "train",
    "run_policy",
    "job_status",
    "setup_show",
    "set_arm",
    "remove_arm",
    "set_camera",
    "remove_camera",
]

_LOGS_DIR = Path("~/.roboclaw/workspace/embodied/jobs").expanduser()
_NO_TTY_MSG = "This action requires a local terminal. Run: roboclaw agent"


class EmbodiedTool(Tool):
    """Control embodied robots via the agent.

    The agent maintains setup.json through structured actions
    (set_arm, remove_arm, set_camera, remove_camera, setup_show).
    All hardware actions read setup.json for arm ports, cameras, calibration dirs.
    """

    def __init__(self, tty_handoff: Any = None):
        self._tty_handoff = tty_handoff

    @property
    def name(self) -> str:
        return "embodied"

    @property
    def description(self) -> str:
        return (
            "ALWAYS use this tool for robot/arm/hardware questions. "
            "NEVER use exec for /dev queries. "
            "Actions: setup_show (view config), identify (detect arms), "
            "calibrate, teleoperate, record, train, run_policy, job_status, "
            "set_arm, remove_arm, set_camera, remove_camera. "
            "Use arm aliases from setup.json for follower_names/leader_names."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": _ACTIONS,
                    "description": "The action to perform.",
                },
                "dataset_name": {
                    "type": "string",
                    "description": "Name for the dataset.",
                },
                "task": {
                    "type": "string",
                    "description": "Task description for recording.",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to record or run.",
                },
                "fps": {
                    "type": "integer",
                    "description": "Frames per second for recording.",
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of training steps.",
                },
                "checkpoint_path": {
                    "type": "string",
                    "description": "Path to a trained policy checkpoint.",
                },
                "job_id": {
                    "type": "string",
                    "description": "ID of a background training job.",
                },
                "device": {
                    "type": "string",
                    "description": "Device for training (default: cuda).",
                },
                "name": {
                    "type": "string",
                    "description": "Arm alias (for set_arm / remove_arm).",
                },
                "arm_type": {
                    "type": "string",
                    "enum": ["so101_follower", "so101_leader"],
                    "description": "Arm hardware type (for set_arm).",
                },
                "port": {
                    "type": "string",
                    "description": "Serial port path (for set_arm).",
                },
                "camera_name": {
                    "type": "string",
                    "description": "Camera name like front/side (for set_camera / remove_camera).",
                },
                "camera_index": {
                    "type": "integer",
                    "description": "Index into scanned_cameras (for set_camera).",
                },
                "follower_names": {
                    "type": "string",
                    "description": "Comma-separated follower arm aliases (for teleoperate/record).",
                },
                "leader_names": {
                    "type": "string",
                    "description": "Comma-separated leader arm aliases (for teleoperate/record).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        from roboclaw.embodied.setup import ensure_setup, load_setup

        action = kwargs.get("action", "")

        if action == "setup_show":
            return json.dumps(load_setup(), indent=2, ensure_ascii=False)

        if action in ("set_arm", "remove_arm", "set_camera", "remove_camera"):
            return self._handle_setup_action(action, kwargs)

        setup = ensure_setup()

        if action == "doctor":
            return await self._do_doctor(setup)
        if action == "identify":
            return await self._do_identify(setup)
        if action == "calibrate":
            return await self._do_calibrate(setup)
        if action == "teleoperate":
            return await self._do_teleoperate(setup, kwargs)
        if action == "record":
            return await self._do_record(setup, kwargs)
        if action == "train":
            return await self._do_train(setup, kwargs)
        if action == "run_policy":
            return await self._do_run_policy(setup, kwargs)
        if action == "job_status":
            return await self._do_job_status(kwargs)

        return f"Unknown action: {action}"

    def _handle_setup_action(self, action: str, kwargs: dict) -> str:
        """Dispatch structured setup mutations. Returns user-facing message."""
        from roboclaw.embodied.setup import remove_arm, remove_camera, set_arm, set_camera

        if action == "set_arm":
            return self._do_set_arm(kwargs, set_arm)
        if action == "remove_arm":
            return self._do_remove_arm(kwargs, remove_arm)
        if action == "set_camera":
            return self._do_set_camera(kwargs, set_camera)
        return self._do_remove_camera(kwargs, remove_camera)

    @staticmethod
    def _do_set_arm(kwargs: dict, fn: Any) -> str:
        from roboclaw.embodied.setup import arm_display_name, find_arm

        alias = kwargs.get("name", "")
        arm_type = kwargs.get("arm_type", "")
        port = kwargs.get("port", "")
        if not all([alias, arm_type, port]):
            return "set_arm requires name, arm_type, and port."
        updated = fn(alias, arm_type, port)
        arm = find_arm(updated["arms"], alias)
        display = arm_display_name(arm)
        return f"Arm '{display}' configured.\n{json.dumps(arm, indent=2)}"

    @staticmethod
    def _do_remove_arm(kwargs: dict, fn: Any) -> str:
        alias = kwargs.get("name", "")
        if not alias:
            return "remove_arm requires name."
        fn(alias)
        return f"Arm '{alias}' removed."

    @staticmethod
    def _do_set_camera(kwargs: dict, fn: Any) -> str:
        name = kwargs.get("camera_name", "")
        index = kwargs.get("camera_index")
        if not name or index is None:
            return "set_camera requires camera_name and camera_index."
        updated = fn(name, index)
        return f"Camera '{name}' configured.\n{json.dumps(updated['cameras'][name], indent=2)}"

    @staticmethod
    def _do_remove_camera(kwargs: dict, fn: Any) -> str:
        name = kwargs.get("camera_name", "")
        if not name:
            return "remove_camera requires camera_name."
        fn(name)
        return f"Camera '{name}' removed."

    async def _do_doctor(self, setup: dict) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner

        result = await self._run(LocalLeRobotRunner(), SO101Controller().doctor())
        return result + f"\n\nCurrent setup:\n{json.dumps(setup, indent=2, ensure_ascii=False)}"

    async def _do_identify(self, setup: dict) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner

        if not self._tty_handoff:
            return _NO_TTY_MSG
        ports = setup.get("scanned_ports", [])
        if not ports:
            return "No serial ports detected. Run onboard first."
        argv = [sys.executable, "-m", "roboclaw.embodied.identify", json.dumps(ports)]
        rc = await self._run_tty(LocalLeRobotRunner(), argv, "identify-arms")
        if rc == 0:
            return "Arm identification complete. Use setup_show to see results."
        return f"Arm identification failed (exit {rc})."

    async def _do_calibrate(self, setup: dict) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner
        from roboclaw.embodied.setup import arm_display_name, mark_arm_calibrated

        arms = setup.get("arms", [])
        if not arms:
            return "No arms configured in setup. Use set_arm to add arms first."
        uncalibrated = [a for a in arms if a.get("calibrated") is not True]
        if not uncalibrated:
            return "All arms are already calibrated."
        if not self._tty_handoff:
            return _NO_TTY_MSG
        controller = SO101Controller()
        runner = LocalLeRobotRunner()
        succeeded, failed = 0, 0
        results = []
        for arm in uncalibrated:
            display = arm_display_name(arm)
            argv = controller.calibrate(arm["type"], arm["port"], arm.get("calibration_dir", ""))
            returncode = await self._run_tty(runner, argv, f"lerobot-calibrate ({display})")
            if returncode == 0:
                succeeded += 1
                mark_arm_calibrated(arm["alias"])
                results.append(f"{display}: OK")
            else:
                failed += 1
                results.append(f"{display}: FAILED (exit {returncode})")
        return f"{succeeded} succeeded, {failed} failed.\n" + "\n".join(results)

    async def _do_teleoperate(self, setup: dict, kwargs: dict) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner
        from roboclaw.embodied.setup import arm_display_name

        resolved = _resolve_operation_arms(setup, kwargs.get("follower_names", ""), kwargs.get("leader_names", ""))
        if isinstance(resolved, str):
            return resolved
        if not self._tty_handoff:
            return _NO_TTY_MSG
        controller = SO101Controller()
        followers, leaders, mode = resolved["followers"], resolved["leaders"], resolved["mode"]
        if mode == "single":
            f, l = followers[0], leaders[0]
            argv = controller.teleoperate(
                robot_type=f["type"], robot_port=f["port"], robot_cal_dir=f["calibration_dir"],
                teleop_type=l["type"], teleop_port=l["port"], teleop_cal_dir=l["calibration_dir"],
            )
            label = f"lerobot-teleoperate ({arm_display_name(f)} + {arm_display_name(l)})"
        else:
            argv = controller.teleoperate_bimanual(
                left_robot=followers[0], right_robot=followers[1],
                left_teleop=leaders[0], right_teleop=leaders[1],
            )
            label = "lerobot-teleoperate (bimanual)"
        rc = await self._run_tty(LocalLeRobotRunner(), argv, label)
        return "Teleoperation finished." if rc == 0 else f"Teleoperation failed (exit {rc})."

    async def _do_record(self, setup: dict, kwargs: dict) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.runner import LocalLeRobotRunner

        resolved = _resolve_operation_arms(setup, kwargs.get("follower_names", ""), kwargs.get("leader_names", ""))
        if isinstance(resolved, str):
            return resolved
        if not self._tty_handoff:
            return _NO_TTY_MSG
        controller = SO101Controller()
        followers, leaders, mode = resolved["followers"], resolved["leaders"], resolved["mode"]
        cameras = self._resolve_cameras(setup)
        dataset_name = kwargs.get("dataset_name", "default")
        record_kwargs = {
            "cameras": cameras,
            "repo_id": f"local/{dataset_name}",
            "task": kwargs.get("task", "default_task"),
            "fps": kwargs.get("fps", 30),
            "num_episodes": kwargs.get("num_episodes", 10),
        }
        if mode == "single":
            f, l = followers[0], leaders[0]
            argv = controller.record(
                robot_type=f["type"], robot_port=f["port"], robot_cal_dir=f["calibration_dir"],
                teleop_type=l["type"], teleop_port=l["port"], teleop_cal_dir=l["calibration_dir"],
                **record_kwargs,
            )
        else:
            argv = controller.record_bimanual(
                left_robot=followers[0], right_robot=followers[1],
                left_teleop=leaders[0], right_teleop=leaders[1],
                **record_kwargs,
            )
        rc = await self._run_tty(LocalLeRobotRunner(), argv, "lerobot-record")
        return "Recording finished." if rc == 0 else f"Recording failed (exit {rc})."

    async def _do_train(self, setup: dict, kwargs: dict) -> str:
        from roboclaw.embodied.learning.act import ACTPipeline
        from roboclaw.embodied.runner import LocalLeRobotRunner

        dataset_name = kwargs.get("dataset_name", "default")
        dataset_root = setup.get("datasets", {}).get("root", "")
        policies_root = setup.get("policies", {}).get("root", "")
        argv = ACTPipeline().train(
            repo_id=f"local/{dataset_name}",
            dataset_root=dataset_root,
            output_dir=policies_root,
            steps=kwargs.get("steps", 100_000),
            device=kwargs.get("device", "cuda"),
        )
        job_id = await LocalLeRobotRunner().run_detached(argv=argv, log_dir=_LOGS_DIR)
        return f"Training started. Job ID: {job_id}"

    async def _do_run_policy(self, setup: dict, kwargs: dict) -> str:
        from roboclaw.embodied.embodiment.so101 import SO101Controller
        from roboclaw.embodied.learning.act import ACTPipeline
        from roboclaw.embodied.runner import LocalLeRobotRunner

        arms = setup.get("arms", [])
        followers = [a for a in arms if "follower" in a.get("type", "")]
        if not followers:
            return "No follower arm configured. Use set_arm to add one."
        follower = followers[0]
        cameras = self._resolve_cameras(setup)
        policies_root = setup.get("policies", {}).get("root", "")
        checkpoint = kwargs.get("checkpoint_path") or ACTPipeline().checkpoint_path(policies_root)
        argv = SO101Controller().run_policy(
            robot_type=follower["type"], robot_port=follower["port"], robot_cal_dir=follower["calibration_dir"],
            cameras=cameras, policy_path=checkpoint,
            num_episodes=kwargs.get("num_episodes", 1),
        )
        return await self._run(LocalLeRobotRunner(), argv)

    async def _do_job_status(self, kwargs: dict) -> str:
        from roboclaw.embodied.runner import LocalLeRobotRunner

        job_id = kwargs.get("job_id", "")
        status = await LocalLeRobotRunner().job_status(job_id=job_id, log_dir=_LOGS_DIR)
        return "\n".join(f"{k}: {v}" for k, v in status.items())

    def _resolve_cameras(self, setup: dict) -> dict[str, dict]:
        """Convert setup cameras to LeRobot camera format {name: {type, index_or_path}}."""
        cameras = setup.get("cameras", {})
        result = {}
        for name, cam in cameras.items():
            path = cam.get("by_path") or cam.get("dev", "")
            if not path:
                continue
            result[name] = {"type": "opencv", "index_or_path": path}
        return result

    async def _run_tty(self, runner: Any, argv: list[str], label: str) -> int:
        """Run interactive command with TTY handoff. Always calls stop even on error."""
        await self._tty_handoff(start=True, label=label)
        try:
            return await runner.run_interactive(argv)
        finally:
            await self._tty_handoff(start=False, label=label)

    @staticmethod
    async def _run(runner: Any, argv: list[str]) -> str:
        returncode, stdout, stderr = await runner.run(argv)
        if returncode != 0:
            return f"Command failed (exit {returncode}).\nstdout: {stdout}\nstderr: {stderr}"
        return stdout or "Done."


def _resolve_operation_arms(
    setup: dict, follower_names_str: str, leader_names_str: str,
) -> dict[str, Any] | str:
    """Resolve follower/leader arm aliases to arm dicts from setup.

    Returns dict with keys: followers (list), leaders (list), mode ("single"/"bimanual").
    Returns error string if resolution fails.
    """
    from roboclaw.embodied.setup import find_arm

    arms = setup.get("arms", [])
    if not arms:
        return "No arms configured. Use set_arm to add arms first."
    follower_names = _parse_names(follower_names_str)
    leader_names = _parse_names(leader_names_str)
    if not follower_names or not leader_names:
        return _auto_resolve(arms, follower_names, leader_names)
    followers = _lookup_arms(arms, follower_names, "follower")
    if isinstance(followers, str):
        return followers
    leaders = _lookup_arms(arms, leader_names, "leader")
    if isinstance(leaders, str):
        return leaders
    return _build_result(followers, leaders)


def _parse_names(names_str: str) -> list[str]:
    """Split comma-separated names into a list, stripping whitespace."""
    if not names_str:
        return []
    return [n.strip() for n in names_str.split(",") if n.strip()]


def _auto_resolve(
    arms: list[dict], follower_names: list[str], leader_names: list[str],
) -> dict[str, Any] | str:
    """Auto-resolve when one or both name lists are empty."""
    all_followers = [a for a in arms if "follower" in a.get("type", "")]
    all_leaders = [a for a in arms if "leader" in a.get("type", "")]

    followers = _lookup_arms(arms, follower_names, "follower") if follower_names else all_followers
    if isinstance(followers, str):
        return followers
    leaders = _lookup_arms(arms, leader_names, "leader") if leader_names else all_leaders
    if isinstance(leaders, str):
        return leaders

    if not followers:
        return "No follower arms configured. Use set_arm to add arms."
    if not leaders:
        return "No leader arms configured. Use set_arm to add arms."
    return _build_result(followers, leaders)


def _lookup_arms(arms: list[dict], names: list[str], label: str) -> list[dict] | str:
    """Look up arms by alias. Returns list of dicts or error string."""
    from roboclaw.embodied.setup import find_arm

    result = []
    for name in names:
        arm = find_arm(arms, name)
        if arm is None:
            return f"No {label} arm named '{name}' found in setup."
        result.append(arm)
    return result


def _build_result(followers: list[dict], leaders: list[dict]) -> dict[str, Any] | str:
    """Build the resolved result dict, validating counts."""
    if len(followers) != len(leaders):
        return f"Follower/leader count mismatch: {len(followers)} followers, {len(leaders)} leaders."
    if len(followers) == 1:
        return {"followers": followers, "leaders": leaders, "mode": "single"}
    if len(followers) == 2:
        return {"followers": followers, "leaders": leaders, "mode": "bimanual"}
    return f"Unsupported arm count: {len(followers)}. Use 1 (single) or 2 (bimanual)."
