#!/usr/bin/env python3

import json
import math
import sys
from pathlib import Path

import pybullet as p
import pybullet_data

RESULT_MARKER = "ROBOCLAW_SO101_RESULT="
JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
ASSET_SHA = "583899971f978b1b03664a1fa25dd377cfe429c6"
BASE_POSE = {joint: 0.0 for joint in JOINT_ORDER}
GREETING_POSE = {
    "shoulder_pan": 0.38,
    "shoulder_lift": -0.72,
    "elbow_flex": 1.18,
    "wrist_flex": -0.55,
    "wrist_roll": 0.0,
    "gripper": 0.24,
}
WAVE_LEFT_POSE = {**GREETING_POSE, "wrist_roll": -0.85}
WAVE_RIGHT_POSE = {**GREETING_POSE, "wrist_roll": 0.85}


def emit(payload: dict) -> None:
    print(f"{RESULT_MARKER}{json.dumps(payload, separators=(',', ':'))}")


def round_float(value: float) -> float:
    return round(float(value), 4)


def round_vector(values: tuple[float, float, float] | list[float]) -> list[float]:
    return [round_float(value) for value in values]


def get_joint_map(robot: int) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for joint_index in range(p.getNumJoints(robot)):
        joint_name = p.getJointInfo(robot, joint_index)[1].decode()
        if joint_name in JOINT_ORDER or joint_name == "gripper_frame_joint":
            mapping[joint_name] = joint_index
    return mapping


def get_joint_positions(robot: int, joint_map: dict[str, int]) -> dict[str, float]:
    return {
        joint_name: round_float(p.getJointState(robot, joint_map[joint_name])[0])
        for joint_name in JOINT_ORDER
    }


def get_end_effector_position(robot: int, joint_map: dict[str, int]) -> tuple[float, float, float]:
    link_state = p.getLinkState(robot, joint_map["gripper_frame_joint"])
    return link_state[0]


def apply_pose(
    robot: int,
    joint_map: dict[str, int],
    target_pose: dict[str, float],
    steps: int,
    peak_joint_excursions: dict[str, float],
    ee_path_positions: list[tuple[float, float, float]],
) -> int:
    for joint_name, target in target_pose.items():
        p.setJointMotorControl2(
            robot,
            joint_map[joint_name],
            p.POSITION_CONTROL,
            targetPosition=float(target),
            force=20.0,
            maxVelocity=1.6,
            positionGain=0.22,
            velocityGain=0.9,
        )

    for _ in range(steps):
        p.stepSimulation()
        ee_path_positions.append(get_end_effector_position(robot, joint_map))
        for joint_name in JOINT_ORDER:
            position = abs(p.getJointState(robot, joint_map[joint_name])[0])
            peak_joint_excursions[joint_name] = max(peak_joint_excursions[joint_name], position)

    return steps


def compute_path_length(positions: list[tuple[float, float, float]]) -> float:
    path_length = 0.0
    for index in range(1, len(positions)):
        prev = positions[index - 1]
        current = positions[index]
        path_length += math.dist(prev, current)
    return path_length


def execute(request: dict) -> dict:
    semantic_action = request.get("semanticAction")
    urdf_path = Path(str(request.get("urdfPath", "")))
    trace_path_raw = request.get("tracePath")
    trace_path = Path(str(trace_path_raw)) if trace_path_raw else None
    sequence_name = str(request.get("parameters", {}).get("sequenceName", ""))

    if semantic_action != "arm.wave":
        return {
            "success": False,
            "simulator": "pybullet",
            "robotModel": "so101",
            "semanticAction": str(semantic_action),
            "sequenceName": sequence_name or "unknown",
            "jointOrder": list(JOINT_ORDER),
            "startJointPositions": {},
            "finalJointPositions": {},
            "peakJointExcursions": {},
            "endEffectorStart": [0.0, 0.0, 0.0],
            "endEffectorFinal": [0.0, 0.0, 0.0],
            "endEffectorPathMeters": 0.0,
            "endEffectorLiftMeters": 0.0,
            "steps": 0,
            "assetSource": f"SO101-Classic-Control@{ASSET_SHA}",
            "tracePath": str(trace_path) if trace_path else None,
            "detail": f"Unsupported semantic action: {semantic_action}",
        }

    if sequence_name != "greeting_wave":
        return {
            "success": False,
            "simulator": "pybullet",
            "robotModel": "so101",
            "semanticAction": str(semantic_action),
            "sequenceName": sequence_name,
            "jointOrder": list(JOINT_ORDER),
            "startJointPositions": {},
            "finalJointPositions": {},
            "peakJointExcursions": {},
            "endEffectorStart": [0.0, 0.0, 0.0],
            "endEffectorFinal": [0.0, 0.0, 0.0],
            "endEffectorPathMeters": 0.0,
            "endEffectorLiftMeters": 0.0,
            "steps": 0,
            "assetSource": f"SO101-Classic-Control@{ASSET_SHA}",
            "tracePath": str(trace_path) if trace_path else None,
            "detail": f"Unsupported SO101 sequence: {sequence_name}",
        }

    if not urdf_path.is_file():
        return {
            "success": False,
            "simulator": "pybullet",
            "robotModel": "so101",
            "semanticAction": str(semantic_action),
            "sequenceName": sequence_name,
            "jointOrder": list(JOINT_ORDER),
            "startJointPositions": {},
            "finalJointPositions": {},
            "peakJointExcursions": {},
            "endEffectorStart": [0.0, 0.0, 0.0],
            "endEffectorFinal": [0.0, 0.0, 0.0],
            "endEffectorPathMeters": 0.0,
            "endEffectorLiftMeters": 0.0,
            "steps": 0,
            "assetSource": f"SO101-Classic-Control@{ASSET_SHA}",
            "tracePath": str(trace_path) if trace_path else None,
            "detail": f"SO101 URDF not found at {urdf_path}. Run scripts/roboclaw/setup-so101-sim.sh first.",
        }

    client_id = p.connect(p.DIRECT)
    steps = 0
    success = False
    result: dict | None = None

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.loadURDF("plane.urdf")
        robot = p.loadURDF(str(urdf_path), useFixedBase=True)
        joint_map = get_joint_map(robot)
        missing_joints = [joint_name for joint_name in JOINT_ORDER if joint_name not in joint_map]
        if "gripper_frame_joint" not in joint_map or missing_joints:
            return {
                "success": False,
                "simulator": "pybullet",
                "robotModel": "so101",
                "semanticAction": str(semantic_action),
                "sequenceName": sequence_name,
                "jointOrder": list(JOINT_ORDER),
                "startJointPositions": {},
                "finalJointPositions": {},
                "peakJointExcursions": {},
                "endEffectorStart": [0.0, 0.0, 0.0],
                "endEffectorFinal": [0.0, 0.0, 0.0],
                "endEffectorPathMeters": 0.0,
                "endEffectorLiftMeters": 0.0,
                "steps": 0,
                "assetSource": f"SO101-Classic-Control@{ASSET_SHA}",
                "tracePath": str(trace_path) if trace_path else None,
                "detail": f"SO101 URDF is missing required joints: {', '.join(missing_joints)}",
            }

        for joint_name in JOINT_ORDER:
            p.resetJointState(robot, joint_map[joint_name], BASE_POSE[joint_name])

        ee_path_positions = [get_end_effector_position(robot, joint_map)]
        peak_joint_excursions = {joint_name: 0.0 for joint_name in JOINT_ORDER}
        start_joint_positions = get_joint_positions(robot, joint_map)
        start_ee = ee_path_positions[0]

        sequence = [
            (GREETING_POSE, 180),
            (WAVE_LEFT_POSE, 160),
            (WAVE_RIGHT_POSE, 160),
            (WAVE_LEFT_POSE, 160),
            (GREETING_POSE, 160),
        ]

        for pose, pose_steps in sequence:
            steps += apply_pose(robot, joint_map, pose, pose_steps, peak_joint_excursions, ee_path_positions)

        final_joint_positions = get_joint_positions(robot, joint_map)
        final_ee = get_end_effector_position(robot, joint_map)
        end_effector_path = compute_path_length(ee_path_positions)
        success = end_effector_path >= 0.08 and abs(final_joint_positions["shoulder_lift"]) >= 0.5

        result = {
            "success": success,
            "simulator": "pybullet",
            "robotModel": "so101",
            "semanticAction": semantic_action,
            "sequenceName": sequence_name,
            "jointOrder": list(JOINT_ORDER),
            "startJointPositions": start_joint_positions,
            "finalJointPositions": final_joint_positions,
            "peakJointExcursions": {
                joint_name: round_float(value) for joint_name, value in peak_joint_excursions.items()
            },
            "endEffectorStart": round_vector(start_ee),
            "endEffectorFinal": round_vector(final_ee),
            "endEffectorPathMeters": round_float(end_effector_path),
            "endEffectorLiftMeters": round_float(final_ee[2] - start_ee[2]),
            "steps": steps,
            "assetSource": f"SO101-Classic-Control@{ASSET_SHA}",
            "tracePath": str(trace_path) if trace_path else None,
            "detail": (
                f"PyBullet SO101 completed the greeting wave in {steps} steps "
                f"with {end_effector_path:.3f}m of end-effector travel."
                if success
                else f"PyBullet SO101 did not complete a credible wave; path length was {end_effector_path:.3f}m."
            ),
        }
    finally:
        p.disconnect(client_id)

    if trace_path and result is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result if result is not None else {
        "success": False,
        "simulator": "pybullet",
        "robotModel": "so101",
        "semanticAction": str(semantic_action),
        "sequenceName": sequence_name,
        "jointOrder": list(JOINT_ORDER),
        "startJointPositions": {},
        "finalJointPositions": {},
        "peakJointExcursions": {},
        "endEffectorStart": [0.0, 0.0, 0.0],
        "endEffectorFinal": [0.0, 0.0, 0.0],
        "endEffectorPathMeters": 0.0,
        "endEffectorLiftMeters": 0.0,
        "steps": steps,
        "assetSource": f"SO101-Classic-Control@{ASSET_SHA}",
        "tracePath": str(trace_path) if trace_path else None,
        "detail": "SO101 simulation exited unexpectedly.",
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "execute":
        print(
            "Usage: pybullet_so101_sim.py execute '<json-request>'",
            file=sys.stderr,
        )
        return 1

    request = json.loads(argv[2])
    payload = execute(request)
    emit(payload)
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
