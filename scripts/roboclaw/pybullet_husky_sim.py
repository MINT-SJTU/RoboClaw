#!/usr/bin/env python3

import json
import sys

import pybullet as p
import pybullet_data

RESULT_MARKER = "ROBOCLAW_PYBULLET_RESULT="
WHEEL_JOINTS = (2, 3, 4, 5)


def emit(payload: dict) -> None:
    print(f"{RESULT_MARKER}{json.dumps(payload, separators=(',', ':'))}")


def round_position(position: tuple[float, float, float]) -> list[float]:
    return [round(value, 4) for value in position]


def execute(request: dict) -> dict:
    semantic_action = request.get("semanticAction")
    parameters = request.get("parameters", {})
    requested_meters = float(parameters.get("meters", 0.0))

    if semantic_action != "base.move.forward":
        return {
            "success": False,
            "simulator": "pybullet",
            "robotModel": "husky",
            "semanticAction": str(semantic_action),
            "requestedMeters": requested_meters,
            "observedForwardMeters": 0.0,
            "lateralDriftMeters": 0.0,
            "startPosition": [0.0, 0.0, 0.0],
            "endPosition": [0.0, 0.0, 0.0],
            "steps": 0,
            "detail": f"Unsupported semantic action: {semantic_action}",
        }

    if requested_meters <= 0 or requested_meters > 2.0:
        return {
            "success": False,
            "simulator": "pybullet",
            "robotModel": "husky",
            "semanticAction": semantic_action,
            "requestedMeters": requested_meters,
            "observedForwardMeters": 0.0,
            "lateralDriftMeters": 0.0,
            "startPosition": [0.0, 0.0, 0.0],
            "endPosition": [0.0, 0.0, 0.0],
            "steps": 0,
            "detail": "Requested meters must be between 0 and 2.0.",
        }

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    robot = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

    for _ in range(120):
        p.stepSimulation()

    start_position, _ = p.getBasePositionAndOrientation(robot)
    wheel_velocity = float(parameters.get("wheelVelocity", 10.0))
    wheel_force = float(parameters.get("wheelForce", 40.0))
    max_steps = int(parameters.get("maxSteps", 1200))

    success = False
    steps = 0
    end_position = start_position

    while steps < max_steps:
        for joint in WHEEL_JOINTS:
            p.setJointMotorControl2(
                robot,
                joint,
                p.VELOCITY_CONTROL,
                targetVelocity=wheel_velocity,
                force=wheel_force,
            )
        p.stepSimulation()
        steps += 1
        end_position, _ = p.getBasePositionAndOrientation(robot)
        if (end_position[0] - start_position[0]) >= requested_meters:
            success = True
            break

    for joint in WHEEL_JOINTS:
        p.setJointMotorControl2(
            robot,
            joint,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=wheel_force,
        )

    end_position, _ = p.getBasePositionAndOrientation(robot)
    p.disconnect(client_id)

    observed_forward_meters = end_position[0] - start_position[0]
    lateral_drift_meters = end_position[1] - start_position[1]

    return {
        "success": success,
        "simulator": "pybullet",
        "robotModel": "husky",
        "semanticAction": semantic_action,
        "requestedMeters": round(requested_meters, 4),
        "observedForwardMeters": round(observed_forward_meters, 4),
        "lateralDriftMeters": round(lateral_drift_meters, 4),
        "startPosition": round_position(start_position),
        "endPosition": round_position(end_position),
        "steps": steps,
        "detail": (
            f"PyBullet Husky moved {observed_forward_meters:.3f} meters forward in {steps} steps."
            if success
            else f"PyBullet Husky reached only {observed_forward_meters:.3f} meters before timeout."
        ),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] != "execute":
        print(
            "Usage: pybullet_husky_sim.py execute '<json-request>'",
            file=sys.stderr,
        )
        return 1

    request = json.loads(argv[2])
    payload = execute(request)
    emit(payload)
    return 0 if payload["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
