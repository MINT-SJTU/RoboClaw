"""Stage-1 ROS2 bridge server for framework-owned embodied execution."""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any

from roboclaw.embodied.execution.integration.adapters.ros2.profiles import (
    SO101_ROS2_PROFILE,
    get_ros2_profile,
)
from roboclaw.embodied.execution.integration.bridges.ros2.so101_feetech import So101FeetechRuntime


class Stage1Ros2BridgeServer:
    """Expose a minimal ROS2 lifecycle and primitive surface."""

    def __init__(
        self,
        *,
        namespace: str,
        profile_id: str,
        robot_id: str,
        device: str,
        calibration_path: str | None,
        calibration_id: str,
        state_rate_hz: float,
    ) -> None:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self._rclpy = rclpy
        self._string = String
        self._namespace = namespace.rstrip("/")
        self._profile_id = profile_id
        self._robot_id = robot_id
        self._lock = threading.RLock()
        self._runtime = self._build_runtime(
            profile_id=profile_id,
            robot_id=robot_id,
            device=device,
            calibration_path=calibration_path,
            calibration_id=calibration_id,
        )
        self._last_primitive: str | None = None
        self._last_error: str | None = None
        self._last_result: dict[str, Any] = {}

        class BridgeNode(Node):
            pass

        self.node = BridgeNode("roboclaw_stage1_bridge")
        self._state_publisher = self.node.create_publisher(String, f"{self._namespace}/state", 10)
        self._health_publisher = self.node.create_publisher(String, f"{self._namespace}/health", 10)
        self._events_publisher = self.node.create_publisher(String, f"{self._namespace}/events", 10)
        self.node.create_timer(max(0.25, 1.0 / max(state_rate_hz, 0.1)), self._publish_status)

        self.node.create_service(Trigger, f"{self._namespace}/connect", self._handle_connect)
        self.node.create_service(Trigger, f"{self._namespace}/disconnect", self._handle_disconnect)
        self.node.create_service(Trigger, f"{self._namespace}/stop", self._handle_stop)
        self.node.create_service(Trigger, f"{self._namespace}/reset", self._handle_reset)
        self.node.create_service(Trigger, f"{self._namespace}/recover", self._handle_recover)
        self.node.create_service(Trigger, f"{self._namespace}/debug_snapshot", self._handle_debug_snapshot)
        self.node.create_service(Trigger, f"{self._namespace}/primitive_gripper_open", self._handle_gripper_open)
        self.node.create_service(Trigger, f"{self._namespace}/primitive_gripper_close", self._handle_gripper_close)
        self.node.create_service(Trigger, f"{self._namespace}/primitive_go_home", self._handle_go_home)

    def _build_runtime(
        self,
        *,
        profile_id: str,
        robot_id: str,
        device: str,
        calibration_path: str | None,
        calibration_id: str,
    ) -> Any:
        profile = get_ros2_profile(profile_id)
        robot_profile = get_ros2_profile(robot_id)
        if profile is None:
            raise ValueError(
                f"Unknown stage-1 ROS2 profile '{profile_id}'."
            )
        if robot_profile is None:
            raise ValueError(f"Unknown stage-1 ROS2 robot '{robot_id}'.")
        if profile.id != robot_profile.id:
            raise ValueError(
                f"Stage-1 ROS2 bridge profile/robot mismatch: profile='{profile.id}' robot='{robot_id}'."
            )
        if profile.id != SO101_ROS2_PROFILE.id:
            raise ValueError(f"Stage-1 ROS2 bridge does not support profile '{profile.id}' yet.")
        return So101FeetechRuntime(
            device=device,
            calibration_path=calibration_path,
            calibration_id=calibration_id,
        )

    def spin(self) -> None:
        self._rclpy.spin(self.node)

    def shutdown(self) -> None:
        try:
            with self._lock:
                self._runtime.disconnect()
        except Exception:
            pass
        self.node.destroy_node()
        try:
            self._rclpy.shutdown()
        except Exception:
            pass

    def _status_payload(self) -> dict[str, Any]:
        with self._lock:
            try:
                snapshot = self._runtime.snapshot()
            except Exception as exc:
                snapshot = {
                    "connected": self._runtime.connected,
                    "snapshot_error": str(exc),
                }
        return {
            "profile_id": self._profile_id,
            "robot_id": self._robot_id,
            "connected": snapshot.get("connected"),
            "device": snapshot.get("device"),
            "gripper_present_raw": snapshot.get("gripper_present_raw"),
            "gripper_percent": snapshot.get("gripper_percent"),
            "last_primitive": self._last_primitive,
            "last_error": self._last_error,
            "snapshot_error": snapshot.get("snapshot_error"),
        }

    def _publish_status(self) -> None:
        try:
            status = json.dumps(self._status_payload(), ensure_ascii=False, sort_keys=True)
            self._state_publisher.publish(self._string(data=status))
            health = json.dumps({"ok": self._last_error is None, "last_error": self._last_error}, ensure_ascii=False)
            self._health_publisher.publish(self._string(data=health))
            event = json.dumps({"primitive": self._last_primitive, "error": self._last_error}, ensure_ascii=False)
            self._events_publisher.publish(self._string(data=event))
        except Exception as exc:
            self._last_error = str(exc)

    def _snapshot_locked(self) -> dict[str, Any]:
        with self._lock:
            return self._runtime.snapshot()

    def _success(self, response: Any, message: str, result: dict[str, Any] | None = None) -> Any:
        self._last_error = None
        self._last_result = result or {}
        response.success = True
        response.message = message
        return response

    def _failure(self, response: Any, exc: Exception) -> Any:
        self._last_error = str(exc)
        response.success = False
        response.message = str(exc)
        return response

    def _handle_connect(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                self._runtime.connect()
                result = self._runtime.snapshot()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "connected", result)

    def _handle_disconnect(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                self._runtime.disconnect()
                result = self._runtime.snapshot()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "disconnected", result)

    def _handle_stop(self, request: Any, response: Any) -> Any:
        del request
        return self._success(response, "stop acknowledged", self._snapshot_locked())

    def _handle_reset(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                if not self._runtime.connected:
                    self._runtime.connect()
                self._last_primitive = "go_named_pose"
                result = self._runtime.go_home()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "reset to home", result)

    def _handle_recover(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                self._runtime.disconnect()
                self._runtime.connect()
                result = self._runtime.snapshot()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "recovered", result)

    def _handle_debug_snapshot(self, request: Any, response: Any) -> Any:
        del request
        payload = self._status_payload()
        return self._success(response, json.dumps(payload, ensure_ascii=False, sort_keys=True), payload)

    def _handle_gripper_open(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                if not self._runtime.connected:
                    self._runtime.connect()
                self._last_primitive = "gripper_open"
                result = self._runtime.open_gripper()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "gripper opened", result)

    def _handle_gripper_close(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                if not self._runtime.connected:
                    self._runtime.connect()
                self._last_primitive = "gripper_close"
                result = self._runtime.close_gripper()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "gripper closed", result)

    def _handle_go_home(self, request: Any, response: Any) -> Any:
        del request
        with self._lock:
            try:
                if not self._runtime.connected:
                    self._runtime.connect()
                self._last_primitive = "go_named_pose"
                result = self._runtime.go_home()
            except Exception as exc:
                return self._failure(response, exc)
        return self._success(response, "home pose reached", result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RoboClaw stage-1 ROS2 bridge.")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--calibration-path")
    parser.add_argument("--calibration-id", default="so101_real")
    parser.add_argument("--state-rate-hz", type=float, default=4.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    import rclpy

    rclpy.init()
    server = Stage1Ros2BridgeServer(
        namespace=args.namespace,
        profile_id=args.profile_id,
        robot_id=args.robot_id,
        device=args.device,
        calibration_path=args.calibration_path,
        calibration_id=args.calibration_id,
        state_rate_hz=args.state_rate_hz,
    )
    try:
        server.spin()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
