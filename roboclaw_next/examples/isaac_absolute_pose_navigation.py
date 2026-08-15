"""Navigate Nova Carter to an absolute simulator pose.

This is the smallest navigation loop we want for RoboClaw simulation:
- no SLAM
- no localization filter
- current pose comes directly from Isaac Sim world state
- path generation uses Isaac Sim quintic path planner
- wheel commands use Isaac Sim wheeled robot controllers
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "width": 1280, "height": 720})

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import carb
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.core.utils.stage import open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.controllers import DifferentialController
from isaacsim.robot.wheeled_robots.controllers.quintic_path_planner import quintic_polynomials_planner
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

KITCHEN_USD = "/home/zhaobo/roboclaw_sim/assets/scenes/Lightwheel_Kitchen/Collected_KitchenRoom/KitchenRoom_visual_preview.usdc"
OUT_DIR = Path("/home/zhaobo/roboclaw_sim/renders")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ROBOT_PRIM = "/World/NovaCarterNavigator"


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def wait_for_stage_loading(timeout_s: float = 180.0) -> None:
    context = omni.usd.get_context()
    start = time.time()
    while context.get_stage_loading_status()[2] > 0:
        simulation_app.update()
        if time.time() - start > timeout_s:
            raise TimeoutError("stage loading timed out")


def step(world: World, frames: int, render: bool = True) -> None:
    for _ in range(frames):
        world.step(render=render)


def pose_xy_yaw(robot: WheeledRobot) -> tuple[np.ndarray, float]:
    position, orientation = robot.get_world_pose()
    yaw = float(quat_to_euler_angles(orientation)[-1])
    return position, yaw


def as_list(values) -> list[float]:
    return [float(x) for x in values]


class FrameRecorder:
    def __init__(self, *, frame_dir: Path, video_path: Path, capture_every: int, fps: int) -> None:
        self.frame_dir = frame_dir
        self.video_path = video_path
        self.capture_every = capture_every
        self.fps = fps
        self.frame_index = 0
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in self.frame_dir.glob("frame_*.png"):
            old_frame.unlink()
        if self.video_path.exists():
            self.video_path.unlink()

    def maybe_capture(self, frame: int) -> None:
        if frame % self.capture_every != 0:
            return
        self.capture()

    def capture(self) -> None:
        frame_path = self.frame_dir / f"frame_{self.frame_index:05d}.png"
        if frame_path.exists():
            frame_path.unlink()
        capture_viewport_to_file(get_active_viewport(), str(frame_path), is_hdr=False)
        for _ in range(180):
            simulation_app.update()
            if frame_path.exists() and frame_path.stat().st_size > 0:
                self.frame_index += 1
                return
        raise RuntimeError(f"video frame was not written: {frame_path}")

    def encode(self) -> None:
        if self.frame_index == 0:
            raise RuntimeError("no captured frames to encode")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(self.fps),
                "-i",
                str(self.frame_dir / "frame_%05d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.video_path),
            ],
            check=True,
        )


def plan_absolute_path(start_position: np.ndarray, start_yaw: float, target_pose: np.ndarray) -> list[np.ndarray]:
    _, rx, ry, _, _, _, _ = quintic_polynomials_planner(
        sx=float(start_position[0]),
        sy=float(start_position[1]),
        syaw=start_yaw,
        sv=0.0,
        sa=0.0,
        gx=float(target_pose[0]),
        gy=float(target_pose[1]),
        gyaw=float(target_pose[2]),
        gv=0.0,
        ga=0.0,
        max_accel=0.8,
        max_jerk=0.8,
        dt=0.25,
    )
    points = [np.array([x, y, 0.0]) for i, (x, y) in enumerate(zip(rx, ry)) if i % 2 == 0]
    if not points or np.linalg.norm(points[-1][:2] - target_pose[:2]) > 1e-6:
        points.append(np.array([target_pose[0], target_pose[1], 0.0]))
    return points


def compute_wheel_command(
    *,
    current_position: np.ndarray,
    current_yaw: float,
    goal_position: np.ndarray,
    target_pose: np.ndarray,
    position_tol: float,
) -> np.ndarray:
    distance_to_goal = float(np.linalg.norm(current_position[:2] - goal_position[:2]))
    distance_to_target = float(np.linalg.norm(current_position[:2] - target_pose[:2]))
    if distance_to_target <= position_tol:
        yaw_error = normalize_angle(float(target_pose[2]) - current_yaw)
        return np.array([0.0, float(np.clip(1.4 * yaw_error, -0.5, 0.5))])

    goal_yaw = math.atan2(
        float(goal_position[1] - current_position[1]),
        float(goal_position[0] - current_position[0]),
    )
    heading_error = normalize_angle(goal_yaw - current_yaw)

    angular_velocity = float(np.clip(1.8 * heading_error, -0.85, 0.85))
    if abs(heading_error) > 0.9:
        return np.array([0.0, angular_velocity])

    heading_scale = max(0.15, math.cos(heading_error))
    distance_scale = min(1.0, max(distance_to_goal, distance_to_target) / 0.45)
    linear_velocity = float(np.clip(0.38 * heading_scale * distance_scale, 0.08, 0.38))
    return np.array([linear_velocity, angular_velocity])


def align_to_yaw(
    world: World,
    robot: WheeledRobot,
    diff_controller: DifferentialController,
    target_yaw: float,
    yaw_tol: float,
    max_frames: int,
    recorder: FrameRecorder | None,
) -> None:
    for frame in range(max_frames):
        _, current_yaw = pose_xy_yaw(robot)
        yaw_error = normalize_angle(target_yaw - current_yaw)
        if abs(yaw_error) <= yaw_tol:
            break
        angular_velocity = float(np.clip(1.2 * yaw_error, -0.5, 0.5))
        robot.apply_wheel_actions(diff_controller.forward(np.array([0.0, angular_velocity])))
        world.step(render=True)
        if recorder is not None:
            recorder.maybe_capture(frame)
    robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
    step(world, 20)
    if recorder is not None:
        recorder.capture()


def navigate_to_pose(
    world: World,
    robot: WheeledRobot,
    target_pose: np.ndarray,
    position_tol: float,
    yaw_tol: float,
    max_frames: int,
    recorder: FrameRecorder | None = None,
) -> dict:
    diff_controller = DifferentialController(
        name="nova_carter_differential_controller",
        wheel_radius=0.04295,
        wheel_base=0.4132,
        max_linear_speed=0.45,
        max_angular_speed=0.8,
        max_wheel_speed=12.0,
    )

    start_position, start_yaw = pose_xy_yaw(robot)
    waypoints = plan_absolute_path(start_position, start_yaw, target_pose)
    waypoint_index = 0
    trace = []

    for frame in range(max_frames):
        current_position, current_yaw = pose_xy_yaw(robot)
        current_goal = waypoints[waypoint_index]
        distance_to_waypoint = float(np.linalg.norm(current_position[:2] - current_goal[:2]))
        distance_to_target = float(np.linalg.norm(current_position[:2] - target_pose[:2]))

        while distance_to_waypoint <= max(position_tol, 0.16) and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            current_goal = waypoints[waypoint_index]
            distance_to_waypoint = float(np.linalg.norm(current_position[:2] - current_goal[:2]))

        if frame % 30 == 0:
            trace.append(
                {
                    "frame": frame,
                    "position": as_list(current_position),
                    "yaw": current_yaw,
                    "waypoint_index": waypoint_index,
                    "distance_to_target": distance_to_target,
                }
            )

        if distance_to_target <= position_tol:
            robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([0.0, 0.0])))
            step(world, 20)
            if recorder is not None:
                recorder.capture()
            align_to_yaw(world, robot, diff_controller, float(target_pose[2]), yaw_tol, max_frames=360, recorder=recorder)
            break

        command = compute_wheel_command(
            current_position=current_position,
            current_yaw=current_yaw,
            goal_position=current_goal,
            target_pose=target_pose,
            position_tol=position_tol,
        )
        robot.apply_wheel_actions(diff_controller.forward(command))
        world.step(render=True)
        if recorder is not None:
            recorder.maybe_capture(frame)

    final_position, final_yaw = pose_xy_yaw(robot)
    position_error = float(np.linalg.norm(final_position[:2] - target_pose[:2]))
    yaw_error = abs(normalize_angle(float(target_pose[2]) - final_yaw))
    arrived = bool(position_error <= position_tol and yaw_error <= yaw_tol)
    return {
        "arrived": arrived,
        "start_position": as_list(start_position),
        "start_yaw": start_yaw,
        "target_pose_xy_yaw": as_list(target_pose),
        "final_position": as_list(final_position),
        "final_yaw": final_yaw,
        "position_error": position_error,
        "yaw_error": yaw_error,
        "waypoint_count": len(waypoints),
        "waypoints": [as_list(point) for point in waypoints],
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-yaw", type=float, required=True, help="Target yaw in radians, world frame.")
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=-3.1)
    parser.add_argument("--position-tol", type=float, default=0.08)
    parser.add_argument("--yaw-tol", type=float, default=0.12)
    parser.add_argument("--max-frames", type=int, default=3600)
    parser.add_argument("--output-prefix", default="nova_carter_absolute_nav")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--capture-every", type=int, default=6)
    parser.add_argument("--video-fps", type=int, default=8)
    args = parser.parse_args()

    print(f"OPEN_STAGE {KITCHEN_USD}", flush=True)
    if not open_stage(KITCHEN_USD):
        raise RuntimeError("failed to open kitchen stage")
    wait_for_stage_loading()

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac assets root is not available")
    carter_usd = assets_root + "/Isaac/Robots/NVIDIA/Carter/nova_carter/nova_carter.usd"

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    robot = world.scene.add(
        WheeledRobot(
            prim_path=ROBOT_PRIM,
            name="nova_carter_navigator",
            wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
            create_robot=True,
            usd_path=carter_usd,
            position=np.array([args.start_x, args.start_y, 0.0]),
        )
    )
    world.reset()
    world.play()
    step(world, 90)

    settings = carb.settings.get_settings()
    settings.set("/persistent/app/viewport/displayOptions", 0)
    settings.set("/app/renderer/resolution/width", 1280)
    settings.set("/app/renderer/resolution/height", 720)
    settings.set("/rtx/post/tonemap/filmIso", 80)
    set_camera_view(eye=[0.0, -7.0, 2.8], target=[0.0, -2.1, 0.65], camera_prim_path="/OmniverseKit_Persp")
    step(world, 80)

    recorder = None
    if args.record_video:
        recorder = FrameRecorder(
            frame_dir=OUT_DIR / f"{args.output_prefix}_frames",
            video_path=OUT_DIR / f"{args.output_prefix}.mp4",
            capture_every=args.capture_every,
            fps=args.video_fps,
        )
        recorder.capture()

    target_pose = np.array([args.target_x, args.target_y, args.target_yaw])
    report = navigate_to_pose(
        world,
        robot,
        target_pose,
        args.position_tol,
        args.yaw_tol,
        args.max_frames,
        recorder=recorder,
    )
    report["carter_usd"] = carter_usd
    report["robot_prim"] = ROBOT_PRIM
    report["position_source"] = "Isaac Sim ground-truth world pose via robot.get_world_pose()"
    report["planner"] = "isaacsim.robot.wheeled_robots.controllers.quintic_path_planner.quintic_polynomials_planner"
    report["controller"] = "Ground-truth waypoint follower + DifferentialController"

    if recorder is not None:
        recorder.capture()
        recorder.encode()
        report["video"] = str(recorder.video_path)
        report["video_frame_count"] = recorder.frame_index

    step(world, 100)

    screenshot = OUT_DIR / f"{args.output_prefix}.png"
    if screenshot.exists():
        screenshot.unlink()
    print(f"CAPTURE {screenshot}", flush=True)
    capture_viewport_to_file(get_active_viewport(), str(screenshot), is_hdr=False)
    for _ in range(360):
        simulation_app.update()
        if screenshot.exists() and screenshot.stat().st_size > 0:
            break
    report["screenshot"] = str(screenshot)

    report_path = OUT_DIR / f"{args.output_prefix}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps({k: report[k] for k in ["arrived", "target_pose_xy_yaw", "final_position", "final_yaw", "position_error", "yaw_error", "waypoint_count"]}, ensure_ascii=False), flush=True)
    print(f"REPORT {report_path}", flush=True)
    print(f"SCREENSHOT {screenshot}", flush=True)
    if recorder is not None:
        print(f"VIDEO {recorder.video_path}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
