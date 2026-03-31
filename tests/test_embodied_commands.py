"""Tests for ArmCommandBuilder and ACTPipeline CLI arg generation."""

import sys

from roboclaw.embodied.embodiment.arm.command_builder import ArmCommandBuilder
from roboclaw.embodied.learning.act import ACTPipeline


def test_doctor_command() -> None:
    argv = ArmCommandBuilder().doctor()
    assert argv[0] == "python3"
    assert "import lerobot" in argv[2]


def test_calibrate_follower() -> None:
    argv = ArmCommandBuilder().calibrate(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/follower",
        "5B14032630",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "calibrate"]
    assert "--robot.type=so101_follower" in argv
    assert "--robot.id=5B14032630" in argv
    assert "--robot.port=/dev/ttyACM0" in argv
    assert any("--robot.calibration_dir=" in arg for arg in argv)


def test_calibrate_leader() -> None:
    argv = ArmCommandBuilder().calibrate(
        "so101_leader",
        "/dev/ttyACM1",
        "/cal/leader",
        "5B14030892",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "calibrate"]
    assert "--teleop.type=so101_leader" in argv
    assert "--teleop.id=5B14030892" in argv
    assert "--teleop.port=/dev/ttyACM1" in argv
    assert any("--teleop.calibration_dir=" in arg for arg in argv)


def test_teleoperate() -> None:
    argv = ArmCommandBuilder().teleoperate(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "5B14032630",
        "so101_leader",
        "/dev/ttyACM1",
        "/cal/l",
        "5B14030892",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "teleoperate"]
    assert "--robot.type=so101_follower" in argv
    assert "--robot.id=5B14032630" in argv
    assert "--teleop.type=so101_leader" in argv
    assert "--teleop.id=5B14030892" in argv


def test_teleoperate_bimanual() -> None:
    argv = ArmCommandBuilder().teleoperate_bimanual(
        robot_id="bimanual",
        robot_cal_dir="/cal/robot",
        left_robot={"port": "/dev/a"},
        right_robot={"port": "/dev/b"},
        teleop_id="bimanual",
        teleop_cal_dir="/cal/teleop",
        left_teleop={"port": "/dev/c"},
        right_teleop={"port": "/dev/d"},
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "teleoperate"]
    assert "--robot.id=bimanual" in argv
    assert "--robot.calibration_dir=/cal/robot" in argv
    assert "--teleop.id=bimanual" in argv
    assert "--teleop.calibration_dir=/cal/teleop" in argv
    assert "--robot.left_arm_config.port=/dev/a" in argv
    assert "--robot.right_arm_config.port=/dev/b" in argv
    assert "--teleop.left_arm_config.port=/dev/c" in argv
    assert "--teleop.right_arm_config.port=/dev/d" in argv
    assert not any(".left_arm_config.calibration_dir=" in arg for arg in argv)
    assert not any(".right_arm_config.calibration_dir=" in arg for arg in argv)


def test_record() -> None:
    cameras = {"front": {"type": "opencv", "index": 0}}
    argv = ArmCommandBuilder().record(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "5B14032630",
        "so101_leader",
        "/dev/ttyACM1",
        "/cal/l",
        "5B14030892",
        cameras=cameras,
        repo_id="local/test_data",
        task="pick and place",
        dataset_root="/data",
        push_to_hub=False,
        fps=30,
        num_episodes=5,
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "record"]
    assert "--robot.type=so101_follower" in argv
    assert "--robot.id=5B14032630" in argv
    assert "--teleop.type=so101_leader" in argv
    assert "--teleop.id=5B14030892" in argv
    assert any("--robot.cameras=" in arg for arg in argv)
    assert "--dataset.repo_id=local/test_data" in argv
    assert "--dataset.root=/data" in argv
    assert "--dataset.push_to_hub=false" in argv
    assert "--dataset.single_task=pick and place" in argv
    assert "--dataset.fps=30" in argv
    assert "--dataset.num_episodes=5" in argv


def test_record_with_episode_time_s() -> None:
    argv = ArmCommandBuilder().record(
        "so101_follower", "/dev/ttyACM0", "/cal/f", "5B14032630",
        "so101_leader", "/dev/ttyACM1", "/cal/l", "5B14030892",
        cameras={}, repo_id="local/test", task="grasp",
        dataset_root="/data", episode_time_s=60,
    )
    assert "--dataset.episode_time_s=60" in argv


def test_record_omits_episode_time_s_when_none() -> None:
    argv = ArmCommandBuilder().record(
        "so101_follower", "/dev/ttyACM0", "/cal/f", "5B14032630",
        "so101_leader", "/dev/ttyACM1", "/cal/l", "5B14030892",
        cameras={}, repo_id="local/test", task="grasp",
        dataset_root="/data",
    )
    assert not any("episode_time_s" in arg for arg in argv)


def test_record_skips_empty_cameras() -> None:
    argv = ArmCommandBuilder().record(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "5B14032630",
        "so101_leader",
        "/dev/ttyACM1",
        "/cal/l",
        "5B14030892",
        cameras={},
        repo_id="local/test_data",
        task="pick and place",
        dataset_root="/data",
    )
    assert not any("--robot.cameras=" in arg for arg in argv)


def test_record_bimanual_uses_per_arm_cameras() -> None:
    cameras = {"front": {"type": "opencv", "index": 0}}
    argv = ArmCommandBuilder().record_bimanual(
        robot_id="bimanual",
        robot_cal_dir="/cal/robot",
        left_robot={"port": "/dev/a"},
        right_robot={"port": "/dev/b"},
        teleop_id="bimanual",
        teleop_cal_dir="/cal/teleop",
        left_teleop={"port": "/dev/c"},
        right_teleop={"port": "/dev/d"},
        cameras=cameras,
        repo_id="local/test_data",
        task="pick and place",
        dataset_root="/data",
        push_to_hub=False,
        fps=30,
        num_episodes=5,
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "record"]
    assert "--dataset.root=/data" in argv
    assert "--dataset.push_to_hub=false" in argv
    assert not any(arg.startswith("--robot.cameras=") for arg in argv)
    assert any(arg.startswith("--robot.left_arm_config.cameras=") for arg in argv)
    assert not any(arg.startswith("--robot.right_arm_config.cameras=") for arg in argv)


def test_replay() -> None:
    argv = ArmCommandBuilder().replay(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "5B14032630",
        repo_id="local/test_data",
        dataset_root="/data",
        episode=3,
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "replay"]
    assert "--robot.type=so101_follower" in argv
    assert "--robot.id=5B14032630" in argv
    assert "--dataset.repo_id=local/test_data" in argv
    assert "--dataset.root=/data" in argv
    assert "--dataset.episode=3" in argv


def test_replay_bimanual() -> None:
    argv = ArmCommandBuilder().replay_bimanual(
        robot_id="bimanual",
        robot_cal_dir="/cal/robot",
        left_robot={"port": "/dev/a"},
        right_robot={"port": "/dev/b"},
        repo_id="local/test_data",
        dataset_root="/data",
        episode=1,
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "replay"]
    assert "--robot.id=bimanual" in argv
    assert "--robot.calibration_dir=/cal/robot" in argv
    assert "--robot.left_arm_config.port=/dev/a" in argv
    assert "--robot.right_arm_config.port=/dev/b" in argv
    assert "--dataset.root=/data" in argv
    assert "--dataset.episode=1" in argv


def test_run_policy() -> None:
    cameras = {"front": {"type": "opencv", "index": 0}}
    argv = ArmCommandBuilder().run_policy(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "5B14032630",
        cameras=cameras,
        policy_path="/models/act_checkpoint",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "record"]
    assert "--robot.type=so101_follower" in argv
    assert "--robot.id=5B14032630" in argv
    assert any("--policy.path=" in arg for arg in argv)
    assert any("--robot.cameras=" in arg for arg in argv)
    assert not any("--teleop" in arg for arg in argv)


def test_run_policy_bimanual() -> None:
    cameras = {"front": {"type": "opencv", "index": 0}}
    argv = ArmCommandBuilder().run_policy_bimanual(
        robot_id="bimanual",
        robot_cal_dir="/cal/robot",
        left_robot={"port": "/dev/a"},
        right_robot={"port": "/dev/b"},
        cameras=cameras,
        policy_path="/models/act_checkpoint",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "record"]
    assert "--robot.type=bi_so_follower" in argv
    assert "--robot.id=bimanual" in argv
    assert "--robot.calibration_dir=/cal/robot" in argv
    assert "--robot.left_arm_config.port=/dev/a" in argv
    assert "--robot.right_arm_config.port=/dev/b" in argv
    assert any("--policy.path=" in arg for arg in argv)
    assert any("--robot.left_arm_config.cameras=" in arg for arg in argv)
    assert not any("--teleop" in arg for arg in argv)


def test_run_policy_skips_empty_cameras() -> None:
    argv = ArmCommandBuilder().run_policy(
        "so101_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "5B14032630",
        cameras={},
        policy_path="/models/act_checkpoint",
    )
    assert not any("--robot.cameras=" in arg for arg in argv)


def test_train() -> None:
    argv = ACTPipeline().train(
        repo_id="local/test_data",
        dataset_root="/data",
        output_dir="/output",
        steps=50000,
        device="cuda",
    )
    assert "lerobot-train" in argv
    assert "--dataset.repo_id=local/test_data" in argv
    assert "--dataset.root=/data" in argv
    assert "--policy.type=act" in argv
    assert "--policy.push_to_hub=false" in argv
    assert "--policy.repo_id=local/test_data" in argv
    assert "--steps=50000" in argv
    assert "--policy.device=cuda" in argv


def test_checkpoint_path() -> None:
    path = ACTPipeline().checkpoint_path("/output")
    assert "checkpoints/last/pretrained_model" in path


# ── Koch arm tests ─────────────────────────────────────────────────


def test_calibrate_koch_follower() -> None:
    argv = ArmCommandBuilder().calibrate(
        "koch_follower",
        "/dev/ttyACM0",
        "/cal/follower",
        "DXL001",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "calibrate"]
    assert "--robot.type=koch_follower" in argv
    assert "--robot.id=DXL001" in argv
    assert "--robot.port=/dev/ttyACM0" in argv


def test_calibrate_koch_leader() -> None:
    argv = ArmCommandBuilder().calibrate(
        "koch_leader",
        "/dev/ttyACM1",
        "/cal/leader",
        "DXL002",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "calibrate"]
    assert "--teleop.type=koch_leader" in argv
    assert "--teleop.id=DXL002" in argv


def test_teleoperate_koch() -> None:
    argv = ArmCommandBuilder().teleoperate(
        "koch_follower",
        "/dev/ttyACM0",
        "/cal/f",
        "DXL001",
        "koch_leader",
        "/dev/ttyACM1",
        "/cal/l",
        "DXL002",
    )
    assert argv[:4] == [sys.executable, "-m", "roboclaw.embodied.lerobot_wrapper", "teleoperate"]
    assert "--robot.type=koch_follower" in argv
    assert "--teleop.type=koch_leader" in argv
