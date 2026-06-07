import json
from pathlib import Path

from typer.testing import CliRunner

from roboclaw.cli.commands import app

runner = CliRunner()


def _write_demo_dataset(path: Path) -> None:
    meta = path / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(
        json.dumps(
            {
                "total_episodes": 2,
                "total_frames": 20,
                "fps": 10,
                "robot_type": "so101",
            }
        ),
        encoding="utf-8",
    )
    (meta / "episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": idx, "length": 10}) for idx in range(2)),
        encoding="utf-8",
    )
    data = path / "data"
    data.mkdir()
    (data / "sample.txt").write_text("demo", encoding="utf-8")


def test_dataset_push_requires_username(tmp_path):
    dataset_path = tmp_path / "red-cube"
    _write_demo_dataset(dataset_path)

    result = runner.invoke(
        app,
        ["dataset", "push", str(dataset_path), "--dry-run"],
        env={"ROBOCLAW_USERNAME": "", "EVOMIND_USER": "", "ROBOCLAW_SERVER_URL": ""},
    )

    assert result.exit_code != 0
    assert "username is required" in result.output
    assert "pearl" not in result.output


def test_dataset_push_dry_run_scans_without_registering(tmp_path):
    dataset_path = tmp_path / "red-cube"
    _write_demo_dataset(dataset_path)
    home = tmp_path / "home"

    result = runner.invoke(
        app,
        ["dataset", "push", str(dataset_path), "--username", "alice", "--dry-run"],
        env={"ROBOCLAW_HOME": str(home), "ROBOCLAW_SERVER_URL": ""},
    )

    assert result.exit_code == 0
    assert "Dataset push plan" in result.output
    assert "episodes: 2" in result.output
    assert not (home / "workspace" / "embodied" / "datasets" / "red-cube").exists()


def test_dataset_push_registers_local_catalog_dataset(tmp_path):
    dataset_path = tmp_path / "red-cube"
    _write_demo_dataset(dataset_path)
    home = tmp_path / "home"

    result = runner.invoke(
        app,
        ["dataset", "push", str(dataset_path), "--username", "alice"],
        env={"ROBOCLAW_HOME": str(home), "ROBOCLAW_SERVER_URL": ""},
    )

    assert result.exit_code == 0
    assert "Dataset registered in the local Evo Studio catalog" in result.output
    assert "evo://alice/red-cube" in result.output
    info_path = home / "workspace" / "embodied" / "datasets" / "red-cube" / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert info["ownerUsername"] == "alice"
    assert info["contributionSource"] == "self_collected"
    assert info["uploadStatus"] == "uploaded"
