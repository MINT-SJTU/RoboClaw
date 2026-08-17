"""Tests for DatasetCatalog local discovery, detail resolution, and deletion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboclaw.data.datasets import DatasetCatalog


def _create_dataset(
    root,
    name: str,
    *,
    total_episodes: int = 5,
    total_frames: int = 500,
    fps: int = 30,
    robot_type: str = "so100",
    features: dict | None = None,
    episode_lengths: list[int] | None = None,
) -> None:
    """Create a minimal LeRobot dataset directory under *root*."""
    ds_dir = root / name
    meta_dir = ds_dir / "meta"
    meta_dir.mkdir(parents=True)

    info = {
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "fps": fps,
        "robot_type": robot_type,
        "features": features or {"observation.state": {}, "action": {}},
    }
    (meta_dir / "info.json").write_text(json.dumps(info), encoding="utf-8")

    if episode_lengths:
        lines = [json.dumps({"episode_index": i, "length": l}) for i, l in enumerate(episode_lengths)]
        (meta_dir / "episodes.jsonl").write_text("\n".join(lines), encoding="utf-8")


class TestDatasetCatalogLocalListing:
    def test_empty_root(self, tmp_path):
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)
        assert catalog.list_local_datasets() == []

    def test_nonexistent_root(self, tmp_path):
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path / "nope")
        assert catalog.list_local_datasets() == []

    def test_single_dataset(self, tmp_path):
        _create_dataset(tmp_path, "pick_cup")
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        result = [item.to_dict() for item in catalog.list_local_datasets()]

        assert len(result) == 1
        assert result[0]["id"] == "pick_cup"
        assert result[0]["kind"] == "local"
        assert result[0]["label"] == "pick_cup"
        assert result[0]["slug"] == "pick_cup"
        assert result[0]["runtime"] is None
        assert result[0]["stats"]["total_episodes"] == 5

    def test_multiple_datasets_sorted(self, tmp_path):
        _create_dataset(tmp_path, "b_dataset")
        _create_dataset(tmp_path, "a_dataset")
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        result = catalog.list_local_datasets()

        assert [dataset.id for dataset in result] == ["a_dataset", "b_dataset"]

    def test_nested_runtime_dataset(self, tmp_path):
        local = tmp_path / "local"
        local.mkdir()
        _create_dataset(local, "nested_ds")
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        result = [item.to_dict() for item in catalog.list_local_datasets()]

        assert len(result) == 1
        assert result[0]["id"] == "local/nested_ds"
        assert result[0]["label"] == "nested_ds"
        assert result[0]["runtime"] == {
            "name": "nested_ds",
            "repo_id": "local/nested_ds",
            "local_path": str(tmp_path / "local" / "nested_ds"),
        }
        assert result[0]["capabilities"]["can_replay"] is True
        assert result[0]["capabilities"]["can_train"] is True

    def test_namespaced_dataset_has_no_runtime(self, tmp_path):
        namespace = tmp_path / "cadene"
        namespace.mkdir()
        _create_dataset(namespace, "droid_1.0.1")
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        result = [item.to_dict() for item in catalog.list_local_datasets()]

        assert len(result) == 1
        assert result[0]["id"] == "cadene/droid_1.0.1"
        assert result[0]["label"] == "cadene/droid_1.0.1"
        assert result[0]["runtime"] is None
        assert result[0]["capabilities"]["can_replay"] is False
        assert result[0]["capabilities"]["can_train"] is False
        assert result[0]["capabilities"]["can_curate"] is True

    def test_episode_lengths_parsed(self, tmp_path):
        _create_dataset(tmp_path, "with_eps", episode_lengths=[100, 150, 200])
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        result = [item.to_dict() for item in catalog.list_local_datasets()]

        assert result[0]["stats"]["episode_lengths"] == [100, 150, 200]

    def test_features_keys_returned(self, tmp_path):
        _create_dataset(
            tmp_path,
            "feat_ds",
            features={"observation.image": {}, "action": {}, "next.reward": {}},
        )
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        result = [item.to_dict() for item in catalog.list_local_datasets()]

        assert set(result[0]["stats"]["features"]) == {"observation.image", "action", "next.reward"}


class TestDatasetCatalogDetailAndDelete:
    def test_get_local_dataset(self, tmp_path):
        _create_dataset(tmp_path, "my_ds", total_episodes=3, fps=15)
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        info = catalog.require_local_dataset("my_ds").to_dict()

        assert info["id"] == "my_ds"
        assert info["label"] == "my_ds"
        assert info["stats"]["total_episodes"] == 3
        assert info["stats"]["fps"] == 15

    def test_get_missing_dataset_raises(self, tmp_path):
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)
        with pytest.raises(ValueError, match="not found"):
            catalog.require_local_dataset("no_such")

    def test_delete_existing(self, tmp_path):
        _create_dataset(tmp_path, "to_delete")
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        catalog.delete_dataset("to_delete")

        assert not (tmp_path / "to_delete").exists()

    def test_delete_nonexistent_raises(self, tmp_path):
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)
        with pytest.raises(ValueError, match="not found"):
            catalog.delete_dataset("nope")


class TestDatasetCatalogRemoteAndImport:
    def test_resolve_remote_dataset(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        monkeypatch.setattr(
            "roboclaw.data.explorer.remote.build_remote_dataset_info",
            lambda dataset: {
                "name": dataset,
                "total_episodes": 2,
                "total_frames": 20,
                "fps": 30,
                "episode_lengths": [8, 12],
                "features": ["action"],
                "robot_type": "aloha",
                "source_dataset": dataset,
            },
        )

        dataset = catalog.resolve_remote_dataset("cadene/droid_1.0.1").to_dict()

        assert dataset["id"] == "cadene/droid_1.0.1"
        assert dataset["kind"] == "remote"
        assert dataset["capabilities"]["can_pull"] is True
        assert dataset["runtime"] is None

    @pytest.mark.asyncio
    async def test_import_job_completion_materializes_dataset(
        self,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        catalog = DatasetCatalog(root_resolver=lambda: tmp_path)

        def _fake_snapshot_download(*, repo_id: str, local_dir: str, **_: object) -> str:
            target_dir = Path(local_dir)
            (target_dir / "meta").mkdir(parents=True, exist_ok=True)
            (target_dir / "meta" / "info.json").write_text(
                json.dumps({"total_episodes": 1, "total_frames": 2, "fps": 30}),
                encoding="utf-8",
            )
            return str(target_dir)

        monkeypatch.setattr("roboclaw.data.datasets.snapshot_download", _fake_snapshot_download, raising=False)
        monkeypatch.setattr("huggingface_hub.snapshot_download", _fake_snapshot_download)

        job = catalog.queue_import_job("job-1", dataset_id="cadene/droid_1.0.1", include_videos=False)
        assert job.status == "queued"

        await catalog.run_import_job(
            "job-1",
            "cadene/droid_1.0.1",
            include_videos=False,
            force=False,
        )
        result = catalog.get_import_job("job-1")

        assert result is not None
        assert result.status == "completed"
        assert result.dataset is not None
        assert result.dataset.id == "cadene/droid_1.0.1"
