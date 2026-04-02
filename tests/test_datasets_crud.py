"""Tests for dataset listing, info retrieval, and deletion."""

from __future__ import annotations

import json

import pytest

from roboclaw.web.dashboard_datasets import (
    delete_dataset,
    get_dataset_info,
    list_datasets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_dataset(root, name: str, *, total_episodes: int = 5,
                    total_frames: int = 500, fps: int = 30,
                    robot_type: str = "so100", features: dict | None = None,
                    episode_lengths: list[int] | None = None) -> None:
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
        lines = [json.dumps({"episode_index": i, "length": l})
                 for i, l in enumerate(episode_lengths)]
        (meta_dir / "episodes.jsonl").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# list_datasets
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_empty_root(self, tmp_path):
        assert list_datasets(tmp_path) == []

    def test_nonexistent_root(self, tmp_path):
        assert list_datasets(tmp_path / "nope") == []

    def test_single_dataset(self, tmp_path):
        _create_dataset(tmp_path, "pick_cup")
        result = list_datasets(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "pick_cup"
        assert result[0]["total_episodes"] == 5

    def test_multiple_datasets_sorted(self, tmp_path):
        _create_dataset(tmp_path, "b_dataset")
        _create_dataset(tmp_path, "a_dataset")
        result = list_datasets(tmp_path)
        names = [d["name"] for d in result]
        assert names == ["a_dataset", "b_dataset"]

    def test_nested_datasets(self, tmp_path):
        """Datasets under root/local/dataset_name/ are discovered."""
        local = tmp_path / "local"
        local.mkdir()
        _create_dataset(local, "nested_ds")
        result = list_datasets(tmp_path)
        assert len(result) == 1
        assert result[0]["name"] == "nested_ds"

    def test_ignores_dirs_without_info_json(self, tmp_path):
        (tmp_path / "not_a_dataset").mkdir()
        assert list_datasets(tmp_path) == []

    def test_ignores_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hi")
        assert list_datasets(tmp_path) == []

    def test_episode_lengths_parsed(self, tmp_path):
        _create_dataset(tmp_path, "with_eps", episode_lengths=[100, 150, 200])
        result = list_datasets(tmp_path)
        assert result[0]["episode_lengths"] == [100, 150, 200]

    def test_features_keys_returned(self, tmp_path):
        _create_dataset(tmp_path, "feat_ds",
                        features={"observation.image": {}, "action": {}, "next.reward": {}})
        result = list_datasets(tmp_path)
        assert set(result[0]["features"]) == {"observation.image", "action", "next.reward"}


# ---------------------------------------------------------------------------
# get_dataset_info
# ---------------------------------------------------------------------------

class TestGetDatasetInfo:
    def test_found(self, tmp_path):
        _create_dataset(tmp_path, "my_ds", total_episodes=3, fps=15)
        info = get_dataset_info(tmp_path, "my_ds")
        assert info is not None
        assert info["total_episodes"] == 3
        assert info["fps"] == 15

    def test_not_found(self, tmp_path):
        assert get_dataset_info(tmp_path, "no_such") is None

    def test_dir_without_meta(self, tmp_path):
        (tmp_path / "empty_dir").mkdir()
        assert get_dataset_info(tmp_path, "empty_dir") is None


# ---------------------------------------------------------------------------
# delete_dataset
# ---------------------------------------------------------------------------

class TestDeleteDataset:
    def test_delete_existing(self, tmp_path):
        _create_dataset(tmp_path, "to_delete")
        delete_dataset(tmp_path, "to_delete")
        assert not (tmp_path / "to_delete").exists()

    def test_delete_nonexistent_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            delete_dataset(tmp_path, "nope")
