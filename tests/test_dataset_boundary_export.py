from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import torch

from roboclaw.data.repair.boundary import (
    BoundaryFrameExporter,
    BoundaryFrameExportRequest,
    export_episode_boundary_frames,
    parse_episode_indices,
    resolve_repo_id,
    select_camera_key,
)


class _FakeDataset:
    def __init__(self) -> None:
        self.root = Path("/tmp/demo")
        self.repo_id = "local/demo"
        self.meta = SimpleNamespace(
            episodes=pa.table(
                {
                    "dataset_from_index": [0, 2],
                    "dataset_to_index": [2, 4],
                    "length": [2, 2],
                    "episode_success": ["success", "failure"],
                }
            )
        )
        self._frames = [
            {"observation.images.right_front": torch.zeros(3, 4, 4, dtype=torch.uint8)},
            {"observation.images.right_front": torch.ones(3, 4, 4, dtype=torch.uint8)},
            {"observation.images.right_front": torch.full((3, 4, 4), 2, dtype=torch.uint8)},
            {"observation.images.right_front": torch.full((3, 4, 4), 3, dtype=torch.uint8)},
        ]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._frames[index]


class _FakeAdapter:
    def open_dataset(self, *, repo_id: str, root: Path) -> _FakeDataset:
        dataset = _FakeDataset()
        dataset.repo_id = repo_id
        dataset.root = root
        return dataset


class TestBoundaryHelpers:
    def test_parse_episode_indices_supports_ranges_and_dedup(self) -> None:
        assert parse_episode_indices("0-1,1,3", 5) == [0, 1, 3]

    def test_select_camera_key_prefers_front_then_largest(self) -> None:
        info = {
            "features": {
                "observation.images.left_wrist": {"dtype": "video", "shape": [32, 32, 3], "names": None},
                "observation.images.right_front": {"dtype": "video", "shape": [16, 64, 3], "names": None},
            }
        }
        assert select_camera_key(info, None) == "observation.images.right_front"

    def test_resolve_repo_id_returns_relative_id_under_dataset_root(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "local" / "demo"
        dataset_root.mkdir(parents=True)
        assert resolve_repo_id(dataset_root, tmp_path) == "local/demo"


class TestBoundaryExport:
    def test_export_episode_boundary_frames_writes_manifest(self, tmp_path: Path) -> None:
        manifest_path = export_episode_boundary_frames(
            dataset=_FakeDataset(),
            output_dir=tmp_path,
            episode_indices=[0, 1],
            camera_key="observation.images.right_front",
        )

        rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
        assert len(rows) == 2
        assert rows[0]["episode_success"] == "success"
        assert (tmp_path / "episode_000_first.png").exists()
        assert (tmp_path / "episode_001_last.png").exists()

    def test_exporter_opens_dataset_and_exports(self, tmp_path: Path) -> None:
        dataset_root = tmp_path / "datasets" / "local" / "demo"
        meta_dir = dataset_root / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "info.json").write_text(
            json.dumps(
                {
                    "total_episodes": 2,
                    "features": {
                        "observation.images.right_front": {
                            "dtype": "video",
                            "shape": [4, 4, 3],
                            "names": None,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        exporter = BoundaryFrameExporter(adapter=_FakeAdapter())
        result = exporter.export(
            BoundaryFrameExportRequest(
                dataset=str(dataset_root),
                output_dir=tmp_path / "out",
                episodes="0-1",
                overwrite=False,
            )
        )

        assert result.episodes_exported == 2
        assert result.camera_key == "observation.images.right_front"
        assert result.manifest_path.exists()
