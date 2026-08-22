from __future__ import annotations

from pathlib import Path
from typing import Any


class LeRobotDatasetAdapter:
    def create_dataset(
        self,
        *,
        repo_id: str,
        fps: int,
        root: Path,
        robot_type: str | None,
        features: dict[str, Any],
        use_videos: bool,
        vcodec: str,
    ) -> Any:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
            vcodec=vcodec,
        )

    def open_dataset(self, *, repo_id: str, root: Path) -> Any:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset(repo_id=repo_id, root=root)

    def encode_video_frames(self, *, frames_dir: Path, video_path: Path, fps: int, vcodec: str) -> None:
        from lerobot.datasets.video_utils import encode_video_frames

        encode_video_frames(frames_dir, video_path, fps, vcodec=vcodec, overwrite=True)
