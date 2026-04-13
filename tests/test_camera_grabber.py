"""Tests for camera frame grabbing helpers."""

from __future__ import annotations

import numpy as np

from roboclaw.embodied.perception.camera_grabber import grab_frame


class _FakeCapture:
    def __init__(self, frame: np.ndarray | None = None) -> None:
        self._frame = frame if frame is not None else np.zeros((2, 2, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def read(self):
        return True, self._frame

    def release(self) -> None:
        pass


def test_grab_frame_uses_numeric_camera_index(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(
        "roboclaw.embodied.perception.camera_grabber._default_camera_configs",
        lambda: {"overhead": {"port": "2"}},
    )

    def _video_capture(arg, *args):
        calls.append(arg)
        return _FakeCapture()

    monkeypatch.setattr("roboclaw.embodied.perception.camera_grabber.cv2.VideoCapture", _video_capture)

    frame = grab_frame("overhead")

    assert frame is not None
    assert calls == [2]


def test_grab_frame_uses_path_for_non_numeric_source(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(
        "roboclaw.embodied.perception.camera_grabber._default_camera_configs",
        lambda: {"wrist": {"port": "/dev/video0"}},
    )

    def _video_capture(arg, *args):
        calls.append(arg)
        return _FakeCapture()

    monkeypatch.setattr("roboclaw.embodied.perception.camera_grabber.cv2.VideoCapture", _video_capture)

    frame = grab_frame("wrist")

    assert frame is not None
    assert calls == ["/dev/video0"]
