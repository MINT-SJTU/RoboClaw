from __future__ import annotations

import pytest

from roboclaw.data import dataset_sessions


def test_session_handle_rejects_path_segments() -> None:
    handle = "session:remote:../../outside"

    assert dataset_sessions.parse_session_handle(handle) is None
    with pytest.raises(ValueError, match="Invalid dataset session handle"):
        dataset_sessions.resolve_session_dataset_path(handle)


def test_uploaded_directory_session_rejects_path_escape(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dataset_sessions, "get_roboclaw_home", lambda: tmp_path)

    with pytest.raises(ValueError, match="Invalid uploaded file path"):
        dataset_sessions.create_uploaded_directory_session(files=[("../escape.txt", b"x")])

    assert not (tmp_path / "cache" / "escape.txt").exists()
