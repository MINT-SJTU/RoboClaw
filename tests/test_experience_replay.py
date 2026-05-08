from __future__ import annotations

from pathlib import Path

import pytest

from roboclaw.agent.experience import ExperienceRecord, ExperienceStore


def test_experience_store_records_and_deduplicates(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    record = ExperienceRecord(
        timestamp="2026-05-07T10:00:00+00:00",
        task_type="train",
        summary="demo -> success",
        outcome="success",
        dataset="demo",
        policy="act",
        provider="local",
    )
    assert store.append(record) is True
    assert store.append(record) is False
    assert len(store.read_all()) == 1


def test_experience_store_search_filters_by_outcome(tmp_path: Path) -> None:
    store = ExperienceStore(tmp_path)
    store.append(ExperienceRecord(
        timestamp="2026-05-07T10:00:00+00:00",
        task_type="train",
        summary="demo -> submitted",
        outcome="submitted",
        dataset="demo",
        policy="act",
        provider="local",
    ))
    store.append(ExperienceRecord(
        timestamp="2026-05-07T11:00:00+00:00",
        task_type="train",
        summary="demo -> success",
        outcome="success",
        dataset="demo",
        policy="act",
        provider="local",
    ))

    results = store.search(
        task_type="train",
        dataset="demo",
        policy="act",
        outcomes=frozenset({"success", "failed", "error", "stopped"}),
        provider="local",
    )
    assert all(r.outcome != "submitted" for r in results)
    assert any(r.outcome == "success" for r in results)
