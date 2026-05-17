from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from roboclaw.account import AccountLedger
from roboclaw.data.curation import service as curation_service
from roboclaw.data.curation.service import CurationService
from roboclaw.data.curation.state import load_workflow_state, save_quality_results, save_workflow_state


def _write_completed_quality_results(dataset_path: Path, *, passed: int, failed: int) -> None:
    episodes = [
        {"episode_index": index, "passed": True}
        for index in range(passed)
    ] + [
        {"episode_index": passed + index, "passed": False}
        for index in range(failed)
    ]
    save_quality_results(
        dataset_path,
        {
            "passed": passed,
            "failed": failed,
            "overall_score": 1.0 if failed == 0 else 0.5,
            "episodes": episodes,
        },
    )
    state = load_workflow_state(dataset_path)
    quality_stage = state["stages"]["quality_validation"]
    quality_stage["status"] = "completed"
    quality_stage["summary"] = {"passed": passed, "failed": failed}
    save_workflow_state(dataset_path, state)


async def _run_quality_binding(
    monkeypatch,
    tmp_path: Path,
    *,
    username: str,
    dataset_id: str = "dataset-1",
    passed: int,
    failed: int = 0,
    ledger: AccountLedger | None = None,
) -> AccountLedger:
    dataset_path = tmp_path / "datasets" / dataset_id
    dataset_path.mkdir(parents=True, exist_ok=True)
    ledger = ledger or AccountLedger(tmp_path / "ledger.json")
    scheduled: list[Any] = []

    def fake_register_workflow_task(
        self: CurationService,
        _dataset_path: Path,
        _stage_key: str,
        coro: Any,
    ) -> None:
        scheduled.append(coro)

    def fake_run_quality_batch(self: Any, *_args: Any, **_kwargs: Any) -> None:
        _write_completed_quality_results(self.dataset_path, passed=passed, failed=failed)

    monkeypatch.setattr(CurationService, "_register_workflow_task", fake_register_workflow_task)
    monkeypatch.setattr(
        curation_service._LegacyCurationService,
        "run_quality_batch",
        fake_run_quality_batch,
    )
    monkeypatch.setattr(curation_service, "AccountLedger", lambda: ledger)

    service = CurationService()
    response = await service.start_quality_run(
        dataset_path,
        dataset_id,
        ["length"],
        None,
        None,
        username,
    )
    assert response == {"status": "started"}
    assert len(scheduled) == 1
    await scheduled[0]
    return ledger


def test_quality_run_grants_points_for_passed_episodes(tmp_path, monkeypatch) -> None:
    ledger = asyncio.run(
        _run_quality_binding(monkeypatch, tmp_path, username="pearl", passed=3),
    )

    wallet = ledger.wallet("pearl")
    records = ledger.records("pearl")

    assert wallet.reward_points == 30
    assert len(records) == 1
    assert records[0].kind == "dataset_reward"
    assert records[0].amount_cents == 30
    assert records[0].job_id == "dataset-1"


def test_quality_run_does_not_grant_points_when_all_failed(tmp_path, monkeypatch) -> None:
    ledger = asyncio.run(
        _run_quality_binding(monkeypatch, tmp_path, username="pearl", passed=0, failed=3),
    )

    assert ledger.wallet("pearl").reward_points == 0
    assert ledger.records("pearl") == []


def test_quality_run_does_not_grant_points_without_username(tmp_path, monkeypatch) -> None:
    ledger = asyncio.run(
        _run_quality_binding(monkeypatch, tmp_path, username="", passed=3),
    )

    assert ledger.records() == []


def test_quality_run_reward_is_idempotent_by_dataset_id(tmp_path, monkeypatch) -> None:
    ledger = AccountLedger(tmp_path / "ledger.json")
    asyncio.run(
        _run_quality_binding(
            monkeypatch,
            tmp_path,
            username="pearl",
            passed=3,
            ledger=ledger,
        ),
    )
    asyncio.run(
        _run_quality_binding(
            monkeypatch,
            tmp_path,
            username="pearl",
            passed=3,
            ledger=ledger,
        ),
    )

    assert ledger.wallet("pearl").reward_points == 30
    records = ledger.records("pearl")
    assert len(records) == 1
    assert records[0].job_id == "dataset-1"
