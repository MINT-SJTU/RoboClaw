from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from threading import Event
from typing import Any

from roboclaw.embodied.board import Board, SessionState
from roboclaw.embodied.calibration.model import CalibrationBatchResult, CalibrationProfile
from roboclaw.embodied.calibration.so101 import AutoCalibrationStopped, SO101AutoCalibrationStrategy
from roboclaw.embodied.calibration.store import CalibrationStore
from roboclaw.embodied.embodiment.arm.registry import get_model
from roboclaw.embodied.embodiment.manifest import Manifest
from roboclaw.embodied.embodiment.manifest.binding import ArmBinding


@dataclass(frozen=True)
class _BatchItem:
    arm: ArmBinding
    result: CalibrationBatchResult
    eligible: bool


class AutoCalibrationBatch:
    def __init__(
        self,
        *,
        board: Board,
        manifest: Manifest,
        store: CalibrationStore | None = None,
        strategy: SO101AutoCalibrationStrategy | None = None,
    ) -> None:
        self.board = board
        self._manifest = manifest
        self._store = store or CalibrationStore()
        self._strategy = strategy or SO101AutoCalibrationStrategy()
        self._task: asyncio.Task[None] | None = None
        self._stop_event = Event()
        self._exit_callback: Any = None

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, arms: list[ArmBinding]) -> int:
        items = self._plan(arms)
        owner = self.board.get("embodiment_owner", "")
        self._stop_event.clear()
        self.board.reset()
        await self.board.update(
            state=SessionState.CALIBRATING,
            embodiment_owner=owner,
            calibration_mode="auto",
            calibration_scope="batch",
            calibration_phase="planning",
            calibration_current_arm="",
            calibration_index=0,
            calibration_total=len(items),
            calibration_results=[item.result.to_dict() for item in items],
            calibration_error="",
            error="",
        )
        self.board.start_timer()
        self._task = asyncio.create_task(self._run(items), name="auto-calibration-batch")
        return len(items)

    async def stop(self) -> None:
        if not self.busy:
            await self.board.update(state=SessionState.IDLE, calibration_phase="")
            return
        self._stop_event.set()
        await self.board.update(state=SessionState.STOPPING, calibration_phase="stopping")
        if self._task is not None:
            await asyncio.shield(self._task)

    def _plan(self, arms: list[ArmBinding]) -> list[_BatchItem]:
        items: list[_BatchItem] = []
        for arm in arms:
            if get_model(arm.arm_type) != "so101":
                items.append(_BatchItem(
                    arm=arm,
                    result=CalibrationBatchResult(alias=arm.alias, status="skipped", reason="unsupported_arm_type"),
                    eligible=False,
                ))
                continue
            if not arm.connected:
                items.append(_BatchItem(
                    arm=arm,
                    result=CalibrationBatchResult(alias=arm.alias, status="skipped", reason="disconnected"),
                    eligible=False,
                ))
                continue
            items.append(_BatchItem(
                arm=arm,
                result=CalibrationBatchResult(alias=arm.alias, status="pending"),
                eligible=True,
            ))
        return items

    async def _run(self, items: list[_BatchItem]) -> None:
        try:
            if not items:
                await self.board.update(
                    state=SessionState.IDLE,
                    calibration_phase="done",
                    calibration_results=[],
                    calibration_total=0,
                )
                return

            active_tasks: dict[asyncio.Task[CalibrationProfile], int] = {}
            for position, item in enumerate(items):
                if not item.eligible:
                    continue
                started = time.time()
                items[position] = replace(
                    item,
                    result=replace(item.result, status="running", reason="", started_at=started, finished_at=None),
                )
                active_tasks[asyncio.create_task(
                    asyncio.to_thread(
                        self._strategy.recalibrate,
                        item.arm,
                        self._store,
                        stop_event=self._stop_event,
                    ),
                    name=f"auto-calibration:{item.arm.alias}",
                )] = position

            await self._publish(
                items,
                index=self._completed_count(items),
                phase="probing" if active_tasks else "done",
                current_arm=self._running_aliases(items),
            )

            while active_tasks:
                done, _ = await asyncio.wait(active_tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    position = active_tasks.pop(task)
                    item = items[position]
                    finished = time.time()
                    try:
                        profile = task.result()
                    except AutoCalibrationStopped:
                        items[position] = replace(
                            item,
                            result=replace(item.result, status="failed", reason="stopped", finished_at=finished),
                        )
                        continue
                    except Exception as exc:
                        items[position] = replace(
                            item,
                            result=replace(item.result, status="failed", reason=str(exc), finished_at=finished),
                        )
                        continue

                    await self.board.update(
                        calibration_phase="persisting",
                        calibration_current_arm=item.arm.alias,
                        calibration_index=self._completed_count(items),
                    )
                    try:
                        self._store.save_profile(item.arm, profile)
                        self._manifest.mark_arm_calibrated(item.arm.alias)
                    except Exception as exc:
                        items[position] = replace(
                            item,
                            result=replace(item.result, status="failed", reason=str(exc), finished_at=finished),
                        )
                    else:
                        items[position] = replace(
                            item,
                            result=replace(item.result, status="success", finished_at=finished),
                        )

                await self._publish(
                    items,
                    index=self._completed_count(items),
                    phase="stopped" if self._stop_event.is_set() and active_tasks else "probing",
                    current_arm=self._running_aliases(items),
                )

            final_phase = "stopped" if self._stop_event.is_set() else "done"
            await self.board.update(
                state=SessionState.IDLE,
                calibration_phase=final_phase,
                calibration_current_arm="",
                calibration_index=self._completed_count(items),
                calibration_results=[item.result.to_dict() for item in items],
            )
        except Exception as exc:
            await self.board.update(
                state=SessionState.ERROR,
                calibration_phase="failed",
                calibration_error=str(exc),
                error=str(exc),
            )
        finally:
            if self._exit_callback and not self._stop_event.is_set():
                self._exit_callback(self)

    def _completed_count(self, items: list[_BatchItem]) -> int:
        return sum(1 for item in items if item.result.status in {"success", "skipped", "failed"})

    def _running_aliases(self, items: list[_BatchItem]) -> str:
        running = [item.arm.alias for item in items if item.result.status == "running"]
        return ", ".join(running)

    async def _publish(
        self,
        items: list[_BatchItem],
        *,
        index: int,
        phase: str = "planning",
        current_arm: str = "",
    ) -> None:
        await self.board.update(
            calibration_phase=phase,
            calibration_current_arm=current_arm,
            calibration_index=index,
            calibration_results=[item.result.to_dict() for item in items],
        )
