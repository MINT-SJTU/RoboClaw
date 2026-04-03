"""Session sub-service: teleop and recording lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from roboclaw.embodied.engine import OperationEngine
from roboclaw.embodied.events import EventBus, SessionStateChangedEvent

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


class SessionService:
    """Manages teleop/recording operations via OperationEngine.

    Acquires/releases the embodiment lock automatically and coordinates
    with HardwareMonitor for recording_active lifecycle.
    """

    def __init__(
        self,
        parent: EmbodiedService,
        event_bus: EventBus,
    ) -> None:
        self._parent = parent
        self._engine = OperationEngine(on_state_change=self._on_engine_state_change)
        self._event_bus = event_bus
        self._recording_started = False

    @property
    def busy(self) -> bool:
        return self._engine.busy

    @property
    def state(self) -> str:
        return self._engine.state

    def get_status(self) -> dict[str, Any]:
        return self._engine.get_status()

    async def start_teleop(self, *, fps: int = 30) -> None:
        manifest = self._parent.manifest.snapshot
        await self._engine.start_teleop(fps=fps, setup=manifest)

    async def start_recording(
        self,
        task: str,
        num_episodes: int = 10,
        fps: int = 30,
        episode_time_s: int = 300,
        reset_time_s: int = 10,
    ) -> str:
        """Start recording. Returns dataset_name."""
        manifest = self._parent.manifest.snapshot
        dataset_name = await self._engine.start_recording(
            task=task,
            num_episodes=num_episodes,
            fps=fps,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            setup=manifest,
        )
        self._recording_started = True
        monitor = self._parent._monitor
        if monitor is not None:
            monitor.set_recording_active(True)
        return dataset_name

    async def stop(self) -> None:
        await self._engine.stop()

    async def save_episode(self) -> None:
        await self._engine.save_episode()

    async def discard_episode(self) -> None:
        await self._engine.discard_episode()

    async def skip_reset(self) -> None:
        await self._engine.skip_reset()

    # -- Internal: state change routing ---------------------------------------

    async def _on_engine_state_change(self, status: dict[str, Any]) -> None:
        """Called by OperationEngine on every state transition."""
        new_state = status.get("state", "idle")
        if new_state == "idle" and self._recording_started:
            self._recording_started = False
            monitor = self._parent._monitor
            if monitor is not None:
                monitor.set_recording_active(False)

        await self._event_bus.emit(SessionStateChangedEvent(**status))
