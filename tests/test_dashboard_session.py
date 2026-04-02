"""Tests for DashboardSession state machine and stdout parsing."""

from __future__ import annotations

import pytest

from roboclaw.web.dashboard_session import DashboardSession


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestStateMachine:
    def test_initial_state_is_idle(self):
        session = DashboardSession()
        assert session.state == "idle"
        assert not session.busy

    def test_busy_when_not_idle(self):
        session = DashboardSession()
        session._state = "preparing"
        assert session.busy

    def test_busy_when_teleoperating(self):
        session = DashboardSession()
        session._state = "teleoperating"
        assert session.busy

    def test_busy_when_recording(self):
        session = DashboardSession()
        session._state = "recording"
        assert session.busy

    def test_require_idle_raises_when_busy(self):
        session = DashboardSession()
        session._state = "recording"
        with pytest.raises(RuntimeError, match="Session busy"):
            session._require_idle_or_raise()

    def test_require_idle_passes_when_idle(self):
        session = DashboardSession()
        session._require_idle_or_raise()  # should not raise


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_idle_status(self):
        session = DashboardSession()
        status = session.get_status()
        assert status["state"] == "idle"
        assert status["dataset"] is None

    def test_recording_status(self):
        session = DashboardSession()
        session._state = "recording"
        session._dataset_name = "test_ds"
        session._saved_episodes = 3
        session._target_episodes = 10
        session._episode_phase = "recording"

        status = session.get_status()
        assert status["state"] == "recording"
        assert status["dataset"] == "test_ds"
        assert status["saved_episodes"] == 3
        assert status["target_episodes"] == 10

    def test_non_recording_hides_dataset(self):
        session = DashboardSession()
        session._state = "teleoperating"
        session._dataset_name = "leftover"
        assert session.get_status()["dataset"] is None


# ---------------------------------------------------------------------------
# _parse_line — episode lifecycle tracking
# ---------------------------------------------------------------------------

class TestParseLine:
    def _session(self) -> DashboardSession:
        s = DashboardSession()
        s._state = "recording"
        return s

    def test_recording_episode_sets_phase(self):
        s = self._session()
        s._parse_line("[lerobot] Recording episode 0")
        assert s._episode_phase == "recording"

    def test_right_arrow_during_recording_sets_saving(self):
        s = self._session()
        s._episode_phase = "recording"
        s._parse_line("Right arrow key pressed. Saving episode.")
        assert s._episode_phase == "saving"

    def test_right_arrow_during_resetting_sets_saving(self):
        s = self._session()
        s._episode_phase = "resetting"
        s._parse_line("Right arrow key pressed.")
        assert s._episode_phase == "saving"

    def test_reset_environment_sets_resetting(self):
        s = self._session()
        s._episode_phase = "recording"
        s._parse_line("[lerobot] Reset the environment")
        assert s._episode_phase == "resetting"

    def test_re_record_sets_recording(self):
        s = self._session()
        s._episode_phase = "saving"
        s._parse_line("[lerobot] Re-record episode")
        assert s._episode_phase == "recording"

    def test_stop_recording_increments_if_saving(self):
        s = self._session()
        s._episode_phase = "saving"
        s._saved_episodes = 2
        s._parse_line("[lerobot] Stop recording")
        assert s._saved_episodes == 3
        assert s._episode_phase == ""

    def test_stop_recording_increments_if_resetting(self):
        s = self._session()
        s._episode_phase = "resetting"
        s._saved_episodes = 0
        s._parse_line("[lerobot] Stop recording")
        assert s._saved_episodes == 1

    def test_stop_recording_no_increment_if_empty(self):
        s = self._session()
        s._episode_phase = ""
        s._saved_episodes = 5
        s._parse_line("[lerobot] Stop recording")
        assert s._saved_episodes == 5

    def test_new_episode_increments_if_previous_saving(self):
        s = self._session()
        s._episode_phase = "saving"
        s._saved_episodes = 1
        s._parse_line("[lerobot] Recording episode 2")
        assert s._saved_episodes == 2
        assert s._episode_phase == "recording"

    def test_new_episode_no_increment_if_first(self):
        s = self._session()
        s._episode_phase = ""
        s._saved_episodes = 0
        s._parse_line("[lerobot] Recording episode 0")
        assert s._saved_episodes == 0
        assert s._episode_phase == "recording"

    def test_frame_count_parsed(self):
        s = self._session()
        s._parse_line("frames: 150")
        assert s._total_frames == 150

    def test_episode_done_parsed(self):
        s = self._session()
        s._parse_line("Episode 2 done")
        # No crash, no state change for this particular line

    def test_unrecognized_line_is_ignored(self):
        s = self._session()
        s._episode_phase = "recording"
        s._parse_line("some random log output")
        assert s._episode_phase == "recording"

    def test_full_lifecycle(self):
        """Simulate: record ep0 → save → reset → record ep1 → save → stop."""
        s = self._session()
        s._target_episodes = 2

        s._parse_line("[lerobot] Recording episode 0")
        assert s._episode_phase == "recording"
        assert s._saved_episodes == 0

        s._parse_line("Right arrow key pressed.")
        assert s._episode_phase == "saving"

        s._parse_line("[lerobot] Reset the environment")
        assert s._episode_phase == "resetting"

        s._parse_line("[lerobot] Recording episode 1")
        assert s._episode_phase == "recording"
        assert s._saved_episodes == 1

        s._parse_line("Right arrow key pressed.")
        assert s._episode_phase == "saving"

        s._parse_line("[lerobot] Stop recording")
        assert s._episode_phase == ""
        assert s._saved_episodes == 2


# ---------------------------------------------------------------------------
# send_key without subprocess
# ---------------------------------------------------------------------------

class TestSendKey:
    @pytest.mark.asyncio
    async def test_send_key_no_process_raises(self):
        session = DashboardSession()
        with pytest.raises(RuntimeError, match="No subprocess stdin"):
            await session.save_episode()
