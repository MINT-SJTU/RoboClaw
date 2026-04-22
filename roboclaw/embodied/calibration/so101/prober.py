"""Object-oriented probe of a single Feetech STS motor.

Each ``MotorProber`` manages the full life-cycle of one motor during auto-calibration:
widen EEPROM limits, Torque_Enable=128 firmware reset to centre the frame, incremental
probe to hardstops in either direction, iterative midpoint resets when goal clamps at
a range edge, then finalise by locking the mechanical midpoint to Present=2048.

The prober exposes both blocking (``probe``, ``run_full``) and non-blocking (``start_probe``,
``probe_tick``) APIs. Module-level helpers drive multiple probers concurrently for gravity-
balanced paired probing of joints that must counterweight each other (shoulder_lift/elbow_flex
on SO-101).

The ``_retry`` wrapper handles transient bus errors (Input voltage, overload latch, comms
timeouts). Callers get a stable interface: a few retries with small delays are invisible to
the orchestration layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event
from typing import Any

from loguru import logger

HALF_TURN = 2048
POSITION_MIN = 0
POSITION_MAX = 4095
PROBE_STEP = 20
PROBE_INTERVAL = 0.04
MOVE_STEP = 16
MOVE_INTERVAL = 0.02
MOVE_TOL = 40
SAT_DELTA = 2
SAT_CYCLES = 6
CLAMP_EDGE_DIST = 50
WRAP_THRESHOLD = 1000
RETREAT_TICKS = 50
PROBE_MAX_TICKS = 250
MOVE_MAX_TICKS = 500
EEPROM_COMMIT_DELAY = 0.05
PROBE_TORQUE_LIMIT = 600    # RAM cap during probing — prevents bus-voltage collapse
                            # when two motors draw concurrently; reset in ``release``

TRANSIENT_ATTEMPTS = 5
TRANSIENT_DELAY = 0.05


class AutoCalibrationStopped(RuntimeError):
    """Raised when the caller signals stop via ``stop_event``."""


def _check_stopped(stop_event: Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise AutoCalibrationStopped("Stopped by user.")


def _retry(label: str, fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Retry a bus op on transient ConnectionError / RuntimeError (voltage / overload / TxRx)."""
    last: Exception | None = None
    for i in range(TRANSIENT_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except AutoCalibrationStopped:
            raise
        except (ConnectionError, RuntimeError) as exc:
            last = exc
            logger.warning(
                "[cal:retry] {} ({}/{}): {}", label, i + 1, TRANSIENT_ATTEMPTS, exc,
            )
            time.sleep(TRANSIENT_DELAY)
    assert last is not None
    raise last


def _read_pos(bus: Any, name: str) -> int:
    return int(_retry(f"read Present_Position {name}", bus.read, "Present_Position", name, normalize=False))


def _write_goal(bus: Any, name: str, goal: int) -> None:
    _retry(f"write Goal_Position {name}", bus.write, "Goal_Position", name, goal, normalize=False, num_retry=2)


@dataclass
class _ProbeState:
    direction: int = 0
    start_pos: int = 0
    last_pos: int = 0
    goal: int = 0
    stalled: int = 0
    done: bool = True
    result_pos: int = 0
    reason: str = ""


class MotorProber:
    """Per-motor probe state machine.

    Instantiate one per motor sharing a ``FeetechMotorsBus``. All EEPROM / torque
    actions use ``_retry`` for Layer-1 transient recovery.
    """

    def __init__(
        self,
        bus: Any,
        name: str,
        motor_id: int,
        *,
        stop_event: Event | None = None,
    ) -> None:
        self._bus = bus
        self._name = name
        self._motor_id = motor_id
        self._stop_event = stop_event
        self._min_pos = HALF_TURN
        self._max_pos = HALF_TURN
        self._min_real = False
        self._max_real = False
        self._orig_min: int | None = None
        self._orig_max: int | None = None
        self._orig_homing: int | None = None
        self._orig_torque_limit: int | None = None
        self._probe = _ProbeState()
        # Track the most recent Goal_Position we wrote. ``concurrent_move`` uses this as
        # the ramp start (instead of Present) so firmware sees smooth goal transitions
        # and doesn't interpret wrap-induced Present/goal mismatch as a huge commanded
        # multi-turn move.
        self._last_goal_written: int | None = None

    # ---------- state snapshot ----------
    @property
    def name(self) -> str: return self._name
    @property
    def motor_id(self) -> int: return self._motor_id
    @property
    def min_pos(self) -> int: return self._min_pos
    @property
    def max_pos(self) -> int: return self._max_pos
    @property
    def min_real(self) -> bool: return self._min_real
    @property
    def max_real(self) -> bool: return self._max_real
    @property
    def center(self) -> int: return (self._min_pos + self._max_pos) // 2
    @property
    def range_ticks(self) -> int: return self._max_pos - self._min_pos
    @property
    def orig_min(self) -> int | None: return self._orig_min
    @property
    def orig_max(self) -> int | None: return self._orig_max
    @property
    def orig_homing(self) -> int | None: return self._orig_homing

    def needs_more(self) -> bool:
        return not (self._min_real and self._max_real)

    def _check_stopped(self) -> None:
        _check_stopped(self._stop_event)

    def _set_goal(self, goal: int) -> None:
        """Write Goal_Position and remember the value for later smooth hand-off."""
        _write_goal(self._bus, self._name, goal)
        self._last_goal_written = goal

    # ---------- EEPROM ----------
    def prepare(self) -> None:
        """Snapshot orig Min/Max/Homing_Offset, fire Torque_Enable=128 (firmware-level
        multi-turn reset + auto-centre Present at 2048), widen Min/Max to full range."""
        self._check_stopped()
        logger.info("[cal:prep] {} prepare", self._name)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        self._orig_min = int(_retry(
            f"read Min_Position_Limit {self._name}",
            self._bus.read, "Min_Position_Limit", self._name, normalize=False,
        ))
        self._orig_max = int(_retry(
            f"read Max_Position_Limit {self._name}",
            self._bus.read, "Max_Position_Limit", self._name, normalize=False,
        ))
        self._orig_homing = int(_retry(
            f"read Homing_Offset {self._name}",
            self._bus.read, "Homing_Offset", self._name, normalize=False,
        ))
        self._orig_torque_limit = int(_retry(
            f"read Torque_Limit {self._name}",
            self._bus.read, "Torque_Limit", self._name, normalize=False,
        ))
        _retry(f"Torque_Enable=128 {self._name}", self._bus.write,
               "Torque_Enable", self._name, 128, normalize=False, num_retry=2)
        time.sleep(0.1)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Min_Position_Limit {self._name}", self._bus.write,
               "Min_Position_Limit", self._name, POSITION_MIN, normalize=False)
        time.sleep(EEPROM_COMMIT_DELAY)
        _retry(f"write Max_Position_Limit {self._name}", self._bus.write,
               "Max_Position_Limit", self._name, POSITION_MAX, normalize=False)
        time.sleep(EEPROM_COMMIT_DELAY)
        # RAM cap — limits peak current during probing so two concurrent-driving motors
        # cannot collapse the bus voltage. Reset to orig in ``release``.
        _retry(f"write Torque_Limit {self._name}", self._bus.write,
               "Torque_Limit", self._name, PROBE_TORQUE_LIMIT, normalize=False)
        time.sleep(EEPROM_COMMIT_DELAY)
        _retry(f"write Operating_Mode {self._name}", self._bus.write,
               "Operating_Mode", self._name, 0, normalize=False)
        self._min_pos = HALF_TURN
        self._max_pos = HALF_TURN
        self._min_real = False
        self._max_real = False
        logger.info(
            "[cal:prep] {} orig[min={} max={} h={} torque={}]",
            self._name, self._orig_min, self._orig_max, self._orig_homing, self._orig_torque_limit,
        )

    def reset_center(self) -> None:
        """Torque_Enable=128 at current physical pose. Clears this iteration's min/max."""
        self._check_stopped()
        logger.info("[cal:prober] {} reset_center", self._name)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"Torque_Enable=128 {self._name}", self._bus.write,
               "Torque_Enable", self._name, 128, normalize=False, num_retry=2)
        time.sleep(0.1)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write,
               "Operating_Mode", self._name, 0, normalize=False)
        self._min_pos = HALF_TURN
        self._max_pos = HALF_TURN
        self._min_real = False
        self._max_real = False

    def finalize_to_center(self) -> None:
        """Move the motor to its current-frame mechanical centre, then Torque_Enable=128.
        After this call, Present=2048 at the mechanical midpoint (real hardstops are at
        HALF_TURN ± range/2). The firmware-written Homing_Offset is the final value to
        persist via ``_build_results`` / ``_apply_results``."""
        self._check_stopped()
        half = self._max_pos - self._min_pos
        if half <= 0:
            logger.warning("[cal:prober] {} finalize_to_center skipped (no range)", self._name)
            return
        mid = (self._min_pos + self._max_pos) // 2
        logger.info("[cal:prober] {} finalize_to_center: move to {} then reset", self._name, mid)
        self.move_to(mid)
        # Torque_Enable=128 at the centre position.
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"Torque_Enable=128 {self._name}", self._bus.write,
               "Torque_Enable", self._name, 128, normalize=False, num_retry=2)
        time.sleep(0.1)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write,
               "Operating_Mode", self._name, 0, normalize=False)
        # Update internal state: min/max symmetric around HALF_TURN in the new frame.
        half_range = (self._max_pos - self._min_pos) // 2
        self._min_pos = HALF_TURN - half_range
        self._max_pos = HALF_TURN + half_range
        # Re-arm torque holding at new centre.
        p = _read_pos(self._bus, self._name)
        _write_goal(self._bus, self._name, p)
        _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=3)

    def capture_current_as_center(self) -> None:
        """For motors without hardstops (wrist_roll): current pose becomes centre.
        Torque_Enable=128 is a no-op aside from writing Homing_Offset so Present=2048 at
        current position; min/max stay at full 0..4095."""
        self._check_stopped()
        logger.info("[cal:prober] {} capture_current_as_center", self._name)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"Torque_Enable=128 {self._name}", self._bus.write,
               "Torque_Enable", self._name, 128, normalize=False, num_retry=2)
        time.sleep(0.1)
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        # Full range; no hardstops.
        self._min_pos = POSITION_MIN
        self._max_pos = POSITION_MAX
        self._min_real = True
        self._max_real = True

    def restore_orig_limits(self) -> None:
        """Write the snapshot taken in ``prepare()`` back to EEPROM (+ RAM Torque_Limit).
        Used on failure."""
        if self._orig_min is None or self._orig_max is None:
            return
        try:
            _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
            _retry(f"write Min_Position_Limit {self._name}", self._bus.write,
                   "Min_Position_Limit", self._name, self._orig_min, normalize=False)
            time.sleep(EEPROM_COMMIT_DELAY)
            _retry(f"write Max_Position_Limit {self._name}", self._bus.write,
                   "Max_Position_Limit", self._name, self._orig_max, normalize=False)
            time.sleep(EEPROM_COMMIT_DELAY)
            if self._orig_homing is not None:
                _retry(f"write Homing_Offset {self._name}", self._bus.write,
                       "Homing_Offset", self._name, self._orig_homing, normalize=False)
                time.sleep(EEPROM_COMMIT_DELAY)
            if self._orig_torque_limit is not None:
                _retry(f"write Torque_Limit {self._name}", self._bus.write,
                       "Torque_Limit", self._name, self._orig_torque_limit, normalize=False)
                time.sleep(EEPROM_COMMIT_DELAY)
            logger.info("[cal:prober] {} restored orig limits", self._name)
        except Exception:
            logger.exception("[cal:prober] {} restore failed", self._name)

    # ---------- probe — non-blocking pump ----------
    def start_probe(self, direction: int) -> None:
        """Arm for probing in the given direction. torque on, goal = current."""
        assert direction in (-1, +1)
        self._check_stopped()
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write,
               "Operating_Mode", self._name, 0, normalize=False)
        p = _read_pos(self._bus, self._name)
        self._probe = _ProbeState(
            direction=direction, start_pos=p, last_pos=p, goal=p, stalled=0, done=False
        )
        self._set_goal(p)
        _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=3)
        time.sleep(0.05)
        logger.debug("[cal:probe] {} start dir={:+d} start={}", self._name, direction, p)

    def probe_tick(self) -> bool:
        """Advance one tick. Returns True when the probe has finished (stall / clamp / wrap)."""
        s = self._probe
        if s.done:
            return True
        self._check_stopped()
        s.goal = max(POSITION_MIN, min(POSITION_MAX, s.goal + s.direction * PROBE_STEP))
        try:
            self._set_goal(s.goal)
        except Exception as e:
            logger.warning("[cal:probe] {} tick write fail: {}", self._name, e)
        try:
            pos = _read_pos(self._bus, self._name)
        except Exception as e:
            logger.warning("[cal:probe] {} tick read fail: {}", self._name, e)
            return False
        if abs(pos - s.last_pos) > WRAP_THRESHOLD:
            edge = POSITION_MIN if s.direction < 0 else POSITION_MAX
            # Pass s.last_pos (pre-wrap, still valid, near the edge) as actual_pos so
            # we park the motor inside the valid frame instead of at the bogus wrapped
            # reading — which would make the multi-turn controller spin a full rotation
            # back toward "that" position.
            self._finish_probe(edge, "wrap", s.last_pos)
            return True
        if abs(pos - s.last_pos) < SAT_DELTA:
            s.stalled += 1
            if s.stalled >= SAT_CYCLES:
                edge = POSITION_MIN if s.direction < 0 else POSITION_MAX
                travel = pos - s.start_pos
                dir_ok = (
                    (s.direction > 0 and travel > 0)
                    or (s.direction < 0 and travel < 0)
                    or abs(travel) < 30
                )
                reason = "clamp" if (abs(pos - edge) < CLAMP_EDGE_DIST or not dir_ok) else "real"
                self._finish_probe(pos, reason, pos)
                return True
        else:
            s.stalled = 0
        s.last_pos = pos
        return False

    def _finish_probe(self, report_pos: int, reason: str, actual_pos: int) -> None:
        s = self._probe
        # Park logic:
        #   real  — retreat 50 ticks off the hardstop so the motor relaxes instead of
        #           pinning while subsequent probes run.
        #   wrap  — ``actual_pos`` was remapped to pre-wrap last_pos (near the edge);
        #           same retreat applies, keeps the motor safely inside the valid frame.
        #   clamp — stalled against the goal-edge clamp without a real hardstop; park
        #           exactly there (no retreat).
        if reason in ("real", "wrap"):
            park_pos = actual_pos - s.direction * RETREAT_TICKS
        else:
            park_pos = actual_pos
        park_pos = max(POSITION_MIN, min(POSITION_MAX, park_pos))
        try:
            self._set_goal(park_pos)
        except Exception as e:
            logger.warning("[cal:probe] {} park fail: {}", self._name, e)
        s.done = True
        s.result_pos = report_pos
        s.reason = reason
        if s.direction < 0:
            self._min_pos = report_pos
            self._min_real = (reason == "real")
        else:
            self._max_pos = report_pos
            self._max_real = (reason == "real")
        logger.info(
            "[cal:probe] {} STOP dir={:+d} pos={} travel={:+d} reason={} park={}",
            self._name, s.direction, actual_pos, actual_pos - s.start_pos, reason, park_pos,
        )

    def is_probe_done(self) -> bool:
        return self._probe.done

    # ---------- blocking probe ----------
    def probe(self, direction: int) -> tuple[int, str]:
        """Blocking probe to stall. Returns (report_pos, reason)."""
        self.start_probe(direction)
        for _ in range(PROBE_MAX_TICKS):
            if self.probe_tick():
                break
            time.sleep(PROBE_INTERVAL)
        return self._probe.result_pos, self._probe.reason

    # ---------- moves ----------
    def goto_interim_midpoint(self) -> None:
        mid = (self._min_pos + self._max_pos) // 2
        logger.info("[cal:prober] {} goto_interim_midpoint {}", self._name, mid)
        self.move_to(mid)

    def move_to(self, target: int, *, tol: int = MOVE_TOL) -> None:
        """Blocking move to target in current frame. Tolerance is how close Present must
        get before we consider it settled (wider tol for moves where gravity would
        otherwise prevent landing exactly on target)."""
        self._check_stopped()
        _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        _retry(f"write Operating_Mode {self._name}", self._bus.write,
               "Operating_Mode", self._name, 0, normalize=False)
        p = _read_pos(self._bus, self._name)
        goal = p
        self._set_goal(p)
        _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=3)
        time.sleep(0.05)
        for _ in range(MOVE_MAX_TICKS):
            self._check_stopped()
            if goal != target:
                diff = target - goal
                step = max(-MOVE_STEP, min(MOVE_STEP, diff))
                goal += step
                try:
                    self._set_goal(goal)
                except Exception as e:
                    logger.warning("[cal:move] {} goal write fail: {}", self._name, e)
            try:
                pos = _read_pos(self._bus, self._name)
            except Exception as e:
                logger.warning("[cal:move] {} pos read fail: {}", self._name, e)
                time.sleep(MOVE_INTERVAL)
                continue
            if goal == target and abs(pos - target) < tol:
                return
            time.sleep(MOVE_INTERVAL)
        logger.warning("[cal:move] {} -> {} BUDGET EXHAUSTED", self._name, target)

    def refresh_hold(self) -> None:
        """Re-arm torque at current physical pose. Voltage-sag protection can silently
        disable torque on held motors; call this between phases of paired probing."""
        try:
            p = _read_pos(self._bus, self._name)
            self._set_goal(p)
            _retry(f"enable_torque {self._name}", self._bus.enable_torque, self._name, num_retry=2)
            logger.debug("[cal:prober] {} refresh_hold at pos={}", self._name, p)
        except Exception as e:
            logger.warning("[cal:prober] {} refresh_hold failed: {}", self._name, e)

    def release(self) -> None:
        """Restore orig Torque_Limit (RAM) so runtime torque is back to user's configured
        value, then disable torque."""
        logger.info("[cal:prober] {} release", self._name)
        if self._orig_torque_limit is not None:
            try:
                _retry(
                    f"restore Torque_Limit {self._name}",
                    self._bus.write, "Torque_Limit", self._name,
                    self._orig_torque_limit, normalize=False,
                )
            except Exception as e:
                logger.warning("[cal:prober] {} Torque_Limit restore failed: {}", self._name, e)
        try:
            _retry(f"disable_torque {self._name}", self._bus.disable_torque, self._name, num_retry=3)
        except Exception as e:
            logger.warning("[cal:prober] {} release failed: {}", self._name, e)

    # ---------- standalone ----------
    def run_full(self, max_iter: int = 4) -> None:
        """Iterate probe both directions until both sides register real stall, then
        leave the motor held at its mechanical midpoint (no final Torque_Enable=128 —
        call ``finalize_to_center`` if you want the frame locked)."""
        for _ in range(max_iter):
            self.probe(-1)
            self.probe(+1)
            if not self.needs_more():
                break
            self.goto_interim_midpoint()
            self.reset_center()
        self.move_to(self.center)


# =============== module-level helpers ===============

def paired_probe(p1: MotorProber, d1: int, p2: MotorProber, d2: int) -> None:
    """Concurrent paired probe — ``d1`` must equal ``-d2`` so the two joints move in
    opposing senses (gravity / arm-linkage balance)."""
    assert d1 == -d2, f"paired probe requires opposite dirs, got {d1=} {d2=}"
    logger.info(
        "[cal:prober] paired_probe {} d={:+d}, {} d={:+d}",
        p1.name, d1, p2.name, d2,
    )
    p1.start_probe(d1)
    p2.start_probe(d2)
    for _ in range(PROBE_MAX_TICKS):
        _check_stopped(p1._stop_event)
        a = p1.is_probe_done() or p1.probe_tick()
        b = p2.is_probe_done() or p2.probe_tick()
        if a and b:
            return
        time.sleep(PROBE_INTERVAL)


def paired_iter_probe(
    p1: MotorProber,
    p2: MotorProber,
    max_iter: int = 4,
    refresh_holds: list[MotorProber] | None = None,
) -> None:
    """Full paired iterative probing: Phase A (p1 -1, p2 +1) + Phase B (p1 +1, p2 -1);
    if either direction clamped, move both to their interim midpoints, Torque_Enable=128
    reset, and retry.

    ``refresh_holds``: other motors currently held (torque on, goal=pos) that should be
    re-armed between phases — voltage sag during paired probing can silently torque-off
    them otherwise."""
    refresh_holds = refresh_holds or []
    for i in range(1, max_iter + 1):
        logger.info("[cal:prober] paired_iter iter {}: {}+{}", i, p1.name, p2.name)
        paired_probe(p1, -1, p2, +1)
        for h in refresh_holds:
            h.refresh_hold()
        paired_probe(p1, +1, p2, -1)
        for h in refresh_holds:
            h.refresh_hold()
        if not (p1.needs_more() or p2.needs_more()):
            logger.info("[cal:prober] paired_iter done in {} iter(s)", i)
            return
        concurrent_move([(p1, p1.center), (p2, p2.center)])
        p1.reset_center()
        p2.reset_center()
        for h in refresh_holds:
            h.refresh_hold()
    logger.warning(
        "[cal:prober] paired_iter hit max_iter={}, {}.needs_more={} {}.needs_more={}",
        max_iter, p1.name, p1.needs_more(), p2.name, p2.needs_more(),
    )


def concurrent_move(
    pairs: list[tuple[MotorProber, int]],
    *,
    tol: int = MOVE_TOL,
) -> None:
    """Drive several motors to individual targets concurrently. Tolerant of transient bus
    errors (one failing tick does not abort the move)."""
    if not pairs:
        return
    logger.info(
        "[cal:prober] concurrent_move {} tol={}",
        [(p.name, t) for p, t in pairs], tol,
    )
    for p, _ in pairs:
        _retry(f"disable_torque {p.name}", p._bus.disable_torque, p.name, num_retry=3)
        _retry(f"write Operating_Mode {p.name}", p._bus.write,
               "Operating_Mode", p.name, 0, normalize=False)
    goals: dict[MotorProber, int] = {}
    for p, _ in pairs:
        # Ramp start: prefer the most recent Goal_Position we wrote (e.g. probe's park).
        # Using Present here would produce a large goal delta on wrap-affected motors
        # (Present wrapped to a value far from the firmware's internal goal), which the
        # firmware then executes as a huge commanded move.
        if p._last_goal_written is not None:
            start = p._last_goal_written
        else:
            start = _read_pos(p._bus, p.name)
        goals[p] = start
        p._set_goal(start)
        _retry(f"enable_torque {p.name}", p._bus.enable_torque, p.name, num_retry=3)
    time.sleep(0.05)
    done: dict[MotorProber, bool] = {p: False for p, _ in pairs}
    # use first prober's stop_event (all share same event in practice)
    stop_event = pairs[0][0]._stop_event
    for _ in range(MOVE_MAX_TICKS):
        _check_stopped(stop_event)
        for p, t in pairs:
            if done[p]:
                continue
            if goals[p] != t:
                diff = t - goals[p]
                step = max(-MOVE_STEP, min(MOVE_STEP, diff))
                goals[p] += step
                try:
                    p._set_goal(goals[p])
                except Exception as e:
                    logger.warning("[cal:move] {} goal write gave up: {}", p.name, e)
        for p, t in pairs:
            if done[p]:
                continue
            try:
                pos = _read_pos(p._bus, p.name)
            except Exception as e:
                logger.warning("[cal:move] {} pos read gave up: {}", p.name, e)
                continue
            if goals[p] == t and abs(pos - t) < tol:
                done[p] = True
                logger.info("[cal:move] {} settled at {} (target {})", p.name, pos, t)
        if all(done.values()):
            return
        time.sleep(MOVE_INTERVAL)
    logger.warning("[cal:move] concurrent_move BUDGET EXHAUSTED")
