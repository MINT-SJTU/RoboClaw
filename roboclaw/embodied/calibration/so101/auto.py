"""SO-101 automatic calibration.

The heavy lifting happens in :mod:`roboclaw.embodied.calibration.so101.prober`. This
module orchestrates one ``MotorProber`` per joint on a fresh bus, runs the probing
sequence, writes narrowed ``Min_Position_Limit`` / ``Max_Position_Limit`` to EEPROM,
moves the arm to a rest pose, and releases torque.

Sequence (designed around SO-101 geometry — gripper independent, wrist_flex kept as a
brake while shoulder/elbow probe, shoulder+elbow probe paired so gravity is balanced):

    1. ``prepare`` all 6 motors (Torque_Enable=128 + widen Min/Max) one-by-one
    2. gripper ``run_full``
    3. wrist_flex ``probe(-1)``       — held retreated as wrist brake
    4. shoulder_pan ``run_full``
    5. ``paired_iter_probe`` on shoulder_lift + elbow_flex (refreshing wrist_flex hold
       between phases so voltage sag does not silently torque-off it)
    6. concurrent move shoulder_lift / elbow_flex to centres (wide tol)
    7. wrist_flex ``probe(+1)``
    8. wrist_roll ``capture_current_as_center`` (no hardstops)
    9. build ``ProbeResult`` per motor (applied = ``min_pos + SAFETY`` / ``max_pos - SAFETY``
       in whichever frame the last ``prepare`` / ``reset_center`` anchored)
    10. concurrent move to final rest pose (m1 centre / m2 min / m3 max /
        m4 (max+centre)/2 / m6 min) — done BEFORE EEPROM writes so the torque-off
        windows in ``_apply_results`` happen at gravity-rest positions (no sag)
    11. write narrowed Min/Max to EEPROM (Homing_Offset already persisted by firmware
        during every Torque_Enable=128)
    12. release torque on all motors
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event

from loguru import logger

from lerobot.motors.feetech.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

from roboclaw.embodied.calibration.model import CalibrationProfile, MotorCalibrationProfile
from roboclaw.embodied.calibration.so101.prober import (
    AutoCalibrationStopped,
    EEPROM_COMMIT_DELAY,
    MotorProber,
    POSITION_MAX,
    POSITION_MIN,
    _retry,
    concurrent_move,
    paired_iter_probe,
)
from roboclaw.embodied.calibration.store import CalibrationStore
from roboclaw.embodied.embodiment.manifest.binding import ArmBinding

SAFETY_MARGIN_TICKS = 20
M2M3_CENTER_TOL = 150


@dataclass(frozen=True)
class ProbeResult:
    motor_id: int
    motor_name: str
    hard_min: int
    hard_max: int
    applied_min: int
    applied_max: int
    homing_offset: int = 0
    drive_mode: int = 0


class _SO101AutoCalibrator:
    ARM_MOTORS: dict[str, int] = {
        "shoulder_pan": 1,
        "shoulder_lift": 2,
        "elbow_flex": 3,
        "wrist_flex": 4,
    }
    GRIPPER_NAME = "gripper"
    GRIPPER_ID = 6
    WRIST_ROLL_NAME = "wrist_roll"
    WRIST_ROLL_ID = 5
    ALL_MOTORS: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "gripper",
        "wrist_roll",
    )

    def __init__(self, port: str, *, stop_event: Event | None = None) -> None:
        self._port = port
        self._stop_event = stop_event
        motors: dict[str, Motor] = {
            name: Motor(id=motor_id, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100)
            for name, motor_id in self.ARM_MOTORS.items()
        }
        motors[self.GRIPPER_NAME] = Motor(
            id=self.GRIPPER_ID, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100,
        )
        motors[self.WRIST_ROLL_NAME] = Motor(
            id=self.WRIST_ROLL_ID, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100,
        )
        self._bus = FeetechMotorsBus(port=port, motors=motors)
        self._probers: dict[str, MotorProber] = {}

    def _check_stopped(self) -> None:
        if self._stop_event is not None and self._stop_event.is_set():
            raise AutoCalibrationStopped("Stopped by user.")

    def calibrate(self) -> dict[str, ProbeResult]:
        logger.info("[cal] begin port={}", self._port)
        self._bus.connect(handshake=True)
        self._probers = {
            "shoulder_pan":  MotorProber(self._bus, "shoulder_pan",  1, stop_event=self._stop_event),
            "shoulder_lift": MotorProber(self._bus, "shoulder_lift", 2, stop_event=self._stop_event),
            "elbow_flex":    MotorProber(self._bus, "elbow_flex",    3, stop_event=self._stop_event),
            "wrist_flex":    MotorProber(self._bus, "wrist_flex",    4, stop_event=self._stop_event),
            "gripper":       MotorProber(self._bus, "gripper",       self.GRIPPER_ID, stop_event=self._stop_event),
            "wrist_roll":    MotorProber(self._bus, "wrist_roll",    self.WRIST_ROLL_ID, stop_event=self._stop_event),
        }
        try:
            self._check_stopped()
            for name in self.ALL_MOTORS:
                self._prepare_motor(name)
            self._run_sequence()
            results = self._build_results()
            self._move_to_final_pose()
            self._apply_results(results)
            for p in self._probers.values():
                p.release()
            logger.info("[cal] complete port={}", self._port)
            return results
        except Exception:
            logger.exception("[cal] failed; restoring orig EEPROM")
            self._restore_via_fresh_bus()
            raise
        finally:
            try:
                self._bus.disconnect(disable_torque=True)
            except Exception as exc:
                logger.warning("[cal] disconnect raised: {}", exc)

    def _prepare_motor(self, name: str) -> None:
        self._probers[name].prepare()

    def _run_sequence(self) -> None:
        """Run the full probing sequence on the current ``self._probers``. After this call,
        every probed motor's ``min_pos`` / ``max_pos`` is valid in whatever frame the last
        ``prepare`` / ``reset_center`` firmware Torque_Enable=128 anchored."""
        p = self._probers

        logger.info("[cal:seq] step 2: gripper full probe")
        p[self.GRIPPER_NAME].run_full()

        logger.info("[cal:seq] step 3: wrist_flex -1 (held as wrist brake)")
        p["wrist_flex"].probe(-1)

        logger.info("[cal:seq] step 4: shoulder_pan full probe")
        p["shoulder_pan"].run_full()

        logger.info("[cal:seq] step 5: m2+m3 paired iter (refresh wrist_flex between phases)")
        paired_iter_probe(
            p["shoulder_lift"], p["elbow_flex"],
            refresh_holds=[p["wrist_flex"]],
        )

        logger.info("[cal:seq] step 6: m2+m3 -> centres (wide tol)")
        p["wrist_flex"].refresh_hold()
        concurrent_move(
            [
                (p["shoulder_lift"], p["shoulder_lift"].center),
                (p["elbow_flex"], p["elbow_flex"].center),
            ],
            tol=M2M3_CENTER_TOL,
        )

        logger.info("[cal:seq] step 7: wrist_flex +1")
        p["wrist_flex"].probe(+1)

        logger.info("[cal:seq] step 8: wrist_roll capture current as centre")
        p[self.WRIST_ROLL_NAME].capture_current_as_center()

    def _build_results(self) -> dict[str, ProbeResult]:
        results: dict[str, ProbeResult] = {}
        for name in self.ALL_MOTORS:
            self._check_stopped()
            p = self._probers[name]
            h_now = int(_retry(
                f"read Homing_Offset {name}",
                self._bus.read, "Homing_Offset", name, normalize=False,
            ))
            if name == self.WRIST_ROLL_NAME:
                results[name] = ProbeResult(
                    motor_id=p.motor_id, motor_name=name,
                    hard_min=POSITION_MIN, hard_max=POSITION_MAX,
                    applied_min=POSITION_MIN, applied_max=POSITION_MAX,
                    homing_offset=h_now, drive_mode=0,
                )
                logger.info(
                    "[cal:build] {} (no-probe) applied=[{},{}] h={}",
                    name, POSITION_MIN, POSITION_MAX, h_now,
                )
                continue
            hard_min, hard_max = p.min_pos, p.max_pos
            applied_min = max(POSITION_MIN, hard_min + SAFETY_MARGIN_TICKS)
            applied_max = min(POSITION_MAX, hard_max - SAFETY_MARGIN_TICKS)
            if applied_min >= applied_max:
                raise RuntimeError(
                    f"{name}: safety margin collapses range: "
                    f"hard=[{hard_min}, {hard_max}] margin={SAFETY_MARGIN_TICKS}"
                )
            results[name] = ProbeResult(
                motor_id=p.motor_id, motor_name=name,
                hard_min=hard_min, hard_max=hard_max,
                applied_min=applied_min, applied_max=applied_max,
                homing_offset=h_now, drive_mode=0,
            )
            logger.info(
                "[cal:build] {} hard=[{},{}] applied=[{},{}] h={}",
                name, hard_min, hard_max, applied_min, applied_max, h_now,
            )
        return results

    def _apply_results(self, results: dict[str, ProbeResult]) -> None:
        """Write the narrowed ``applied_min`` / ``applied_max`` to EEPROM. Homing_Offset
        was already written by the firmware when ``finalize_to_center`` fired."""
        for name, r in results.items():
            self._check_stopped()
            logger.info(
                "[cal:apply] {} Min={} Max={} (h={} already in EEPROM)",
                name, r.applied_min, r.applied_max, r.homing_offset,
            )
            _retry(f"disable_torque {name}", self._bus.disable_torque, name, num_retry=3)
            _retry(
                f"write Min_Position_Limit {name}", self._bus.write,
                "Min_Position_Limit", name, r.applied_min, normalize=False,
            )
            time.sleep(EEPROM_COMMIT_DELAY)
            _retry(
                f"write Max_Position_Limit {name}", self._bus.write,
                "Max_Position_Limit", name, r.applied_max, normalize=False,
            )
            time.sleep(EEPROM_COMMIT_DELAY)
            _retry(f"enable_torque {name}", self._bus.enable_torque, name, num_retry=3)

    def _move_to_final_pose(self) -> None:
        """Leave the arm in a compact rest pose: shoulder_pan at centre, shoulder_lift at
        min, elbow_flex at max, wrist_flex halfway between centre and max, gripper closed.
        wrist_roll is not moved (user's pose is already the centre)."""
        p = self._probers
        m4_target = (p["wrist_flex"].max_pos + p["wrist_flex"].center) // 2
        targets: list[tuple[MotorProber, int]] = [
            (p["shoulder_pan"], p["shoulder_pan"].center),
            (p["shoulder_lift"], p["shoulder_lift"].min_pos + SAFETY_MARGIN_TICKS),
            (p["elbow_flex"], p["elbow_flex"].max_pos - SAFETY_MARGIN_TICKS),
            (p["wrist_flex"], m4_target),
            (p[self.GRIPPER_NAME], p[self.GRIPPER_NAME].min_pos + SAFETY_MARGIN_TICKS),
        ]
        logger.info(
            "[cal:final] rest pose {}",
            {p.name: t for p, t in targets},
        )
        concurrent_move(targets, tol=M2M3_CENTER_TOL)

    def _restore_via_fresh_bus(self) -> None:
        """On failure path: drop the current bus, reconnect without handshake (tolerates a
        motor stuck in Overload latch), and restore each motor's snapshot of orig
        Min/Max/Homing_Offset from when ``prepare()`` fired."""
        try:
            self._bus.disconnect(disable_torque=True)
        except Exception as exc:
            logger.warning("[cal] disconnect before restore raised: {}", exc)
        time.sleep(0.3)
        try:
            self._bus.connect(handshake=False)
        except Exception:
            logger.exception("[cal] could not reconnect bus to restore EEPROM")
            return
        for name in self.ALL_MOTORS:
            prober = self._probers.get(name)
            if prober is None:
                continue
            prober.restore_orig_limits()


class SO101AutoCalibrationStrategy:
    def recalibrate(
        self,
        arm: ArmBinding,
        store: CalibrationStore,
        *,
        stop_event: Event | None = None,
    ) -> CalibrationProfile:
        del store  # no baseline needed — homing, range, drive_mode all computed from probe
        calibrator = _SO101AutoCalibrator(arm.port, stop_event=stop_event)
        probed = calibrator.calibrate()
        motors = {
            name: MotorCalibrationProfile(
                id=result.motor_id,
                drive_mode=result.drive_mode,
                homing_offset=result.homing_offset,
                range_min=result.applied_min,
                range_max=result.applied_max,
            )
            for name, result in probed.items()
        }
        return CalibrationProfile(motors)
