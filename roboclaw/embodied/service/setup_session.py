"""SetupSession — stateful setup workflow, sub-service of EmbodiedService.

Drives the discover → identify → assign → commit workflow.
Owns embodiment locking for scan/motion operations.
Shared by CLI agent and Web UI.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.hardware.discovery import HardwareDiscovery
from roboclaw.embodied.hardware.scan import restore_stderr, suppress_stderr
from roboclaw.embodied.interface import Interface, SerialInterface, VideoInterface
from roboclaw.i18n import t

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService


class SetupPhase(str, Enum):
    IDLE = "idle"
    DISCOVERING = "discovering"
    ASSIGNING = "assigning"
    IDENTIFYING = "identifying"
    COMMITTED = "committed"


@dataclass
class Assignment:
    """A pending assignment — interface matched to alias + spec, not yet committed."""

    alias: str
    spec_name: str  # e.g. "so101_follower", "inspire_rh56", "opencv"
    interface: Interface
    slave_id: int = 0  # hand-specific, set during assign if needed


class SetupSession:
    """Drives the discover → identify → assign → commit workflow.

    This is a direct sub-service of EmbodiedService, replacing ScanningService.
    It manages embodiment locking internally.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent
        self._phase = SetupPhase.IDLE
        self._model: str = ""
        self._discovery: HardwareDiscovery | None = None
        self._candidates: list[SerialInterface | VideoInterface] = []
        self._assignments: list[Assignment] = []
        # Prompting protocol state
        self._awaiting_alias_for: str = ""
        self._pending_role: str = ""  # "follower" or "leader", set after role selection
        self._pending_kwargs: dict[str, Any] = {}
        self._result: str = ""
        self._camera_index: int = 0
        self._embodiment_category: str = ""
        self._language: str = "en"

    @property
    def phase(self) -> SetupPhase:
        return self._phase

    @property
    def _current_spec(self):
        """Look up the EmbodimentSpec for the current model."""
        if not self._model:
            return None
        from roboclaw.embodied.embodiment.catalog import get_spec
        try:
            return get_spec(self._model)
        except ValueError:
            return None

    @property
    def motion_active(self) -> bool:
        return self._phase == SetupPhase.IDENTIFYING

    @property
    def candidates(self) -> list[SerialInterface | VideoInterface]:
        return list(self._candidates)

    @property
    def assignments(self) -> list[Assignment]:
        return list(self._assignments)

    @property
    def unassigned(self) -> list[SerialInterface | VideoInterface]:
        assigned_ids = {a.interface.stable_id for a in self._assignments}
        return [c for c in self._candidates if c.stable_id not in assigned_ids]

    # -- Scan ----------------------------------------------------------------

    def run_full_scan(self, model: str = "") -> dict[str, Any]:
        """Scan ports + cameras. Resets session state.

        Acquires/releases embodiment lock during scan.
        Returns {ports: list[SerialInterface], cameras: list[VideoInterface]}.
        """
        self._parent.acquire_embodiment("scanning")
        try:
            self.reset()
            self._model = model
            self._phase = SetupPhase.DISCOVERING
            self._discovery = HardwareDiscovery()
            if model:
                serial = self._discovery.discover(model)
            else:
                serial = self._discovery.discover_all()
            video = self._discovery.discover_cameras()
            self._candidates = [*serial, *video]
            self._phase = SetupPhase.ASSIGNING
            ports = [c for c in self._candidates if isinstance(c, SerialInterface)]
            cameras = [c for c in self._candidates if isinstance(c, VideoInterface)]
            return {"ports": ports, "cameras": cameras}
        except Exception:
            self.reset()
            raise
        finally:
            self._parent.release_embodiment()

    def capture_previews(self, output_dir: str) -> list[dict]:
        """Capture camera preview frames."""
        self._parent.acquire_embodiment("camera-preview")
        try:
            discovery = self._discovery or HardwareDiscovery()
            if not discovery.scanned_cameras:
                discovery.discover_cameras()
            return discovery.capture_camera_previews(output_dir)
        finally:
            self._parent.release_embodiment()

    # -- Identify (motion detection) -----------------------------------------

    def start_motion_detection(self) -> int:
        """Start motion detection. Acquires embodiment lock until stop."""
        if self._phase == SetupPhase.IDLE:
            raise RuntimeError("No scan performed. Run scan first.")
        self._parent.acquire_embodiment("motion-detection")
        try:
            return self._start_identify()
        except Exception:
            self._parent.release_embodiment()
            raise

    def _start_identify(self) -> int:
        serial = [c for c in self.unassigned if isinstance(c, SerialInterface)]
        if not serial:
            raise RuntimeError("No serial interfaces to identify.")
        saved = suppress_stderr()
        try:
            for iface in serial:
                iface.motion_detector.capture_baseline()
        finally:
            restore_stderr(saved)
        self._phase = SetupPhase.IDENTIFYING
        return len(serial)

    def stop_motion_detection(self) -> None:
        """Stop motion detection and release embodiment lock."""
        if self._phase == SetupPhase.IDENTIFYING:
            for c in self._candidates:
                if isinstance(c, SerialInterface):
                    c.motion_detector.reset()
            self._phase = SetupPhase.ASSIGNING
        self._parent.release_embodiment(owner="motion-detection")

    def poll_motion(self) -> list[dict[str, Any]]:
        """Poll motion on unassigned serial candidates."""
        if self._phase != SetupPhase.IDENTIFYING:
            raise RuntimeError("Motion detection not started.")
        serial = [c for c in self.unassigned if isinstance(c, SerialInterface)]
        results: list[dict[str, Any]] = []
        saved = suppress_stderr()
        try:
            for iface in serial:
                result = iface.motion_detector.poll()
                results.append({
                    "stable_id": iface.stable_id,
                    "dev": iface.dev,
                    "by_id": iface.by_id,
                    "motor_ids": list(iface.motor_ids),
                    "delta": result.delta,
                    "moved": result.moved,
                })
        finally:
            restore_stderr(saved)
        return results

    # -- Assign / Commit -----------------------------------------------------

    def assign(
        self, interface_stable_id: str, alias: str, spec_name: str,
    ) -> Assignment:
        """Assign a discovered interface to an alias and spec."""
        if self._phase not in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            raise RuntimeError(f"Cannot assign in {self._phase} phase.")
        interface = None
        for c in self.unassigned:
            if c.stable_id == interface_stable_id:
                interface = c
                break
        if interface is None:
            raise ValueError(
                f"Interface {interface_stable_id} not found or already assigned."
            )
        if any(a.alias == alias for a in self._assignments):
            raise ValueError(f"Alias '{alias}' already assigned.")
        assignment = Assignment(alias=alias, spec_name=spec_name, interface=interface)
        self._assignments.append(assignment)
        return assignment

    def unassign(self, alias: str) -> None:
        """Remove an assignment, returning the interface to unassigned."""
        if self._phase not in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            raise RuntimeError(f"Cannot unassign in {self._phase} phase.")
        for i, a in enumerate(self._assignments):
            if a.alias == alias:
                self._assignments.pop(i)
                return
        raise ValueError(f"No assignment with alias '{alias}'.")

    def commit(self) -> int:
        """Write all assignments to manifest.

        Transitions: ASSIGNING/IDENTIFYING → COMMITTED.
        Returns number of bindings created.
        """
        if self._phase not in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            raise RuntimeError(f"Cannot commit in {self._phase} phase.")
        if not self._assignments:
            raise RuntimeError("No assignments to commit.")
        if self._phase == SetupPhase.IDENTIFYING:
            self.stop_motion_detection()
        manifest = self._parent.manifest
        for a in self._assignments:
            self._commit_one(manifest, a)
        count = len(self._assignments)
        self._phase = SetupPhase.COMMITTED
        return count

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state for API responses."""
        return {
            "phase": self._phase.value,
            "model": self._model,
            "candidates": [
                {"stable_id": c.stable_id, "interface_type": c.interface_type, **c.to_dict()}
                for c in self._candidates
            ],
            "assignments": [
                {
                    "alias": a.alias,
                    "spec_name": a.spec_name,
                    "interface_stable_id": a.interface.stable_id,
                }
                for a in self._assignments
            ],
            "unassigned": [c.stable_id for c in self.unassigned],
        }

    # -- Prompting protocol (used by TtySession) ----------------------------

    def interaction_spec(self):
        from roboclaw.embodied.adapters.protocol import PromptingSpec

        return PromptingSpec(label="setup-identify")

    async def run_identify(self, kwargs: dict[str, Any], tty_handoff: Any) -> str:
        """Entry point for the identify flow with TtySession."""
        self.reset()
        self._pending_kwargs = kwargs
        self._language = kwargs.get("language", "en")
        if tty_handoff:
            from roboclaw.embodied.adapters.tty import TtySession

            try:
                return await TtySession(tty_handoff).run(self)
            except Exception:
                self._cleanup_motion()
                raise
        return self._conversational_identify(kwargs)

    def _cleanup_motion(self) -> None:
        """Release motion detection lock if active."""
        if self._phase == SetupPhase.IDENTIFYING:
            self.stop_motion_detection()

    def next_step(self):
        """Return the next interaction step, or None when done."""
        from roboclaw.embodied.adapters.protocol import PollStep, PromptStep

        if self._result:
            return None

        if self._phase == SetupPhase.IDLE:
            return self._next_step_idle()

        if self._awaiting_alias_for:
            lang = self._language
            spec = self._current_spec
            roles = spec.roles if spec else ()
            if roles and not self._pending_role:
                options = [t(f"role_{r}", lang) for r in roles]
                return PromptStep("role", t("selectRole", lang), options=options)
            return PromptStep("alias", t("aliasPrompt", lang))

        if self._phase in (SetupPhase.ASSIGNING, SetupPhase.IDENTIFYING):
            return self._next_step_assigning()

        return None

    def submit_answer(self, prompt_id: str, answer: str) -> None:
        """Process user answer for the current step."""
        if prompt_id == "embodiment_type":
            self._submit_embodiment_type(answer)
        elif prompt_id == "model":
            self._submit_model(answer)
        elif prompt_id == "motion":
            self._awaiting_alias_for = answer
            port_short = answer.rsplit("/", 1)[-1] if "/" in answer else answer
            print(t("detectedMotion", self._language, port=port_short))
        elif prompt_id == "role":
            spec = self._current_spec
            roles = spec.roles if spec else ("follower", "leader")
            idx = int(answer) - 1
            self._pending_role = roles[idx]
        elif prompt_id == "alias":
            self._submit_alias(answer)
        elif prompt_id == "confirm":
            self._submit_confirm(answer)
        elif prompt_id.startswith("camera_"):
            self._handle_camera_answer(prompt_id, answer)

    def result(self) -> str:
        if self._result:
            return self._result
        return json.dumps({
            "status": "no_assignments",
            "message": t("noAssignments", self._language),
            "bindings": 0,
        }, ensure_ascii=False)

    # -- Prompting helpers (private) -----------------------------------------

    def _next_step_idle(self):
        from roboclaw.embodied.adapters.protocol import PromptStep
        from roboclaw.embodied.embodiment.catalog import (
            EmbodimentCategory, models_for,
        )
        lang = self._language

        model = self._pending_kwargs.get("model", "")
        if model:
            self._do_scan(model)
            return self.next_step()

        # Step 1: embodiment type selection (if not yet chosen)
        if not self._embodiment_category:
            return PromptStep(
                "embodiment_type",
                t("selectEmbodimentType", lang) + ":",
                options=[
                    t("arm", lang),
                    t("hand", lang),
                    t("humanoidSoon", lang),
                    t("mobileSoon", lang),
                ],
            )

        # Step 2: model selection within the chosen category
        category = EmbodimentCategory(self._embodiment_category)
        specs = models_for(category)
        if not specs:
            self._set_result("not_supported")
            return None
        options = [s.name.upper().replace("_", " ") for s in specs]
        n = len(options)
        return PromptStep(
            "model",
            t("selectModel", lang, n=f"1/{n}"),
            options=options,
        )

    def _next_step_assigning(self):
        from roboclaw.embodied.adapters.protocol import PollStep, PromptStep

        serial_unassigned = [u for u in self.unassigned if isinstance(u, SerialInterface)]

        # Start motion detection if serial ports remain
        if serial_unassigned and not self.motion_active:
            self.start_motion_detection()

        if self.motion_active and serial_unassigned:
            return PollStep(
                "motion",
                t("unassignedPorts", self._language, n=len(serial_unassigned)),
                poll_fn=self._poll_one_motion,
                timeout_s=30.0,
                timeout_message=t("timeout", self._language),
                retry_prompt=t("retryPrompt", self._language),
            )

        # Stop motion detection when all serial ports are done
        if self.motion_active:
            self.stop_motion_detection()

        # Camera naming
        camera_step = self._next_camera_step()
        if camera_step:
            return camera_step

        # Confirmation
        if self._assignments:
            self._show_assignments()
            return PromptStep("confirm", t("commitPrompt", self._language))

        return None

    def _do_scan(self, model: str) -> None:
        """Run full scan and print summary."""
        lang = self._language
        print(t("scanningModel", lang, model=model))
        try:
            result = self.run_full_scan(model)
        except (ValueError, RuntimeError) as exc:
            print(f"  Error: {exc}")
            self._set_result("not_supported")
            return
        ports = result["ports"]
        cameras = result["cameras"]
        print(t("foundPorts", lang, ports=len(ports), cameras=len(cameras)))
        if not ports and not cameras:
            self._set_result("no_hardware")
            return
        for i, port in enumerate(ports):
            port_id = port.by_id or port.dev or "?"
            print(f"  [{i}] {port_id}  ({len(port.motor_ids)} {t('motorsFound', lang)})")

    def _poll_one_motion(self) -> str | None:
        """Poll motion; return stable_id of first moved port, or None."""
        results = self.poll_motion()
        for r in results:
            if r["moved"]:
                return r["stable_id"]
        return None

    def _submit_model(self, answer: str) -> None:
        from roboclaw.embodied.embodiment.catalog import EmbodimentCategory, models_for
        category = EmbodimentCategory(self._embodiment_category) if self._embodiment_category else EmbodimentCategory.ARM
        specs = models_for(category)
        # Map numeric answer to spec name
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(specs):
                model = specs[idx].name
            else:
                model = answer.lower()
        except ValueError:
            model = answer.lower()
        self._do_scan(model)

    def _submit_alias(self, answer: str) -> None:
        lang = self._language
        stable_id = self._awaiting_alias_for
        role = self._pending_role
        self._awaiting_alias_for = ""
        self._pending_role = ""
        if not answer:
            print(t("skipped", lang))
            return
        spec = self._current_spec
        alias = f"{answer}_{role}" if role else answer
        spec_name = spec.spec_name_for(role) if spec else f"{self._model}_{role}"
        try:
            self.assign(stable_id, alias, spec_name)
            print(t("assigned", lang, alias=alias, spec=spec_name))
        except ValueError as exc:
            print(f"  Error: {exc}")

    def _submit_confirm(self, answer: str) -> None:
        if answer.lower() == "y":
            count = self.commit()
            self._set_result("committed", count=count)
        else:
            self.reset()
            self._set_result("cancelled")

    @property
    def _video_candidates(self) -> list[VideoInterface]:
        """All video candidates (stable order, not affected by assignments)."""
        return [c for c in self._candidates if isinstance(c, VideoInterface)]

    def _next_camera_step(self):
        from roboclaw.embodied.adapters.protocol import PromptStep

        lang = self._language
        all_video = self._video_candidates
        if not all_video:
            return None
        if self._camera_index == 0:
            print(t("cameraNaming", lang))
        if self._camera_index >= len(all_video):
            return None
        cam = all_video[self._camera_index]
        # Skip already-assigned cameras
        assigned_ids = {a.interface.stable_id for a in self._assignments}
        if cam.stable_id in assigned_ids:
            self._camera_index += 1
            return self._next_camera_step()
        dev = cam.dev or "?"
        res = f"{cam.width}x{cam.height}" if cam.width else "?"
        print(f"  [{self._camera_index}] {dev} ({res} @ {cam.fps}fps)")
        return PromptStep(
            f"camera_{self._camera_index}",
            t("cameraNamePrompt", lang, index=self._camera_index),
        )

    def _handle_camera_answer(self, prompt_id: str, answer: str) -> None:
        all_video = self._video_candidates
        idx = int(prompt_id.split("_", 1)[1])
        if idx < len(all_video) and answer:
            cam = all_video[idx]
            try:
                self.assign(cam.stable_id, answer, "opencv")
                print(t("assigned", self._language, alias=answer, spec="opencv"))
            except ValueError as exc:
                print(f"  Error: {exc}")
        self._camera_index = idx + 1

    def _show_assignments(self) -> None:
        lang = self._language
        print(t("assignments", lang, count=len(self._assignments)))
        for a in self._assignments:
            sid = a.interface.stable_id[:30]
            print(f"  {a.alias} -> {a.spec_name} ({sid}...)")

    def _submit_embodiment_type(self, answer: str) -> None:
        from roboclaw.embodied.embodiment.catalog import EmbodimentCategory
        cats = list(EmbodimentCategory)
        try:
            idx = int(answer) - 1
            if 0 <= idx < len(cats):
                self._embodiment_category = cats[idx].value
            else:
                return
        except ValueError:
            low = answer.lower()
            try:
                EmbodimentCategory(low)
                self._embodiment_category = low
            except ValueError:
                return

    def on_timeout(self) -> None:
        """Called when motion detection times out and user declines retry."""
        ports = [c for c in self._candidates if isinstance(c, SerialInterface)]
        cameras = [c for c in self._candidates if isinstance(c, VideoInterface)]
        self._set_result("timeout_declined", ports=len(ports), cameras=len(cameras))

    def _set_result(self, status: str, **kwargs: Any) -> None:
        """Set structured result message."""
        lang = self._language
        message_keys = {
            "committed": "resultCommitted",
            "cancelled": "resultCancelled",
            "timeout_declined": "resultTimeout",
            "no_hardware": "resultNoHardware",
            "not_supported": "resultNotSupported",
        }
        key = message_keys.get(status, "noAssignments")
        message = t(key, lang, **kwargs)
        self._result = json.dumps({
            "status": status,
            "message": message,
            "bindings": kwargs.get("count", 0),
        }, ensure_ascii=False)

    def _conversational_identify(self, kwargs: dict[str, Any]) -> str:
        """Return session state as JSON for conversational agents (no TTY)."""
        import asyncio
        import json

        if self._phase == SetupPhase.IDLE:
            model = kwargs.get("model", "")
            if not model:
                return json.dumps({
                    "phase": "idle",
                    "message": "What robot model do you have?",
                    "options": ["so101", "koch"],
                }, ensure_ascii=False)
            asyncio.get_event_loop().run_in_executor(None, self.run_full_scan, model)
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    # -- Internal ------------------------------------------------------------

    def reset(self) -> None:
        """Reset session to idle state."""
        self._phase = SetupPhase.IDLE
        self._model = ""
        self._discovery = None
        self._candidates = []
        self._assignments = []
        self._awaiting_alias_for = ""
        self._pending_kwargs = {}
        self._result = ""
        self._camera_index = 0
        self._embodiment_category = ""
        # Note: do NOT reset self._language here — it persists through reset

    @staticmethod
    def _commit_one(manifest: Any, assignment: Assignment) -> None:
        from roboclaw.embodied.embodiment.arm.registry import all_arm_types
        from roboclaw.embodied.embodiment.hand.registry import all_hand_types

        if assignment.spec_name in all_arm_types():
            manifest.set_arm(assignment.alias, assignment.spec_name, assignment.interface)
        elif assignment.spec_name in all_hand_types():
            manifest.set_hand(
                assignment.alias, assignment.spec_name,
                assignment.interface, assignment.slave_id,
            )
        elif isinstance(assignment.interface, VideoInterface):
            manifest.set_camera(assignment.alias, assignment.interface)
        else:
            raise ValueError(f"Unknown spec type: {assignment.spec_name}")


