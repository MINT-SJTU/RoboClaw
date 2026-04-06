---
title: "fix: Flexible device rebind without re-running full identify wizard"
type: fix
status: active
date: 2026-04-06
---

# fix: Flexible device rebind without re-running full identify wizard

## Overview

When a user makes a mistake during setup-identify (wrong role, wrong alias), the only way to fix it is unbind + re-run the entire wizard from scratch. This requires re-scanning, re-selecting model, re-doing motion detection for ALL arms. Add a `rebind` operation so users can change an arm's role or alias in one step.

## Problem Frame

Real user scenario (observed):
1. User runs `setup-identify`, accidentally sets both arms as `leader`
2. Wants to change one to `follower` and rename it
3. Must: unbind both → re-run full wizard → re-select model → re-do motion detection for ALL arms → re-name both
4. First re-identify attempt fails because old bindings still occupy the ports
5. Must: explicitly unbind → THEN re-run wizard again
6. Total: 3 full wizard runs to fix a typo. Should be 1 command.

## Requirements Trace

- R1. User can change an existing arm's role (leader→follower or vice versa) without re-running the identify wizard
- R2. User can change an existing arm's alias without re-running the identify wizard (rename already exists)
- R3. Agent knows when to use rebind vs full identify (AGENTS.md guidance)
- R4. Web UI device management supports role change (today's SetupView)

## Scope Boundaries

- NOT changing the identify wizard itself (that's a separate, larger UX redesign)
- NOT adding partial re-identify (fix one arm while keeping another)
- NOT changing how motion detection works

## Context & Research

### Relevant Code and Patterns

- `roboclaw/embodied/manifest/state.py` — `Manifest.set_arm()` creates new Binding, `rename_arm()` changes alias only
- `roboclaw/embodied/tool.py` — `_MODIFY_DISPATCH` maps operations to manifest methods. Currently: `unbind`, `rename`
- `roboclaw/embodied/service/config.py` — does not exist, config CRUD is in manifest/state.py directly
- `roboclaw/embodied/embodiment/arm/registry.py` — `get_arm_spec(type)` and `get_role(type)` parse "koch_leader" into spec + role
- `roboclaw/templates/AGENTS.md` — guides Agent to use `setup(action="modify", operation="rename|unbind")`

### Key Pattern

`rename_arm()` in `manifest/state.py` creates a new Binding with old spec/interface but new alias. `rebind_arm()` should follow the same pattern: create new Binding with same interface but new type_name (and optionally new alias).

## Key Technical Decisions

- **Rebind reuses the same Interface object** — no need for motion detection because we already know which physical port this arm is on
- **Rebind preserves calibration when type_name base model is the same** — changing koch_leader→koch_follower keeps calibration. Changing koch_leader→so101_follower clears it.
- **Single `rebind` operation, not separate `change_role` + `rename`** — simpler API, one manifest write

## Implementation Units

- [x] **Unit 1: Add `rebind_arm()` to Manifest**

**Goal:** Allow changing an existing arm's type_name and alias in one atomic operation

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Modify: `roboclaw/embodied/manifest/state.py`
- Test: `tests/test_manifest_rebind.py`

**Approach:**
- Add `rebind_arm(alias, new_alias, new_type)` method to `Manifest`
- Find existing binding by alias
- Validate new_type exists in arm registry
- Create new Binding with same interface + guard, but new type_name and alias
- If base model changed (koch→so101), clear calibration flag
- If base model same (koch_leader→koch_follower), preserve calibration
- Emit ConfigChangedEvent
- Atomic persist

**Patterns to follow:**
- `rename_arm()` in same file — same structure, same persist pattern

**Test scenarios:**
- Happy path: rebind "left_leader" to "left_follower" (koch_leader→koch_follower) → type changes, interface preserved, calibration preserved
- Happy path: rebind with new alias "left_leader" → "right_follower" → both alias and type change
- Edge case: rebind to same type → no-op or idempotent success
- Edge case: new_alias conflicts with existing binding → clear error
- Error path: alias not found → raise with message
- Error path: invalid new_type (e.g., "nonexistent_leader") → raise with message
- Integration: rebind emits ConfigChangedEvent that EventBus subscribers receive

**Verification:**
- `manifest.find_arm("right_follower")` returns binding with `type_name="koch_follower"` and same `interface.stable_id` as the original

---

- [x] **Unit 2: Add `rebind` action to tool layer**

**Goal:** Expose rebind_arm to the Agent via the manifest tool group

**Requirements:** R1, R3

**Dependencies:** Unit 1

**Files:**
- Modify: `roboclaw/embodied/tool.py`
- Modify: `tests/test_embodied_tool.py`

**Approach:**
- Add `"rebind"` to `_MODIFY_DISPATCH` alongside existing `"unbind"` and `"rename"`
- New action: `setup(action="modify", operation="rebind", target="arm", alias="old", new_alias="new", new_type="koch_follower")`
- `new_type` is required for rebind (unlike rename which only changes alias)

**Patterns to follow:**
- Existing `_execute_modify()` dispatch in tool.py — same if/elif pattern

**Test scenarios:**
- Happy path: tool.execute(action="modify", operation="rebind", target="arm", alias="left_leader", new_alias="left", new_type="koch_follower") → calls manifest.rebind_arm() → returns success message
- Error path: missing new_type parameter → clear error message
- Error path: alias not found → error propagated from manifest

**Verification:**
- Agent can say `setup(action="modify", operation="rebind", ...)` and it works end-to-end with stub mode

---

- [x] **Unit 3: Add rebind to AGENTS.md guidance**

**Goal:** Tell the Agent when to use rebind vs full identify

**Requirements:** R3

**Dependencies:** Unit 2

**Files:**
- Modify: `roboclaw/templates/AGENTS.md`

**Approach:**
- Add rule: "When user wants to change an arm's role (leader↔follower) or fix a naming mistake, use `setup(action='modify', operation='rebind')`. Do NOT re-run the full identify wizard for simple corrections."
- Add example: "把 left_leader 改成从臂" → rebind, not identify

**Test expectation:** none — pure documentation change

**Verification:**
- Agent reads AGENTS.md and uses rebind when user says "把这个臂改成从臂"

---

- [ ] **Unit 4: Add rebind to Web API (optional, if SetupView supports it)**

**Goal:** Web device management page can change arm role

**Requirements:** R4

**Dependencies:** Unit 1

**Files:**
- Modify: `roboclaw/http/routes/setup.py`
- Modify: `ui/src/controllers/setup.ts` (or `dashboard.ts`)

**Approach:**
- Add `POST /api/devices/arm/rebind` endpoint (or extend existing CRUD)
- Frontend: add "Change Role" dropdown or button in DeviceList component
- Follow today's Elvinyk pattern in `register_setup_routes()`

**Test scenarios:**
- Happy path: POST rebind → 200 + updated binding
- Error path: arm busy (embodiment lock held) → 409 Conflict
- Edge case: concurrent rebind requests → only first succeeds

**Verification:**
- In browser, can click "Change Role" on a device and see it update without page refresh

## System-Wide Impact

- **Interaction graph:** Rebind triggers ConfigChangedEvent → HardwareMonitor picks up the change → Web dashboard updates via EventBus
- **Unchanged invariants:** identify wizard, calibration flow, teleop/record — all unchanged. Rebind only modifies the manifest binding metadata.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Rebind during active session (teleop/record) | Check embodiment_busy before rebind, same as rename/unbind |
| Calibration loss on cross-model rebind | Detect base model change (koch vs so101), only clear calibration when models differ |

## Sources & References

- Related code: `roboclaw/embodied/manifest/state.py` (rename_arm pattern)
- User report: setup-identify UX bug observed during Koch arm testing
