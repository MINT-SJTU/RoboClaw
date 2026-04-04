"""Hand spec registry — single source of truth for supported hand types."""

from __future__ import annotations

from roboclaw.embodied.embodiment.hand.base import HandSpec

# ---------------------------------------------------------------------------
# Inspire RH56 (Modbus RTU, 115200)
# ---------------------------------------------------------------------------

INSPIRE_RH56 = HandSpec(
    name="inspire_rh56",
    finger_labels=("little", "ring", "middle", "index", "thumb_bend", "thumb_rotation"),
    num_fingers=6,
    baudrate=115200,
    default_slave_id=1,
    open_positions=(1000, 1000, 1000, 1000, 1000, 1000),
    close_positions=(0, 0, 0, 0, 0, 0),
    probe_register=1546,       # _REG_ANGLE_ACT
    probe_register_count=6,
    probe_candidates=None,     # defaults to range(1, 17) in probe function
    controller_module="roboclaw.embodied.embodiment.hand.inspire_rh56",
    controller_class="InspireController",
)

# ---------------------------------------------------------------------------
# BrainCo Revo2 (bc_stark_sdk, Modbus RS-485, 460800)
# ---------------------------------------------------------------------------

REVO2 = HandSpec(
    name="revo2",
    finger_labels=("thumb", "thumb_aux", "index", "middle", "ring", "pinky"),
    num_fingers=6,
    baudrate=460800,
    default_slave_id=0x7E,
    open_positions=(0, 0, 0, 0, 0, 0),
    close_positions=(400, 0, 1000, 1000, 1000, 1000),
    probe_register=0,
    probe_register_count=1,
    probe_candidates=tuple(list(range(1, 17)) + [0x7E, 0x7F]),
    controller_module="roboclaw.embodied.embodiment.hand.revo2",
    controller_class="Revo2Controller",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, HandSpec] = {
    "inspire_rh56": INSPIRE_RH56,
    "revo2": REVO2,
}


def get_hand_spec(hand_type: str) -> HandSpec:
    """Look up hand spec by type name."""
    spec = _REGISTRY.get(hand_type)
    if spec is None:
        raise ValueError(f"Unknown hand type: '{hand_type}'")
    return spec


def all_hand_types() -> tuple[str, ...]:
    """Return all registered hand type names."""
    return tuple(_REGISTRY.keys())


def all_hand_specs() -> dict[str, HandSpec]:
    """Return a copy of the registry."""
    return dict(_REGISTRY)
