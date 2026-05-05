"""Embodied verification interfaces and preflight checks."""

from roboclaw.embodied.service.verification.preflight import (
    PreflightVerifier,
    RecordPreflightVerifier,
    TrainPreflightVerifier,
    Verifier,
)
from roboclaw.embodied.service.verification.types import (
    VerificationRequest,
    VerificationResult,
    Violation,
)

__all__ = [
    "PreflightVerifier",
    "RecordPreflightVerifier",
    "TrainPreflightVerifier",
    "VerificationRequest",
    "VerificationResult",
    "Verifier",
    "Violation",
]
