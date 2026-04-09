"""Board channel constants and WebSocket type mapping."""

# Channel names
CH_SESSION = "session"
CH_CALIBRATION = "calibration"
CH_CONFIG = "config"
CH_FAULT_DETECTED = "fault.detected"
CH_FAULT_RESOLVED = "fault.resolved"

# Channel → WebSocket message type
WS_TYPES: dict[str, str] = {
    CH_SESSION: "dashboard.session.state_changed",
    CH_CALIBRATION: "dashboard.calibration.state_changed",
    CH_CONFIG: "dashboard.config.changed",
    CH_FAULT_DETECTED: "dashboard.fault.detected",
    CH_FAULT_RESOLVED: "dashboard.fault.resolved",
}
