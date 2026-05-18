from roboclaw.heartbeat.routing import pick_heartbeat_target


def test_pick_heartbeat_target_prefers_explicit_enabled_target() -> None:
    target = pick_heartbeat_target(
        enabled_channels=["web", "telegram"],
        sessions=[{"key": "web:recent"}],
        target_channel="telegram",
        target_chat_id="chat-1",
    )

    assert target == ("telegram", "chat-1")


def test_pick_heartbeat_target_falls_back_to_recent_external_session() -> None:
    target = pick_heartbeat_target(
        enabled_channels=["web"],
        sessions=[
            {"key": "cli:direct"},
            {"key": "system:web:internal"},
            {"key": "web:recent"},
            {"key": "telegram:disabled"},
        ],
        target_channel="telegram",
        target_chat_id="disabled-channel",
    )

    assert target == ("web", "recent")


def test_pick_heartbeat_target_falls_back_to_cli_when_no_route() -> None:
    target = pick_heartbeat_target(
        enabled_channels=["web"],
        sessions=[{"key": "cli:direct"}, {"key": "telegram:disabled"}],
    )

    assert target == ("cli", "direct")
