from types import SimpleNamespace

from roboclaw.bus.events import OutboundMessage
from roboclaw.bus.queue import MessageBus
from roboclaw.channels.base import BaseChannel


class _DummyChannel(BaseChannel):
    name = "dummy"

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send(self, msg: OutboundMessage) -> None:
        return None


def test_is_allowed_requires_exact_match() -> None:
    channel = _DummyChannel(SimpleNamespace(allow_from=["allow@email.com"]), MessageBus())

    assert channel.is_allowed("allow@email.com") is True
    assert channel.is_allowed("attacker|allow@email.com") is False


def test_is_allowed_supports_dict_config() -> None:
    channel = _DummyChannel({"allow_from": ["*"]}, MessageBus())

    assert channel.is_allowed("web:user") is True
