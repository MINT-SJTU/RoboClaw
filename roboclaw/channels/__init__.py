"""Chat channels module with plugin architecture."""

from roboclaw.channels.base import BaseChannel
from roboclaw.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
