"""Root conftest — register custom markers and shared fixtures."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "pty: PTY integration tests (require pexpect)")
