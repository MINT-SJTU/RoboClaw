"""Root conftest — register custom markers and shared fixtures."""

from __future__ import annotations

import sys
import types

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "pty: PTY integration tests (require pexpect)")


def _ensure_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _install_alibaba_shims() -> None:
    try:
        import alibabacloud_pai_dlc20201203  # noqa: F401
        return
    except ImportError:
        pass

    root = _ensure_module("alibabacloud_pai_dlc20201203")
    client_mod = _ensure_module("alibabacloud_pai_dlc20201203.client")
    models_mod = _ensure_module("alibabacloud_pai_dlc20201203.models")

    class _Stub:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class CreateJobRequest(_Stub):
        pass

    class GetJobRequest(_Stub):
        pass

    class StopJobRequest(_Stub):
        pass

    class JobSpec(_Stub):
        pass

    class ResourceConfig(_Stub):
        pass

    class EnvVar(_Stub):
        pass

    class Client:
        def __init__(self, *args, **kwargs):
            pass

    models_mod.CreateJobRequest = CreateJobRequest
    models_mod.GetJobRequest = GetJobRequest
    models_mod.StopJobRequest = StopJobRequest
    models_mod.JobSpec = JobSpec
    models_mod.ResourceConfig = ResourceConfig
    models_mod.EnvVar = EnvVar
    client_mod.Client = Client
    root.models = models_mod
    root.client = client_mod

    tea_root = _ensure_module("alibabacloud_tea_openapi")
    tea_models = _ensure_module("alibabacloud_tea_openapi.models")

    class Config(_Stub):
        pass

    tea_models.Config = Config
    tea_root.models = tea_models

    oss2_mod = _ensure_module("oss2")

    class Auth:
        def __init__(self, *args, **kwargs):
            pass

    class Bucket:
        def __init__(self, *args, **kwargs):
            pass

    class ObjectIterator:
        def __init__(self, bucket, prefix=""):
            self._bucket = bucket
            self._prefix = prefix

        def __iter__(self):
            return iter([])

    oss2_mod.Auth = Auth
    oss2_mod.Bucket = Bucket
    oss2_mod.ObjectIterator = ObjectIterator


_install_alibaba_shims()
