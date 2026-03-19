import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Ensure config is created with test-specific settings.
    monkeypatch.setenv("EMBEDDING_BACKEND", "mock")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    from app.core import config as config_module

    config_module.get_settings.cache_clear()
    importlib.invalidate_caches()

    # Reload app.main so it picks up updated settings.
    from app import main as main_module

    importlib.reload(main_module)
    with TestClient(main_module.app) as test_client:
        yield test_client

