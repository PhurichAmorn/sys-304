import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def isolated_api_model_store(tmp_path, monkeypatch):
    import app

    monkeypatch.setattr(app, "MODEL_ROOT", tmp_path / "models")
    monkeypatch.setattr(app, "initialize_database", lambda: None)
    monkeypatch.setattr(app, "log_model_version", lambda **kwargs: None)
    monkeypatch.setattr(app, "log_request", lambda **kwargs: None)
    app.artifacts.clear()
