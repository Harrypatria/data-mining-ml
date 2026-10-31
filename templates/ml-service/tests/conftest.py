"""Fixtures: a reference bundle trained once per session on simulated data, and app clients."""
import json
import sys
from pathlib import Path

import pytest

SERVICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE))
try:
    import mlkit  # noqa: F401  installed package (Docker image, pip install -e .)
except ImportError:
    sys.path.insert(0, str(SERVICE.parent / "src"))  # repository layout: templates/src or project/src

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from mlkit.workflow import train_reference_bundle  # noqa: E402


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    train_reference_bundle(root, seed=7, n=2500, n_boot=100)
    return root


def make_settings(models_dir: Path, **overrides) -> Settings:
    base = {"model_dir": models_dir, "app_env": "development", "cors_origins": "http://localhost:5173",
            "llm_provider": "none", "rate_limit_per_minute": 1000}
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture
def make_client(models_dir):
    clients = []

    def factory(**overrides) -> TestClient:
        client = TestClient(create_app(make_settings(models_dir, **overrides)))
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


@pytest.fixture
def profile(client) -> dict:
    return {f["name"]: f["default"] for f in client.get("/v1/schema").json()["features"]}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return events
