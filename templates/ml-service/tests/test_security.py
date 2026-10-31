import pytest
from pydantic import ValidationError

from conftest import make_settings


def test_api_key_required_when_configured(make_client, profile):
    client = make_client(api_key="local-test-key")  # ml-ignore E02 test-only key for the auth check
    assert client.get("/v1/model").status_code == 401
    assert client.get("/v1/model", headers={"x-api-key": "wrong"}).status_code == 401
    assert client.get("/v1/model", headers={"x-api-key": "local-test-key"}).status_code == 200
    assert client.get("/health/live").status_code == 200


def test_cors_allowlist(client):
    ok = client.options("/v1/model", headers={"origin": "http://localhost:5173", "access-control-request-method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in ok.headers
    bad = client.get("/health/live", headers={"origin": "https://evil.example"})
    assert "access-control-allow-origin" not in bad.headers


def test_wildcard_cors_is_refused(models_dir):
    with pytest.raises(ValidationError):
        make_settings(models_dir, cors_origins="*")


def test_body_size_limit(make_client, profile):
    client = make_client(max_body_bytes=1024)
    r = client.post("/v1/predict", json={"profile": profile, "padding": "x" * 5000})
    assert r.status_code == 413


def test_rate_limit(make_client):
    client = make_client(rate_limit_per_minute=3)
    codes = [client.get("/v1/model").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[-1] == 429


def test_metrics_and_drift(client, profile):
    for i in range(120):
        client.post("/v1/predict", json={"profile": {**profile, "support_calls": i % 3}})
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/v1/predict",status="200"}' in text
    assert "prediction_probability_bin" in text and 'feature_drift_psi{feature="support_calls"}' in text


def test_request_id_round_trip(client):
    r = client.get("/health/live", headers={"x-request-id": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"
