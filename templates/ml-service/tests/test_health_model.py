import json
import shutil

from mlkit.persistence import resolve_model_dir


def test_live_and_ready(client):
    assert client.get("/health/live").json() == {"status": "live"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"


def test_model_info_reports_intervals_and_threshold(client):
    info = client.get("/v1/model").json()
    assert 0 < info["threshold"] < 1
    pr = info["metrics"]["pr_auc"]
    assert pr["low"] <= pr["value"] <= pr["high"]
    assert info["limitations"] and info["tested_model_note"]


def test_schema_matches_bundle(client, models_dir):
    schema = client.get("/v1/schema").json()
    on_disk = json.loads((resolve_model_dir(models_dir) / "schema.json").read_text())
    assert schema["columns"] == on_disk["columns"]
    for feat in schema["features"]:
        if feat["type"] in ("number", "integer"):
            assert feat["min"] <= feat["default"] <= feat["max"] and feat["step"] > 0


def test_tampered_model_is_not_loaded(make_client, models_dir, tmp_path):
    bundle = tmp_path / "v9.9.9"
    shutil.copytree(resolve_model_dir(models_dir), bundle)
    with (bundle / "model.pkl").open("ab") as f:
        f.write(b"tamper")
    client = make_client(model_dir=bundle)
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 503 and "checksum" in ready.json()["reason"]
    assert client.get("/v1/model").status_code == 503


def test_no_upload_or_model_path_endpoint(client):
    spec = client.app.openapi()
    for path, ops in spec["paths"].items():
        for op in ops.values():
            body = op.get("requestBody", {}).get("content", {})
            assert "multipart/form-data" not in body, path
    assert not any("upload" in p or "load" in p for p in spec["paths"])
