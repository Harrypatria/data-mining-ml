def test_predict_returns_probability_and_label(client, profile):
    out = client.post("/v1/predict", json={"profile": profile}).json()
    assert 0 <= out["probability"] <= 1
    assert out["label"] == ("above threshold" if out["probability"] >= out["threshold"] else "below threshold")
    assert out["extrapolated"] is False and set(out["extrapolation"]) == set(profile)
    assert out["request_id"]


def test_field_order_does_not_matter(client, profile):
    a = client.post("/v1/predict", json={"profile": profile}).json()["probability"]
    b = client.post("/v1/predict", json={"profile": dict(reversed(list(profile.items())))}).json()["probability"]
    assert a == b


def test_unknown_field_is_rejected(client, profile):
    r = client.post("/v1/predict", json={"profile": {**profile, "shoe_size": 42}})
    assert r.status_code == 422


def test_unknown_category_is_rejected(client, profile):
    r = client.post("/v1/predict", json={"profile": {**profile, "plan": "platinum"}})
    assert r.status_code == 422


def test_missing_field_is_rejected(client, profile):
    profile.pop("tenure_months")
    assert client.post("/v1/predict", json={"profile": profile}).status_code == 422


def test_out_of_range_needs_explicit_flag(client, profile):
    far = {**profile, "tenure_months": 500}
    r = client.post("/v1/predict", json={"profile": far})
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "out_of_range"
    ok = client.post("/v1/predict", json={"profile": far, "allow_extrapolation": True}).json()
    assert ok["extrapolated"] is True and ok["extrapolation"]["tenure_months"] is True


def test_batch(client, profile):
    profiles = [profile, {**profile, "support_calls": 6}]
    out = client.post("/v1/predict/batch", json={"profiles": profiles}).json()["predictions"]
    assert len(out) == 2 and out[1]["probability"] > out[0]["probability"]


def test_batch_size_limit(make_client, profile):
    small = make_client(max_batch_rows=2)
    assert small.post("/v1/predict/batch", json={"profiles": [profile] * 3}).status_code == 422
