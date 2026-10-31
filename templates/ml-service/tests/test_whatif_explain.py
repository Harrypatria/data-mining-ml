import pytest


def test_whatif_delta_and_additivity(client, profile):
    scenario = {**profile, "tenure_months": 3, "support_calls": 6}
    out = client.post("/v1/whatif", json={"scenario": scenario}).json()
    assert out["delta"] == pytest.approx(out["scenario"]["probability"] - out["baseline"]["probability"])
    assert {c["name"] for c in out["changed"]} == {"tenure_months", "support_calls"}
    total = sum(c["contribution"] for c in out["contributions"]["scenario"])
    assert out["base_value"] + total == pytest.approx(out["scenario"]["probability"], abs=1e-9)


def test_whatif_explicit_baseline(client, profile):
    baseline = {**profile, "autopay": True}
    out = client.post("/v1/whatif", json={"baseline": baseline, "scenario": {**baseline, "autopay": False}}).json()
    assert [c["name"] for c in out["changed"]] == ["autopay"]


def test_explain_local_is_additive_and_sorted(client, profile):
    out = client.post("/v1/explain/local", json={"profile": profile}).json()
    sizes = [abs(c["contribution"]) for c in out["contributions"]]
    assert sizes == sorted(sizes, reverse=True)
    assert out["base_value"] + sum(c["contribution"] for c in out["contributions"]) == pytest.approx(out["prediction"], abs=1e-9)
    assert "not causes" in out["note"]


def test_counterfactuals_cross_threshold_inside_range(client, profile):
    schema = {f["name"]: f for f in client.get("/v1/schema").json()["features"]}
    threshold = client.get("/v1/model").json()["threshold"]
    calls = range(int(schema["support_calls"]["min"]), int(schema["support_calls"]["max"]) + 1)
    profiles = [{**profile, "support_calls": c} for c in calls]
    probs = [p["probability"] for p in client.post("/v1/predict/batch", json={"profiles": profiles}).json()["predictions"]]
    risky, p = next((pr, pb) for pr, pb in zip(profiles, probs) if pb >= threshold)
    suggestions = client.post("/v1/explain/counterfactuals", json={"profile": risky}).json()["suggestions"]
    assert suggestions, "lowering support calls from the first value above the threshold must cross it"
    for s in suggestions:
        assert (s["prediction"] >= threshold) != (p >= threshold)
        f = schema[s["feature"]]
        if f["type"] in ("number", "integer"):
            assert f["min"] <= s["to"] <= f["max"]
