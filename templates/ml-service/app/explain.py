"""Local Shapley values, what-if comparison and counterfactual search, all recomputed on the server."""
from mlkit.explain import counterfactuals, local_explanation
from mlkit.explainer import predict_frame

from .model_store import ModelStore


def _predict(store: ModelStore):
    return lambda frame: predict_frame(store.bundle, frame)


def local(store: ModelStore, profile: dict, background_rows: int) -> dict:
    frame = store.frame([profile])
    return local_explanation(_predict(store), frame, store.bundle.background.head(background_rows))


def what_if(store: ModelStore, baseline: dict, scenario: dict, background_rows: int) -> dict:
    base, scen = local(store, baseline, background_rows), local(store, scenario, background_rows)
    changed = [{"name": k, "old": baseline[k], "new": v} for k, v in scenario.items() if baseline.get(k) != v]
    return {"baseline": base, "scenario": scen, "changed": changed, "delta": scen["prediction"] - base["prediction"]}


def nearest_changes(store: ModelStore, profile: dict, max_results: int = 3) -> list[dict]:
    """Smallest single-predictor moves inside the training range that cross the stored threshold."""
    if not store.classification:
        return []
    return counterfactuals(_predict(store), store.frame([profile]), store.schema, store.metadata["threshold"], max_results)
