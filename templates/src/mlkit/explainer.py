"""Server-side RTCC context assembly and the grounded answer loop, shared by the API and the notebook demo.

Everything the LLM sees is recomputed here from the loaded bundle and the raw profile values.
Clients never send predictions, contributions or other explanation values.
"""
from __future__ import annotations

from importlib import resources

import numpy as np
import pandas as pd

from . import narrative
from .explain import counterfactuals, local_explanation
from .persistence import Bundle, coerce_frame

STRICT_REMINDER = ("Your previous answer used numbers that are not in the CONTEXT JSON: {bad}. Rewrite it using only "
                   "numbers that appear in the JSON, keep every other constraint, and do not mention this correction.")


def packaged_prompt(name: str) -> str:
    """Read a prompt file shipped inside mlkit (synced from templates/prompts by scripts/gen_prompts.py)."""
    return resources.files("mlkit").joinpath("prompts", name).read_text(encoding="utf-8")


def predict_frame(bundle: Bundle, frame: pd.DataFrame) -> np.ndarray:
    frame = coerce_frame(frame, bundle.schema)
    if bundle.metadata.get("task") == "binary_classification":
        return bundle.pipeline.predict_proba(frame)[:, 1]
    return np.asarray(bundle.pipeline.predict(frame), dtype=float)


def extrapolation_flags(schema: dict, profile: dict) -> dict[str, bool]:
    flags = {}
    for f in schema["features"]:
        v = profile.get(f["name"])
        if f["type"] in ("number", "integer"):
            flags[f["name"]] = v is not None and not (f["min"] <= float(v) <= f["max"])
        elif f["type"] == "category":
            flags[f["name"]] = v not in f["allowed"]
        else:
            flags[f["name"]] = False
    return flags


def median_profile(schema: dict) -> dict:
    return {f["name"]: f["default"] for f in schema["features"]}


def model_card(meta: dict, schema: dict) -> dict:
    metric = {k: (round(v["value"], 4) if isinstance(v, dict) else v) for k, v in (meta.get("metrics") or {}).items()}
    return {
        "purpose": meta.get("model_purpose", ""),
        "target": meta.get("target", ""),
        "target_label": meta.get("target_label", meta.get("target", "")),
        "task": meta.get("task", "binary_classification"),
        "model_version": str(meta.get("model_version", "")),
        "metric": metric,
        "threshold": meta.get("threshold"),
        "limitations": list(meta.get("limitations", []))[:10],
        "features": [{"feature": f["name"], "label": f.get("label", f["name"]), "unit": f.get("unit", "")} for f in schema["features"]],
    }


def prediction_record(meta: dict, value: float) -> dict:
    if meta.get("task") == "binary_classification":
        label = "above threshold" if value >= meta["threshold"] else "below threshold"
        return {"value": None, "probability": float(value), "label": label}
    return {"value": float(value), "probability": None, "label": None}


def build_rtcc_context(bundle: Bundle, baseline: dict | None, scenario: dict, history: list[dict] | None = None,
                       background_rows: int = 100, with_what_if: bool = True) -> dict:
    schema, meta = bundle.schema, bundle.metadata
    baseline = baseline or median_profile(schema)
    background = bundle.background.head(background_rows)

    def predict(frame):
        return predict_frame(bundle, frame)

    base_frame = coerce_frame(pd.DataFrame([baseline]), schema)
    scen_frame = coerce_frame(pd.DataFrame([scenario]), schema)
    exp_base = local_explanation(predict, base_frame, background)
    exp_scen = local_explanation(predict, scen_frame, background)
    what_if = []
    if with_what_if and meta.get("task") == "binary_classification":
        what_if = counterfactuals(predict, scen_frame, schema, meta["threshold"])
    ranges = {f["name"]: ({"min": f["min"], "max": f["max"]} if f["type"] in ("number", "integer")
                          else {"allowed": ", ".join(map(str, f.get("allowed", [True, False])))}) for f in schema["features"]}
    context = narrative.build_context(
        model_card=model_card(meta, schema),
        baseline_profile=baseline, scenario_profile=scenario,
        predictions={"baseline": prediction_record(meta, exp_base["prediction"]),
                     "scenario": prediction_record(meta, exp_scen["prediction"])},
        shap_baseline=exp_base["contributions"], shap_scenario=exp_scen["contributions"],
        global_importance=meta.get("global_importance", []),
        training_ranges=ranges, extrapolation_flags=extrapolation_flags(schema, scenario),
        calibration_note=meta.get("calibration_note", ""), what_if=what_if, history=history,
        units={f["name"]: f.get("unit", "") for f in schema["features"]})
    return context


def grounded_answer(context: dict, question: str, system_prompt: str, generate, max_attempts: int = 2) -> dict:
    """Ask the LLM, check every number against the context, regenerate once, else fall back to the template.

    generate(system_prompt, messages) -> str may raise anything; any failure returns the template explanation.
    """
    messages = [{"role": "user", "content": narrative.build_user_message(context, question)}]
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            text = generate(system_prompt, messages)
        except Exception as exc:  # provider, network and parsing failures all end in the deterministic fallback
            return {"text": narrative.template_explanation(context), "source": "template",
                    "reason": f"provider unavailable: {type(exc).__name__}", "grounding": None, "attempts": attempt}
        result = narrative.check_grounding(text, context)
        if result["ok"]:
            return {"text": text, "source": "llm", "reason": "", "grounding": result, "attempts": attempt}
        last = result
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": STRICT_REMINDER.format(bad=", ".join(result["unsupported"][:10]))}]
    return {"text": narrative.template_explanation(context), "source": "template",
            "reason": "answer failed the numeric grounding check twice", "grounding": last, "attempts": max_attempts}
