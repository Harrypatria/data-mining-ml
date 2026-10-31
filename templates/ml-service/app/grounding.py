"""Numeric grounding for LLM explanations.

Every number in the model's answer must appear in the RTCC context, as a percent form of a
context value, or as a simple delta of two context values, within the rounding the answer
used. On failure the answer is regenerated once with a stricter reminder; a second failure
returns the deterministic template explanation and the event is logged and counted.
"""
import logging

from mlkit import narrative
from mlkit.explainer import grounded_answer

log = logging.getLogger("app.grounding")


def check(text: str, context: dict) -> dict:
    return narrative.check_grounding(text, context)


def guarded(context: dict, question: str, system_prompt: str, generate, metrics=None) -> dict:
    result = grounded_answer(context, question, system_prompt, generate, max_attempts=2)
    if result["source"] == "template":
        log.warning("explainer fallback", extra={"reason": result["reason"],
                                                 "unsupported": (result["grounding"] or {}).get("unsupported", [])[:10]})
        if metrics is not None:
            metrics.count("explainer_fallback_total", {"reason": "grounding" if result["grounding"] else "provider"})
    elif result["attempts"] > 1 and metrics is not None:
        metrics.count("explainer_regenerated_total", {})
    return result
