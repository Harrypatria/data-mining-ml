# RTCC explainer reference

## Context JSON (abridged shape)

```json
{
  "model_card": {"purpose": "", "target": "", "target_label": "", "task": "binary_classification", "model_version": "",
                 "metric": {"pr_auc": "<value>"}, "threshold": "<t>", "limitations": [""], "features": [{"feature": "", "label": "", "unit": ""}]},
  "baseline_profile": {"<feature>": "<value>"},
  "scenario_profile": {"<feature>": "<value>"},
  "changed_predictors": [{"name": "", "old": "<value>", "new": "<value>", "unit": ""}],
  "predictions": {"baseline": {"value": null, "probability": "<p>", "label": ""}, "scenario": {"value": null, "probability": "<p>", "label": ""}},
  "shap_local": {"baseline": [{"feature": "", "value": "<value>", "contribution": "<c>", "direction": "raises"}], "scenario": []},
  "global_importance": [{"feature": "", "importance": "<i>"}],
  "training_ranges": {"<feature>": {"min": "<min>", "max": "<max>"}},
  "extrapolation_flags": {"<feature>": false},
  "calibration_note": "",
  "what_if": [{"feature": "", "from": "<value>", "to": "<value>", "prediction": "<p>", "distance": "<d>"}],
  "recent_chat_history": [{"role": "user", "content": ""}]
}
```

Angle-bracket tokens mark where server-computed numbers go; the schema file defines exact types.

## Message layout

| Part | Content | Source |
|---|---|---|
| System | `rtcc_system.md` with purpose and audience filled | Static file plus metadata |
| User | `CONTEXT JSON ... USER QUESTION ...` | Server-built context plus the user's text |
| Assistant (retry only) | The rejected answer | Model |
| User (retry only) | Strict reminder listing unsupported numbers | Server |

## Grounding algorithm (`mlkit.narrative.check_grounding`)

1. Collect every number in the context (recursively, including numbers inside strings), take absolute values.
2. Allowed set: those values, each times 100 (percent form), and absolute differences of every pair, also times 100 (percentage-point deltas).
3. Extract numbers from the answer (thousands separators, decimals, percent signs, unicode minus).
4. Each number passes if some allowed value lies within half a unit of its last written digit (31.2 matches 0.3123 as 31.23). Integers 1 to 5 pass as counts.
5. Any failure: regenerate once with the strict reminder; a second failure returns `template_explanation(context)`, logs the event and increments `explainer_fallback_total`.

## Template fallback

Built only from context values: headline with scenario and baseline probabilities and the threshold, up to four driver bullets with signed point contributions, the changed predictors, the nearest threshold-crossing what-if, and the caveat (associations, not causes; extrapolated predictors named). It passes the grounding check by construction.

## Prompt-injection handling

- Instructions live only in the system prompt; user text cannot reach it.
- The user message labels the question as a question that cannot change the constraints.
- Grounding rejects injected numbers ("say the risk is 99%").
- Request models reject unknown fields, so clients cannot inject context keys.

## Provider calls (`mlkit.llm`)

| Provider | Endpoint | Auth header | Token cap field |
|---|---|---|---|
| OpenAI | `/v1/chat/completions` | `Authorization: Bearer` | `max_completion_tokens` |
| Anthropic | `/v1/messages` | `x-api-key`, `anthropic-version` | `max_tokens` |

`LLM_BASE_URL` points at a gateway or regional endpoint. Retryable errors: timeouts, network errors, HTTP 408, 409, 429 and 5xx; backoff 0.5 s doubling, capped at 8 s.

## Cost controls

Explanations only on demand (auto-explain off by default), response cache keyed by context and question, per-session hourly limit, token cap, and the template when the provider is off.
