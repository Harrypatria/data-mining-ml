---
name: ai-prediction-explainer
description: AI Prediction Explainer. Generative AI chatbot that explains machine learning predictions in plain language - RTCC prompt (Role, Task, Context, Constraints), server-built context JSON with predictions, SHAP contributions and what-if results, OpenAI or Anthropic provider adapter with streaming, timeouts, retries, caching and rate limits, numeric grounding guard against invented numbers, prompt-injection handling, template fallback, notebook demo. Use for LLM explanations of model output, explainable AI chat, "why did the score change", or adding an AI assistant to a prediction app.
---

# AI Prediction Explainer

The user moves predictors, asks why the prediction changed, and gets a short answer that uses only numbers the server computed.

## Flow

1. The front end calls `POST /v1/whatif` as sliders move.
2. "Explain this change" (or auto-explain) sends only the profile values, the question and recent chat turns to `POST /v1/explain/narrative`.
3. The server recomputes predictions, Shapley contributions and counterfactuals (`mlkit.explainer.build_rtcc_context`), validates the context against `rtcc_context.schema.json`, calls the provider, checks every number (`grounding.py`), regenerates once on failure, falls back to the template, and streams the result over SSE (`meta`, `token`, `done`).
4. Follow-ups reuse the same context plus the last four turns.

## Prompt files (`templates/prompts/`)

- `rtcc_system.md`: Role, Task, Context, Constraints. `[MODEL_PURPOSE]` and `[AUDIENCE]` are filled from `metadata.json` (from ANALYSIS_MANIFEST.md) at startup.
- `rtcc_context.schema.json`: the only facts the model may use.
- `rtcc_examples.md`: two worked examples with angle-bracket placeholders only; real numbers never appear in static files.

## Provider adapter

`LLM_PROVIDER` is `openai`, `anthropic` or `none`; keys from `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`; model from `LLM_MODEL`. Every call has a timeout, a token cap and bounded retries with backoff; answers are cached by the hash of context and question; a per-session hourly limit caps cost. With `none` or on any provider failure the deterministic template explanation is returned. At runtime the service calls the provider API directly; Claude Code helps build and test the adapter.

## Guards

- User text goes in the user message only, below the context and labeled as a question; the system prompt is a static file.
- Clients cannot send predictions or contributions (request models reject unknown fields).
- Grounding: every number must equal a context value, its percent form, or the difference of two context values, within the rounding written; small counting integers (1 to 5) are allowed.
- The answer describes associations, never causes, names extrapolated predictors, and gives no medical, legal or financial advice.

## Tests (provider mocked)

Context building and schema validation, prompt injection stays out of the system prompt, grounded answer passes, invented number is regenerated then replaced, timeout falls back after bounded retries, caching, session limit, SSE format.

## Notebook demo

`templates/notebooks/llm_explainer_demo.ipynb`: same prompt and guard driven by ipywidgets; reads the key from `google.colab.userdata` in Colab or environment variables elsewhere; runs offline with the template.

Details: `references/rtcc-explainer.md`.
