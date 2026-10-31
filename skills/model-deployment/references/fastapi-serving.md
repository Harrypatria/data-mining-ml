# FastAPI serving reference

## Layout

```
api/
  app/
    main.py          create_app(): lifespan loader, CORS, middleware, routers
    config.py        pydantic-settings: MODEL_DIR, APP_ENV, LLM_*, CORS_ORIGINS, API_KEY, RATE_LIMIT_PER_MINUTE, MAX_BODY_BYTES
    schemas.py       request models generated from schema.json; response models
    model_store.py   verified load (checksum, version guard), frame coercion, prediction records
    explain.py       Shapley contributions, what-if, counterfactual search within schema bounds
    llm.py           provider adapter (OpenAI, Anthropic, none): streaming, retries, timeout, token cap, cache, session limit
    grounding.py     numeric grounding check, regenerate once, template fallback, logged and counted
    monitoring.py    JSON logs, request ids, size and rate limits, Prometheus metrics, PSI drift
    prompts/         rtcc_system.md, rtcc_context.schema.json, rtcc_examples.md
    routes/          health.py, model.py, predict.py, whatif.py, explain.py
  tests/             pytest with TestClient; the provider is mocked
  scripts/export_openapi.py
  Dockerfile  requirements.txt  requirements-dev.txt  .env.example  pytest.ini
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models` | Version folder or models root with `CURRENT` |
| `APP_ENV` | `production` | `production` raises on version mismatch and hides `/docs` |
| `CORS_ORIGINS` | `http://localhost:5173` | Comma-separated; wildcard refused |
| `API_KEY` | empty | When set, `/v1/*` requires `X-API-Key` |
| `RATE_LIMIT_PER_MINUTE` | 120 | Token bucket per client address |
| `MAX_BODY_BYTES` | 65536 | 413 above this |
| `LLM_PROVIDER` | `none` | `openai`, `anthropic` or `none` |
| `LLM_MODEL` | empty | Provider model name, never hardcoded |
| `LLM_TIMEOUT_S`, `LLM_MAX_TOKENS`, `LLM_MAX_RETRIES` | 20, 400, 2 | Bounded calls |
| `LLM_SESSION_LIMIT_PER_HOUR` | 30 | Per `X-Session-Id` |

## Running

```bash
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 2
gunicorn -k uvicorn.workers.UvicornWorker -w 4 "app.main:create_app()"     # when a process manager is required
```

## Monitoring

| Metric | Alert |
|---|---|
| `http_requests_total{status="5xx"}` | Any sustained rate |
| `request_latency_seconds` | p95 above the SLO |
| `prediction_probability_bin` | Distribution shift versus training |
| `input_range_violations_total` | Rising trend: new population |
| `feature_drift_psi` | Above 0.2 (moderate), 0.25 (major) |
| `explainer_fallback_total` | Provider outage or grounding failures |

Retraining trigger: PSI alert or recall decay on labelled outcomes for two consecutive periods. Owner and runbook live in the model card.

## Batch scoring

```python
from mlkit.persistence import load_bundle, coerce_frame
bundle = load_bundle("models")                       # operator path, checksum verified
for chunk in pd.read_parquet("scoring/input.parquet", columns=bundle.schema["columns"]).pipe(lambda d: np.array_split(d, 20)):
    scores = bundle.pipeline.predict_proba(coerce_frame(chunk, bundle.schema))[:, 1]
```

Write scores with the model version and run timestamp; schedule with cron, Airflow or the warehouse's task runner.

## Non-Python consumers and air-gapped environments

- ONNX: `skl2onnx.to_onnx(pipeline, X_sample)`, serve with ONNX Runtime from Java, .NET (`Microsoft.ML.OnnxRuntime`) or C++. Validate parity against the pickle on a held-out sample.
- PMML: `sklearn2pmml` for scorecards and trees; score with JPMML in Spring Boot.
- Wrap scoring in a REST endpoint or batch job in the host stack; pin dependencies; avoid runtime calls to external services.
- Offline wheels: `pip download -r api/requirements.txt -d vendor/`, build with `PIP_EXTRA="--no-index --find-links=/build/vendor"`.

## Security checklist

- No upload or model-path parameter anywhere in the API.
- `model.sha256` checked before unpickling; read-only model mount.
- Secrets only in environment variables; nginx adds `X-API-Key`; the browser never holds provider keys.
- Logs carry ids, routes, status and timing, never profiles or questions.
