---
name: model-deployment
description: Model Deployment. Serve a trained model in production - load a pickled pipeline safely, FastAPI prediction service (predict, batch, what-if, SHAP explanations, LLM narrative), Pydantic validation from schema.json, Docker multi-stage image, docker-compose with nginx, health checks, API keys, CORS, rate limits, JSON logging, Prometheus metrics, drift monitoring (PSI, KS), batch scoring, ONNX or PMML export and air-gapped installs. Use for deploy, serve, productionize, REST API, MLOps, model serving, monitoring or containerizing a model.
---

# Model Deployment

Scaffold: `templates/ml-service/` (becomes `api/` via `bin/data-mining-ml init`). The pickled pipeline from model-training is the model of record.

## Steps

1. Verify the bundle: `bin/data-mining-ml verify-model models/<version>`.
2. Copy the scaffold, install `src/<package>` into the image, pin `api/requirements.txt` to `metadata.json` `libraries`.
3. Run tests: `cd api && pytest` (schema validation, predict, what-if additivity, explanations, narrative with the provider mocked, grounding, security limits, tamper detection).
4. Run locally: `uvicorn app.main:create_app --factory --port 8000` with `MODEL_DIR=../models APP_ENV=development`.
5. Build and run: `docker compose up --build` (API plus nginx serving the React app; nginx injects the API key).
6. Wire monitoring: scrape `/metrics`; alert on `feature_drift_psi` above 0.2, rising `input_range_violations_total`, and explainer fallbacks.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health/live`, `GET /health/ready` | Liveness; readiness means loaded and verified |
| `GET /v1/model`, `GET /v1/schema` | Version, metrics with intervals, threshold, limits; predictor definitions for the UI |
| `POST /v1/predict`, `POST /v1/predict/batch` | Probability or value, label at the stored threshold, version, request id, extrapolation flags |
| `POST /v1/whatif` | Baseline versus scenario, delta, contributions for both |
| `POST /v1/explain/local`, `POST /v1/explain/counterfactuals` | Shapley contributions; smallest threshold-crossing changes |
| `POST /v1/explain/narrative` | Server-built RTCC context, LLM answer, grounding check, SSE stream |

## Engineering rules

- Load once in the `lifespan` handler; keep it on `app.state`; never load inside a route.
- Inference is blocking: plain `def` routes (FastAPI threadpool) or `run_in_threadpool`.
- Pydantic models generated from `schema.json`; unknown fields rejected; categories enumerated; out-of-range numerics need `allow_extrapolation` and come back flagged.
- Column order and dtypes come from the schema, enforced before `predict`.
- The server recomputes everything it sends to the LLM; clients never send explanation values.
- API key dependency, explicit CORS allowlist without credentials, body size limit, rate limit, no upload endpoint.
- JSON logs with request id and no raw personal data; `/metrics` with request counts, latency, prediction distribution, range violations and PSI drift.
- Docker: multi-stage, slim base, pinned requirements, non-root user, no `--reload`, `HEALTHCHECK`, model directory mounted read-only. Each worker holds its own model copy; size workers to memory.
- Air-gapped: vendored wheels (`pip download` then `--no-index --find-links`), `LLM_PROVIDER=none` makes no external calls and uses template explanations.

Details, batch scoring, ONNX and PMML, Java and .NET: `references/fastapi-serving.md`.
