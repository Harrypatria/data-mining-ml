---
description: Start and test the FastAPI prediction service for the current model bundle
argument-hint: [models dir]
---

Use the `model-deployment` skill.

1. Verify the bundle: `python "${CLAUDE_PLUGIN_ROOT}/bin/data-mining-ml" verify-model ${ARGUMENTS:-models}`.
2. If `api/` is missing, scaffold it from `${CLAUDE_PLUGIN_ROOT}/templates/ml-service` with prompts from `templates/prompts`.
3. Run `cd api && pytest`; fix failures at the cause.
4. Start locally in the background: `uvicorn app.main:create_app --factory --port 8000` with `MODEL_DIR` and `APP_ENV=development`, then check `/health/ready`, `/v1/model` and one `/v1/predict`.
5. Report endpoints, test results and the docker compose command.
