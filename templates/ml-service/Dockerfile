# syntax=docker/dockerfile:1
# Build context: the project root (docker build -f api/Dockerfile .), so src/ and api/ are both visible.
# Air-gapped builds: pip download -r api/requirements.txt -d vendor/ on a connected machine, copy vendor/,
# then build with --build-arg PIP_EXTRA="--no-index --find-links=/build/vendor".

FROM python:3.11-slim AS build
ARG PIP_EXTRA=""
WORKDIR /build
COPY api/requirements.txt api/requirements.txt
COPY pyproject.toml README.md ./
COPY src/ src/
COPY vendor/ vendor/
RUN pip wheel --no-cache-dir ${PIP_EXTRA} --wheel-dir /wheels -r api/requirements.txt . \
 && rm -rf vendor

FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/srv/models APP_ENV=production
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels
WORKDIR /srv
COPY api/app ./app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2).status == 200 else 1)"
# Each worker loads its own copy of the model into memory; size workers to RAM / model size.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
