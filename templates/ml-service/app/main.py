"""App factory. Run with: uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000"""
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from mlkit.features import NUMERIC

from . import __version__
from .config import Settings, get_settings
from .llm import ExplainerService
from .model_store import ModelStore
from .monitoring import BodySizeLimitMiddleware, Metrics, RateLimitMiddleware, RequestContextMiddleware, configure_logging
from .routes import explain, health, model, predict, require_api_key, whatif
from .schemas import build_models

log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()
    store = ModelStore(settings)
    models = build_models(store.schema, settings.max_batch_rows)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once per worker process, verified by checksum and version guard; never inside a route.
        await run_in_threadpool(store.load)
        if store.ready:
            numeric = [f["name"] for f in store.schema["features"] if f["type"] in ("number", "integer")]
            metrics.drift.set_reference(store.bundle.background, numeric or NUMERIC)
        app.state.explainer = ExplainerService(settings, store.metadata.get("model_purpose", ""),
                                               store.metadata.get("audience", ""), metrics)
        yield

    app = FastAPI(title="Model prediction and explanation API", version=__version__, lifespan=lifespan,
                  docs_url="/docs" if settings.app_env == "development" else None, redoc_url=None)
    app.state.settings, app.state.store, app.state.metrics = settings, store, metrics

    # Added innermost first: CORS ends up outermost so 413 and 429 responses still carry CORS headers.
    app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_minute)
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["content-type", "x-api-key", "x-session-id", "x-request-id"],
                       expose_headers=["x-request-id"])

    secured = [Depends(require_api_key)]
    app.include_router(health.router)
    app.include_router(model.build_router(), dependencies=secured)
    app.include_router(predict.build_router(models), dependencies=secured)
    app.include_router(whatif.build_router(models), dependencies=secured)
    app.include_router(explain.build_router(models), dependencies=secured)
    return app
