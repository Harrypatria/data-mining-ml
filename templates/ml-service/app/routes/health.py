from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

router = APIRouter(tags=["health"])


@router.get("/health/live")
def live() -> dict:
    return {"status": "live"}


@router.get("/health/ready")
def ready(request: Request):
    store = request.app.state.store
    if store.ready:
        return {"status": "ready", "model_version": str(store.metadata.get("model_version"))}
    return JSONResponse({"status": "not ready", "reason": store.error or "loading"}, status_code=503)


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics(request: Request) -> str:
    return request.app.state.metrics.render()
