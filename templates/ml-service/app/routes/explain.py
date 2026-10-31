import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from mlkit.explainer import build_rtcc_context

from .. import explain
from ..llm import ContextInvalid, SessionLimitExceeded
from ..schemas import ApiModels, ExplainLocalOut
from . import check_ranges, require_ready

CHUNK = 48  # characters per SSE token event


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def build_router(models: ApiModels) -> APIRouter:
    router = APIRouter(prefix="/v1/explain", tags=["explain"])
    ExplainRequest, NarrativeRequest = models.ExplainRequest, models.NarrativeRequest

    @router.post("/local", response_model=ExplainLocalOut)
    def local(body: ExplainRequest, request: Request, store=Depends(require_ready)):
        profile = body.profile.model_dump()
        check_ranges(store, profile, body.allow_extrapolation)
        result = explain.local(store, profile, request.app.state.settings.explain_background_rows)
        return {**result, "request_id": request.state.request_id}

    @router.post("/counterfactuals")
    def counterfactuals(body: ExplainRequest, request: Request, store=Depends(require_ready)) -> dict:
        """Smallest single-predictor changes inside the training range that cross the decision threshold."""
        profile = body.profile.model_dump()
        check_ranges(store, profile, body.allow_extrapolation)
        return {"suggestions": explain.nearest_changes(store, profile), "request_id": request.state.request_id}

    @router.post("/narrative", response_class=StreamingResponse,
                 responses={200: {"content": {"text/event-stream": {}},
                                  "description": "Server-sent events: meta, token (repeated), done; or error"}})
    def narrative(body: NarrativeRequest, request: Request, store=Depends(require_ready)):
        """Recompute predictions and contributions server-side, ask the LLM, verify every number, stream the result."""
        baseline = body.baseline.model_dump() if body.baseline else store.median_profile()
        scenario = body.scenario.model_dump()
        check_ranges(store, baseline, body.allow_extrapolation)
        check_ranges(store, scenario, body.allow_extrapolation)
        service = request.app.state.explainer
        session_id = request.headers.get("x-session-id", "")[:64] or (request.client.host if request.client else "anonymous")
        if service.enabled:
            try:
                service.check_session(session_id)
            except SessionLimitExceeded as exc:
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
        history = [t.model_dump() for t in body.history]
        context = build_rtcc_context(store.bundle, baseline, scenario, history=history,
                                     background_rows=request.app.state.settings.explain_background_rows)
        rid = request.state.request_id
        meta = store.metadata

        def events():
            yield sse("meta", {"request_id": rid, "model_version": str(meta.get("model_version")),
                               "provider": request.app.state.settings.llm_provider if service.enabled else "template",
                               **context["predictions"]})
            try:
                result = service.answer(context, body.question, session_id)
            except ContextInvalid as exc:
                yield sse("error", {"code": "context_invalid", "detail": str(exc)})
                return
            text = result["text"]
            for i in range(0, len(text), CHUNK):
                yield sse("token", {"text": text[i:i + CHUNK]})
            yield sse("done", {"source": result["source"], "grounded": result["source"] == "llm",
                               "reason": result["reason"], "cached": result.get("cached", False),
                               "unsupported": (result.get("grounding") or {}).get("unsupported", [])})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router
