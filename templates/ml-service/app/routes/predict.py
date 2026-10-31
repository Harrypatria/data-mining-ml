from fastapi import APIRouter, Depends, Request

from ..schemas import ApiModels, BatchOut, PredictionOut
from . import check_ranges, require_ready


def build_router(models: ApiModels) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["predict"])
    PredictRequest, BatchRequest = models.PredictRequest, models.BatchRequest

    @router.post("/predict", response_model=PredictionOut)
    def predict(body: PredictRequest, request: Request, store=Depends(require_ready)):
        profile = body.profile.model_dump()
        flags = check_ranges(store, profile, body.allow_extrapolation)
        value = float(store.predict([profile])[0])
        request.app.state.metrics.observe_prediction(value, flags)
        request.app.state.metrics.drift.observe(profile)
        return store.record(value, flags, request.state.request_id)

    @router.post("/predict/batch", response_model=BatchOut)
    def predict_batch(body: BatchRequest, request: Request, store=Depends(require_ready)):
        profiles = [p.model_dump() for p in body.profiles]
        flags = [check_ranges(store, p, body.allow_extrapolation) for p in profiles]
        values = store.predict(profiles)
        rid = request.state.request_id
        for p, v, f in zip(profiles, values, flags):
            request.app.state.metrics.observe_prediction(float(v), f)
            request.app.state.metrics.drift.observe(p)
        return {"predictions": [store.record(float(v), f, f"{rid}-{i}") for i, (v, f) in enumerate(zip(values, flags))]}

    return router
