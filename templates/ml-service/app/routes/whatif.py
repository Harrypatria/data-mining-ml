from fastapi import APIRouter, Depends, Request

from .. import explain
from ..schemas import ApiModels, WhatIfOut
from . import check_ranges, require_ready


def build_router(models: ApiModels) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["what-if"])
    WhatIfRequest = models.WhatIfRequest

    @router.post("/whatif", response_model=WhatIfOut)
    def what_if(body: WhatIfRequest, request: Request, store=Depends(require_ready)):
        baseline = body.baseline.model_dump() if body.baseline else store.median_profile()
        scenario = body.scenario.model_dump()
        base_flags = check_ranges(store, baseline, body.allow_extrapolation)
        scen_flags = check_ranges(store, scenario, body.allow_extrapolation)
        rows = request.app.state.settings.explain_background_rows
        result = explain.what_if(store, baseline, scenario, rows)
        rid = request.state.request_id
        request.app.state.metrics.observe_prediction(result["scenario"]["prediction"], scen_flags)
        return {
            "baseline": store.record(result["baseline"]["prediction"], base_flags, rid),
            "scenario": store.record(result["scenario"]["prediction"], scen_flags, rid),
            "delta": result["delta"],
            "changed": result["changed"],
            "contributions": {"baseline": result["baseline"]["contributions"], "scenario": result["scenario"]["contributions"]},
            "base_value": result["scenario"]["base_value"],
        }

    return router
