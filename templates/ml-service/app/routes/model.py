from fastapi import APIRouter, Depends

from ..schemas import ModelInfo, SchemaOut
from . import require_ready


def build_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["model"])

    @router.get("/model", response_model=ModelInfo)
    def model_info(store=Depends(require_ready)):
        m = store.metadata
        return ModelInfo(
            model_version=str(m["model_version"]), created_utc=m["created_utc"], task=m["task"], target=m["target"],
            target_label=m.get("target_label", m["target"]), model_family=m.get("model_family"),
            threshold=m.get("threshold"), threshold_rule=m.get("threshold_rule"),
            metrics={k: v for k, v in (m.get("metrics") or {}).items() if isinstance(v, dict)},
            test_positive_count=m.get("test_positive_count"), limitations=m.get("limitations", []),
            calibration_note=m.get("calibration_note", ""), tested_model_note=m.get("tested_model_note"),
            model_purpose=m.get("model_purpose", ""))

    @router.get("/schema", response_model=SchemaOut)
    def schema(store=Depends(require_ready)):
        return store.schema

    return router
