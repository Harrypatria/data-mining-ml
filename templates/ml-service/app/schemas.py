"""Pydantic v2 request and response models. Profile fields are generated from schema.json at startup."""
from dataclasses import dataclass
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

Scalar = Union[float, int, str, bool, None]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatTurn(Strict):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class Interval(BaseModel):
    value: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None


class PredictionOut(BaseModel):
    probability: Optional[float] = Field(None, description="Predicted probability of the positive class")
    value: Optional[float] = Field(None, description="Predicted value for regression models")
    label: Optional[str] = Field(None, description="Decision at the stored threshold")
    threshold: Optional[float] = None
    model_version: str
    request_id: str
    extrapolation: dict[str, bool] = Field(description="True for each predictor outside its training range")
    extrapolated: bool


class BatchOut(BaseModel):
    predictions: list[PredictionOut]


class Contribution(BaseModel):
    feature: str
    value: Scalar
    contribution: float
    direction: Literal["raises", "lowers", "no effect"]


class ExplainLocalOut(BaseModel):
    prediction: float
    base_value: float = Field(description="Average prediction over the background sample")
    contributions: list[Contribution]
    request_id: str
    note: str = "Contributions describe the model's associations, not causes. Correlated predictors share credit."


class ChangedPredictor(BaseModel):
    name: str
    old: Scalar
    new: Scalar


class WhatIfOut(BaseModel):
    baseline: PredictionOut
    scenario: PredictionOut
    delta: float = Field(description="Scenario minus baseline, in probability (classification) or target units")
    changed: list[ChangedPredictor]
    contributions: dict[str, list[Contribution]]
    base_value: float


class ModelInfo(BaseModel):
    model_version: str
    created_utc: str
    task: str
    target: str
    target_label: str
    model_family: Optional[str] = None
    threshold: Optional[float]
    threshold_rule: Optional[str] = None
    metrics: dict[str, Interval]
    test_positive_count: Optional[int] = None
    limitations: list[str]
    calibration_note: str
    tested_model_note: Optional[str] = None
    model_purpose: str


class FeatureOut(BaseModel):
    name: str
    label: str
    unit: str = ""
    description: str = ""
    type: Literal["number", "integer", "category", "boolean"]
    adjustable: bool = True
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    default: Scalar
    allowed: Optional[list[str]] = None


class SchemaOut(BaseModel):
    columns: list[str]
    features: list[FeatureOut]


class ErrorOut(BaseModel):
    detail: Union[str, list, dict]


@dataclass
class ApiModels:
    Profile: type[BaseModel]
    PredictRequest: type[BaseModel]
    BatchRequest: type[BaseModel]
    WhatIfRequest: type[BaseModel]
    ExplainRequest: type[BaseModel]
    NarrativeRequest: type[BaseModel]


def _field(feat: dict):
    desc = f"{feat.get('label', feat['name'])}" + (f" ({feat['unit']})" if feat.get("unit") else "")
    if feat["type"] == "number":
        return (float, Field(description=desc, json_schema_extra={"x-min": feat["min"], "x-max": feat["max"]}))
    if feat["type"] == "integer":
        return (int, Field(description=desc, json_schema_extra={"x-min": feat["min"], "x-max": feat["max"]}))
    if feat["type"] == "boolean":
        return (bool, Field(description=desc))
    return (Literal[tuple(feat["allowed"])], Field(description=desc))


def build_models(schema: dict, max_batch_rows: int) -> ApiModels:
    """Request models from schema.json: unknown fields are rejected, categories are enumerated."""
    profile = create_model("Profile", __config__=ConfigDict(extra="forbid"),
                           **{f["name"]: _field(f) for f in schema["features"]})
    opt = Optional[profile]
    extrap = (bool, Field(False, description="Accept numeric values outside the training range and flag them"))
    predict = create_model("PredictRequest", __config__=ConfigDict(extra="forbid"), profile=(profile, ...), allow_extrapolation=extrap)
    batch = create_model("BatchRequest", __config__=ConfigDict(extra="forbid"),
                         profiles=(list[profile], Field(min_length=1, max_length=max_batch_rows)), allow_extrapolation=extrap)
    whatif = create_model("WhatIfRequest", __config__=ConfigDict(extra="forbid"),
                          baseline=(opt, Field(None, description="Defaults to the training median profile")),
                          scenario=(profile, ...), allow_extrapolation=extrap)
    explain = create_model("ExplainRequest", __config__=ConfigDict(extra="forbid"), profile=(profile, ...), allow_extrapolation=extrap)
    narrative = create_model("NarrativeRequest", __config__=ConfigDict(extra="forbid"),
                             baseline=(opt, Field(None)), scenario=(profile, ...),
                             question=(str, Field("Explain this prediction.", min_length=1, max_length=1000)),
                             history=(list[ChatTurn], Field(default_factory=list, max_length=6)),
                             allow_extrapolation=extrap)
    return ApiModels(profile, predict, batch, whatif, explain, narrative)
