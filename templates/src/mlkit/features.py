"""Feature definitions, preprocessing and the feature dictionary for the simulated churn data."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .transformers import RatioFeatures, Winsorizer

TARGET = "churned"
ID_COLUMNS = ["customer_id"]
NUMERIC = ["tenure_months", "monthly_spend", "support_calls", "usage_hours"]
CATEGORICAL = ["plan", "region"]
BOOLEAN = ["autopay"]
ENGINEERED = ["spend_per_hour"]
PREDICTORS = NUMERIC + CATEGORICAL + BOOLEAN

# One record per predictor: drives schema.json, the React sliders, tooltips and the feature dictionary.
FEATURE_INFO: dict[str, dict] = {
    "tenure_months": {"label": "Tenure", "unit": "months", "source": "billing system",
                      "description": "Months since the customer's first paid invoice.",
                      "transform": "median impute, winsorize 1-99 percent, standardize",
                      "rationale": "Longer relationships usually mean higher switching costs.",
                      "available_at": "snapshot date"},
    "monthly_spend": {"label": "Monthly spend", "unit": "USD", "source": "billing system",
                      "description": "Average invoice amount over the last three months.",
                      "transform": "median impute, winsorize, standardize; also numerator of spend_per_hour",
                      "rationale": "Price sensitivity differs by spend level.",
                      "available_at": "snapshot date"},
    "support_calls": {"label": "Support calls", "unit": "calls in 90 days", "source": "support desk",
                      "description": "Tickets opened in the 90 days before the snapshot.",
                      "transform": "median impute, winsorize, standardize",
                      "rationale": "Repeated contact signals unresolved problems.",
                      "available_at": "snapshot date"},
    "usage_hours": {"label": "Usage", "unit": "hours per month", "source": "product telemetry",
                    "description": "Hours of active product use in the last month.",
                    "transform": "median impute with missing indicator, winsorize, standardize",
                    "rationale": "Low engagement precedes cancellation.",
                    "available_at": "snapshot date"},
    "plan": {"label": "Plan", "unit": "", "source": "billing system",
             "description": "Subscription tier at the snapshot date.",
             "transform": "one-hot, unknown categories ignored",
             "rationale": "Tiers differ in features and contract terms.",
             "available_at": "snapshot date"},
    "region": {"label": "Region", "unit": "", "source": "CRM",
               "description": "Sales region of the billing address.",
               "transform": "one-hot, unknown categories ignored",
               "rationale": "Competition and pricing vary by region.",
               "available_at": "snapshot date"},
    "autopay": {"label": "Autopay", "unit": "", "source": "billing system",
                "description": "Whether invoices are paid automatically.",
                "transform": "passthrough as 0 or 1",
                "rationale": "Autopay removes a monthly cancellation moment.",
                "available_at": "snapshot date"},
    "spend_per_hour": {"label": "Spend per usage hour", "unit": "USD per hour", "source": "engineered",
                       "description": "monthly_spend / (usage_hours + 1).",
                       "transform": "computed in RatioFeatures inside the pipeline",
                       "rationale": "Value for money: high spend with low use suggests poor fit.",
                       "available_at": "snapshot date"},
}


def build_preprocessor(numeric=None, categorical=None, boolean=None, winsorize: bool = True,
                       scale: bool = True) -> ColumnTransformer:
    """ColumnTransformer used in training and serving. Every statistic is fitted on training folds only."""
    numeric = NUMERIC + ENGINEERED if numeric is None else numeric
    categorical = CATEGORICAL if categorical is None else categorical
    boolean = BOOLEAN if boolean is None else boolean
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if winsorize:
        num_steps.append(("clip", Winsorizer(0.01, 0.99)))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(num_steps), numeric),
             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical)]
    if boolean:
        parts.append(("bool", "passthrough", boolean))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    """Ratio features, preprocessing and the estimator as one object: the only artifact that gets saved."""
    return Pipeline([("ratios", RatioFeatures()), ("pre", build_preprocessor(scale=scale)), ("model", estimator)])


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[PREDICTORS].copy(), df[TARGET].copy()


def feature_dictionary(importance: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature dictionary: name, meaning, source, transform, type, rationale, availability, importance, direction.

    `importance` is optional with columns feature, importance_mean, importance_low, importance_high, direction.
    """
    rows = []
    for name, info in FEATURE_INFO.items():
        kind = "numeric" if name in NUMERIC + ENGINEERED else "categorical" if name in CATEGORICAL else "boolean"
        rows.append({"name": name, "meaning": info["description"], "source": info["source"],
                     "transform": info["transform"], "type": kind, "unit": info["unit"],
                     "rationale": info["rationale"], "available_at": info["available_at"]})
    out = pd.DataFrame(rows)
    if importance is not None:
        out = out.merge(importance.rename(columns={"feature": "name"}), on="name", how="left", validate="one_to_one")
    return out
