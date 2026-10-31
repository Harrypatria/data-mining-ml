"""Reference train-test-save run shared by the dashboard, story and explainer notebooks and the API tests.

Protocol: stratified train / validation / test split, one pipeline, threshold chosen on
validation, test scored once with intervals, final refit on train plus validation, bundle saved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import metrics as mx
from .data import make_customer_churn
from .features import FEATURE_INFO, TARGET, build_pipeline, feature_dictionary, split_xy
from .persistence import Bundle, load_bundle, resolve_model_dir, save_bundle, verify_bundle

MODEL_PURPOSE = "flagging subscription customers who are likely to cancel within 30 days"
AUDIENCE = "account managers on the retention team"
TARGET_LABEL = "cancelling within 30 days"
LIMITATIONS = [
    "Trained on simulated data; retrain and re-evaluate on real customer records before use.",
    "Scores rank risk; they do not say why a customer would cancel.",
    "Not evaluated for customers with more than 72 months of tenure.",
]


def split_three(X: pd.DataFrame, y: pd.Series, seed: int, test_size: float = 0.2, val_size: float = 0.25):
    """Stratified train / validation / test. val_size is a share of the non-test rows."""
    X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=val_size, stratify=y_trval, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def render_model_card(meta: dict, slices: pd.DataFrame, intervals: pd.DataFrame) -> str:
    """Markdown model card from computed results only."""
    def table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, row in frame.iterrows():
            lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |")
        return "\n".join(lines)

    return "\n".join([
        f"# Model card: {meta['name']} v{meta['version']}",
        "",
        "## Intended use",
        f"Supports {meta['purpose']}. Audience for explanations: {meta['audience']}. Not for automated decisions without review.",
        "",
        "## Data",
        f"Rows used for the final refit: {meta['rows']}. Positive share in training data: {meta['prevalence']:.3f}. Source: {meta['source']}.",
        "",
        "## Decision rule",
        f"Threshold {meta['threshold']:.2f}, chosen on validation data by rule: {meta['rule']}.",
        "",
        "## Test metrics (scored once, 95 percent intervals)",
        table(intervals),
        "",
        "## Metrics by slice (test set, same threshold)",
        table(slices),
        "",
        "## Fairness",
        "Recall and precision are compared across slices above. For people-facing use, repeat this table for each "
        "protected attribute with equal opportunity (recall parity) as the chosen definition, because missed "
        "at-risk customers are the costly error.",
        "",
        "## Limitations",
        *[f"- {item}" for item in meta["limitations"]],
        "",
        "## Monitoring",
        "Weekly PSI per numeric predictor against background.csv (alert above 0.2), monthly recall on labelled "
        "outcomes, retrain when either alert persists for two periods. Owner: the retention analytics team.",
    ])


def slice_metrics(y_true, proba, threshold: float, groups: pd.Series) -> pd.DataFrame:
    rows = []
    for g in sorted(groups.unique()):
        mask = (groups == g).to_numpy()
        m = mx.classification_metrics(np.asarray(y_true)[mask], np.asarray(proba)[mask], threshold)
        rows.append({"slice": f"{groups.name}={g}", "n": m["n"], "positives": m["positives"],
                     "recall": m["recall"], "precision": m["precision"], "pr_auc": m["pr_auc"]})
    return pd.DataFrame(rows)


def train_reference_bundle(models_dir, seed: int = 42, n: int = 6000, prevalence: float = 0.15,
                           version: str = "1.0.0", min_precision: float = 0.35, n_boot: int = 300) -> Path:
    df = make_customer_churn(n, prevalence, seed)
    assert df["customer_id"].is_unique, "one row per customer expected"
    assert not df.drop(columns="customer_id").duplicated().any(), "duplicated rows would leak across splits"
    X, y = split_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_three(X, y, seed)
    candidates = {"logistic": LogisticRegression(max_iter=2000),
                  "gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = {name: float(cross_val_score(build_pipeline(est), X_train, y_train, cv=cv, scoring="average_precision").mean())
                 for name, est in candidates.items()}
    family = max(cv_scores, key=cv_scores.get)
    baseline = float(cross_val_score(DummyClassifier(strategy="prior"), X_train, y_train, cv=cv, scoring="average_precision").mean())
    if cv_scores[family] <= baseline:
        raise RuntimeError(f"no candidate beats the prior baseline (CV PR-AUC {baseline:.3f}); do not ship a model")
    pipe = build_pipeline(clone(candidates[family])).fit(X_train, y_train)
    choice = mx.choose_threshold(y_val, pipe.predict_proba(X_val)[:, 1], min_precision=min_precision)
    threshold = choice["threshold"]
    p_test = pipe.predict_proba(X_test)[:, 1]
    intervals = mx.metric_with_intervals(y_test, p_test, threshold, n_boot=n_boot, seed=seed)
    slices = pd.concat([slice_metrics(y_test, p_test, threshold, X_test["plan"]),
                        slice_metrics(y_test, p_test, threshold, X_test["region"])], ignore_index=True)
    perm = permutation_importance(pipe, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=seed)
    importance = pd.DataFrame({"feature": X_test.columns, "importance_mean": perm.importances_mean,
                               "importance_low": np.percentile(perm.importances, 2.5, axis=1),
                               "importance_high": np.percentile(perm.importances, 97.5, axis=1)})
    X_trval, y_trval = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
    final = build_pipeline(clone(candidates[family])).fit(X_trval, y_trval)
    test_metrics = {r.metric: {"value": r.value, "low": r.low, "high": r.high} for r in intervals.itertuples()}
    card = render_model_card({"name": "churn-risk", "version": version, "purpose": MODEL_PURPOSE, "audience": AUDIENCE,
                              "rows": len(X_trval), "prevalence": float(y_trval.mean()), "source": "mlkit.data.make_customer_churn (simulated)",
                              "threshold": threshold, "rule": choice["rule"], "limitations": LIMITATIONS}, slices, intervals)
    return save_bundle(
        final, models_dir, X_trval, y_trval, version=version, task="binary_classification", target=TARGET,
        classes=[0, 1], threshold=threshold, metrics=test_metrics, seed=seed,
        cost_matrix=None, feature_info=FEATURE_INFO, model_purpose=MODEL_PURPOSE, audience=AUDIENCE,
        calibration_note=f"Probabilities come from {family.replace('_', ' ')} without resampling or class weights; check the calibration curve before reading them as exact rates.",
        model_card=card, feature_dictionary=feature_dictionary(importance),
        extra={"model_family": family, "cv_pr_auc": {**cv_scores, "dummy_prior": baseline}, "target_label": TARGET_LABEL, "limitations": LIMITATIONS, "threshold_rule": choice["rule"],
               "tested_model_note": "Test metrics come from the model fitted on train only; the shipped model was refitted on train plus validation.",
               "global_importance": [{"feature": r.feature, "importance": float(r.importance_mean)} for r in importance.itertuples()],
               "test_positive_count": int(y_test.sum())},
    )


def load_or_train(models_dir, mode: str = "development", **kwargs) -> Bundle:
    """Load models/CURRENT when it verifies; otherwise run the reference training and load the result."""
    models_dir = Path(models_dir)
    if not (models_dir / "CURRENT").is_file() or verify_bundle(resolve_model_dir(models_dir)):
        train_reference_bundle(models_dir, **kwargs)
    return load_bundle(models_dir, mode=mode)
