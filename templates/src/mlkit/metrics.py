"""Metric atoms: confusion-matrix metrics, intervals, cost curves and threshold selection."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             cohen_kappa_score, f1_score, fbeta_score, log_loss, matthews_corrcoef,
                             roc_auc_score)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {"tp": int(((y_true == 1) & (y_pred == 1)).sum()), "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()), "tn": int(((y_true == 0) & (y_pred == 0)).sum())}


def _ratio(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def classification_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    """Every metric from the playbook at one threshold, plus the positive count behind them."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    both = len(np.unique(y_true)) == 2
    clipped = np.clip(proba, 1e-6, 1 - 1e-6)
    return {
        "threshold": float(threshold),
        "positives": int(y_true.sum()), "n": int(len(y_true)), "prevalence": float(y_true.mean()),
        **c,
        "accuracy": _ratio(c["tp"] + c["tn"], len(y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else float("nan"),
        "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
        "recall": _ratio(c["tp"], c["tp"] + c["fn"]),
        "specificity": _ratio(c["tn"], c["tn"] + c["fp"]),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if both else float("nan"),
        "kappa": float(cohen_kappa_score(y_true, y_pred)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if both else float("nan"),
        "brier": float(brier_score_loss(y_true, clipped)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
    }


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion such as recall (TP out of all positives)."""
    if trials == 0:
        return float("nan"), float("nan")
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(metric, y_true, y_score, n_boot: int = 500, seed: int = 0, alpha: float = 0.05,
                 stratified: bool = True) -> tuple[float, float, float]:
    """Point estimate and percentile interval of metric(y_true, y_score) over bootstrap resamples.

    Stratified resampling keeps the positive count fixed so rare-class metrics stay defined.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    point = float(metric(y_true, y_score))
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true != 1)
    stats = []
    for _ in range(n_boot):
        if stratified and len(pos) and len(neg):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        else:
            idx = rng.integers(0, len(y_true), len(y_true))
        stats.append(metric(y_true[idx], y_score[idx]))
    low, high = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high)


def expected_cost(y_true, proba, threshold: float, cost_fp: float, cost_fn: float, value_tp: float = 0.0) -> float:
    """Average cost per case at a threshold: FP x cost_fp + FN x cost_fn - TP x value_tp, divided by n."""
    c = confusion_counts(y_true, np.asarray(proba) >= threshold)
    return (c["fp"] * cost_fp + c["fn"] * cost_fn - c["tp"] * value_tp) / max(len(np.asarray(y_true)), 1)


def threshold_table(y_true, proba, cost_fp: float = 1.0, cost_fn: float = 1.0, value_tp: float = 0.0,
                    thresholds=None) -> pd.DataFrame:
    """Precision, recall, F1 and expected cost for a grid of thresholds."""
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2) if thresholds is None else np.asarray(thresholds)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for t in thresholds:
        c = confusion_counts(y_true, proba >= t)
        precision, recall = _ratio(c["tp"], c["tp"] + c["fp"]), _ratio(c["tp"], c["tp"] + c["fn"])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        cost = (c["fp"] * cost_fp + c["fn"] * cost_fn - c["tp"] * value_tp) / len(y_true)
        rows.append({"threshold": float(t), "precision": precision, "recall": recall, "f1": f1,
                     "flagged": c["tp"] + c["fp"], "expected_cost": cost, **c})
    return pd.DataFrame(rows)


def choose_threshold(y_val, p_val, cost_fp: float | None = None, cost_fn: float | None = None,
                     value_tp: float = 0.0, min_precision: float | None = None,
                     min_recall: float | None = None) -> dict:
    """Pick the decision threshold on validation data.

    Priority: a precision floor (maximize recall subject to it), a recall floor (maximize
    precision subject to it), otherwise the minimum expected cost from the cost matrix,
    otherwise maximum F1. Returns the rule used so the choice is auditable.
    """
    table = threshold_table(y_val, p_val, cost_fp or 1.0, cost_fn or 1.0, value_tp)
    if min_precision is not None:
        ok = table[table["precision"] >= min_precision]
        if len(ok):
            row = ok.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]
            return {"threshold": float(row.threshold), "rule": f"max recall with precision >= {min_precision}", **row.to_dict()}
    if min_recall is not None:
        ok = table[table["recall"] >= min_recall]
        if len(ok):
            row = ok.sort_values(["precision", "threshold"], ascending=[False, False]).iloc[0]
            return {"threshold": float(row.threshold), "rule": f"max precision with recall >= {min_recall}", **row.to_dict()}
    if cost_fp is not None and cost_fn is not None:
        row = table.sort_values(["expected_cost", "threshold"]).iloc[0]
        return {"threshold": float(row.threshold), "rule": f"min expected cost (FP {cost_fp}, FN {cost_fn})", **row.to_dict()}
    row = table.sort_values(["f1", "threshold"], ascending=[False, True]).iloc[0]
    return {"threshold": float(row.threshold), "rule": "max F1", **row.to_dict()}


def precision_at_k(y_true, score, k: int) -> float:
    order = np.argsort(-np.asarray(score))[:k]
    return float(np.asarray(y_true)[order].mean()) if k else float("nan")


def metric_with_intervals(y_true, proba, threshold: float, n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Headline metrics with 95 percent intervals: Wilson for recall and precision, bootstrap for the rest."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    m = classification_metrics(y_true, proba, threshold)
    rows = []
    rec = wilson_interval(m["tp"], m["tp"] + m["fn"])
    prec = wilson_interval(m["tp"], m["tp"] + m["fp"])
    rows.append(("recall", m["recall"], *rec, "Wilson", m["tp"] + m["fn"]))
    rows.append(("precision", m["precision"], *prec, "Wilson", m["tp"] + m["fp"]))
    scorers = {"pr_auc": average_precision_score, "roc_auc": roc_auc_score,
               "brier": lambda y, p: brier_score_loss(y, p),
               "f1": lambda y, p: f1_score(y, (p >= threshold).astype(int), zero_division=0),
               "mcc": lambda y, p: matthews_corrcoef(y, (p >= threshold).astype(int))}
    for name, fn in scorers.items():
        point, low, high = bootstrap_ci(fn, y_true, proba, n_boot=n_boot, seed=seed)
        rows.append((name, point, low, high, "bootstrap", int(y_true.sum())))
    return pd.DataFrame(rows, columns=["metric", "value", "low", "high", "method", "denominator"])
