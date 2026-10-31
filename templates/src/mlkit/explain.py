"""Model-agnostic local explanations on raw predictors.

shapley_values computes interventional Shapley values (the quantity SHAP's Exact and
Permutation explainers estimate) by swapping raw columns between the profile and a
background sample, so one-hot columns and engineered ratios are attributed back to the
predictor a user can actually move. Exact enumeration for up to `max_exact` predictors,
antithetic permutation sampling above that.

Caveats that travel with every output: values explain the model, not the world; correlated
predictors share or distort attribution; background choice changes the base value.
"""
from __future__ import annotations

from math import factorial

import numpy as np
import pandas as pd


def _blend(x: pd.DataFrame, background: pd.DataFrame, masks: np.ndarray) -> pd.DataFrame:
    """Rows = every mask x every background row; column j takes the profile value where mask[:, j]."""
    n_bg = len(background)
    rep = np.repeat(masks, n_bg, axis=0)
    out = {}
    for j, col in enumerate(background.columns):
        bg = np.tile(background[col].to_numpy(dtype=object), len(masks))
        out[col] = np.where(rep[:, j], x.iloc[0][col], bg)
    frame = pd.DataFrame(out)
    for col in background.columns:
        try:
            frame[col] = frame[col].astype(background[col].dtype)
        except (TypeError, ValueError):
            continue  # mixed values stay object dtype; the pipeline's encoders accept them
    return frame


def shapley_values(predict_fn, x: pd.DataFrame, background: pd.DataFrame, max_exact: int = 10,
                   n_permutations: int = 32, seed: int = 0) -> tuple[float, np.ndarray, float]:
    """Return (base_value, contributions, prediction). contributions.sum() == prediction - base_value."""
    cols = list(background.columns)
    x = x[cols]
    d = len(cols)
    if d <= max_exact:
        masks = ((np.arange(2 ** d)[:, None] >> np.arange(d)) & 1).astype(bool)
        values = np.asarray(predict_fn(_blend(x, background, masks)), dtype=float).reshape(2 ** d, -1).mean(axis=1)
        weights = np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])
        sizes = masks.sum(axis=1)
        phi = np.zeros(d)
        for i in range(d):
            without = np.flatnonzero(~masks[:, i])
            phi[i] = np.sum(weights[sizes[without]] * (values[without | (1 << i)] - values[without]))
        return float(values[0]), phi, float(values[-1])
    rng = np.random.default_rng(seed)
    phi = np.zeros(d)
    perms = []
    for _ in range(max(1, n_permutations // 2)):
        p = rng.permutation(d)
        perms += [p, p[::-1]]
    masks = []
    for p in perms:
        m = np.zeros(d, dtype=bool)
        masks.append(m.copy())
        for j in p:
            m[j] = True
            masks.append(m.copy())
    values = np.asarray(predict_fn(_blend(x, background, np.array(masks))), dtype=float).reshape(len(masks), -1).mean(axis=1)
    k = 0
    for p in perms:
        prev = values[k]
        for step, j in enumerate(p, start=1):
            phi[j] += values[k + step] - prev
            prev = values[k + step]
        k += d + 1
    phi /= len(perms)
    base, pred = float(values[0]), float(values[d])
    return base, phi, pred


def local_explanation(predict_fn, x: pd.DataFrame, background: pd.DataFrame, **kwargs) -> dict:
    """Contributions sorted by absolute size, with direction labels for plain-language use."""
    base, phi, pred = shapley_values(predict_fn, x, background, **kwargs)
    rows = []
    for col, value in zip(background.columns, phi):
        raw = x.iloc[0][col]
        rows.append({"feature": col, "value": raw.item() if hasattr(raw, "item") else raw,
                     "contribution": float(value),
                     "direction": "raises" if value > 0 else "lowers" if value < 0 else "no effect"})
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return {"base_value": base, "prediction": pred, "contributions": rows}


def sensitivity_curve(predict_fn, x: pd.DataFrame, feature: str, grid) -> pd.DataFrame:
    """Vary one predictor across a grid with everything else fixed (an ICE curve for one profile)."""
    frame = pd.concat([x] * len(grid), ignore_index=True)
    frame[feature] = list(grid)
    return pd.DataFrame({"value": list(grid), "prediction": np.asarray(predict_fn(frame), dtype=float)})


def feature_grid(feat: dict, points: int = 25) -> list:
    if feat["type"] == "category":
        return list(feat["allowed"])
    if feat["type"] == "boolean":
        return [False, True]
    grid = np.linspace(feat["min"], feat["max"], points)
    if feat["type"] == "integer":
        return sorted({int(round(v)) for v in grid})
    digits = max(0, int(-np.floor(np.log10(feat["step"])))) if feat.get("step") else 4
    return sorted({round(float(v), digits) for v in grid})  # values a slider can actually reach


def counterfactuals(predict_fn, x: pd.DataFrame, schema: dict, threshold: float, max_results: int = 3,
                    points: int = 25) -> list[dict]:
    """Smallest single-predictor changes, within training ranges, that move the prediction across the threshold.

    Distance is the change divided by the predictor's training range (categorical and boolean
    changes count as 1). Searches adjustable predictors only. Returns an empty list when no
    single change inside the training range crosses the threshold.
    """
    current = float(np.asarray(predict_fn(x), dtype=float)[0])
    above = current >= threshold
    found = []
    for feat in schema["features"]:
        if not feat.get("adjustable", True):
            continue
        name = feat["name"]
        grid = [v for v in feature_grid(feat, points) if v != x.iloc[0][name]]
        if not grid:
            continue
        curve = sensitivity_curve(predict_fn, x, name, grid)
        crossing = curve[(curve["prediction"] >= threshold) != above]
        if crossing.empty:
            continue
        if feat["type"] in ("number", "integer"):
            span = max(feat["max"] - feat["min"], 1e-12)
            crossing = crossing.assign(distance=(crossing["value"].astype(float) - float(x.iloc[0][name])).abs() / span)
        else:
            crossing = crossing.assign(distance=1.0)
        best = crossing.sort_values("distance").iloc[0]
        new_value = grid[int(crossing.sort_values("distance").index[0])]  # keep the grid's own type (int, str, bool)
        found.append({"feature": name, "from": x.iloc[0][name].item() if hasattr(x.iloc[0][name], "item") else x.iloc[0][name],
                      "to": new_value.item() if hasattr(new_value, "item") else new_value,
                      "prediction": float(best["prediction"]), "distance": float(best["distance"])})
    found.sort(key=lambda r: r["distance"])
    return found[:max_results]
