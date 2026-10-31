"""Custom scikit-learn transformers. Importable so pickled pipelines load in any process."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def log1p_nonneg(X):
    """log(1 + x) after clipping at zero; a named function so FunctionTransformer pickles cleanly."""
    return np.log1p(np.clip(np.asarray(X, dtype=float), 0, None))


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training fold only."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        self.n_features_in_ = arr.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(names, dtype=object)


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Append ratio columns `numerator / (denominator + offset)` to a DataFrame.

    `ratios` is a tuple of (numerator, denominator, new_name). Stateless apart from the
    input column names, so it is safe to fit anywhere, but it still lives in the pipeline
    so training and serving share one code path.
    """

    def __init__(self, ratios=(("monthly_spend", "usage_hours", "spend_per_hour"),), offset: float = 1.0):
        self.ratios = ratios
        self.offset = offset

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RatioFeatures expects a pandas DataFrame with named columns")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        out = X.copy()
        for num, den, name in self.ratios:
            out[name] = out[num].astype(float) / (out[den].astype(float) + self.offset)
        return out

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(base + [name for _, _, name in self.ratios], dtype=object)
