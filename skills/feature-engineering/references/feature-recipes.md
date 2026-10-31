# Feature recipes

All snippets assume `SEED` and run inside a `Pipeline`; custom functions live in `src/<package>/` so pickles load in the API.

## Extraction

| Method | Input | Inspect | Code |
|---|---|---|---|
| PCA | Scaled numerics | Explained variance ratio, loadings | `make_pipeline(StandardScaler(), PCA(n_components=0.95, random_state=SEED))` |
| Truncated SVD | Sparse TF-IDF | Explained variance | `TruncatedSVD(n_components=100, random_state=SEED)` |
| NMF | Non-negative counts | Topic terms per component | `NMF(n_components=10, init="nndsvda", random_state=SEED)` |
| ICA | Mixed signals | Component kurtosis | `FastICA(n_components=k, random_state=SEED)` |
| Kernel PCA | Non-linear structure | Reconstruction error with `fit_inverse_transform=True` | `KernelPCA(kernel="rbf")` |
| Autoencoder bottleneck | High-dimensional dense | Reconstruction error on validation | PyTorch encoder; save the encoder with the pipeline |
| TF-IDF and n-grams | Text | Top terms per class | `TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)` |
| Pretrained embeddings | Text or images | Nearest neighbors sanity check | Compute offline, version the encoder, cache vectors |
| Signal features | Sensor series | Spectrum plot | FFT band energy, rolling mean, std, min, max per window |

## Engineering

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

def date_parts(X: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(X["event_time"])
    return pd.DataFrame({"hour_sin": np.sin(2 * np.pi * d.dt.hour / 24), "hour_cos": np.cos(2 * np.pi * d.dt.hour / 24),
                         "dow": d.dt.dayofweek, "is_month_end": d.dt.is_month_end.astype(int)}, index=X.index)

date_features = FunctionTransformer(date_parts, feature_names_out="one-to-one")  # function lives in src/<package>/features.py
```

| Recipe | Rationale pattern | Leakage guard |
|---|---|---|
| Cyclical encoding (sin, cos) | Hour 23 is close to hour 0 | None needed |
| Lags `groupby(id)[x].shift(k)` | Recent behavior predicts the next period | k at least the label horizon |
| Rolling windows `groupby(id)[x].transform(lambda s: s.shift(1).rolling(28).mean())` | Trend and volatility | Shift before rolling |
| Group aggregations | Entity history | Computed from data before the snapshot |
| Ratios and differences | Value for money, utilization | Denominator offset to avoid division by zero |
| Interactions, polynomials | Known joint effects | Regularize; degree 2 is usually enough |
| Binning (`KBinsDiscretizer`) | Non-linear effects in linear models | Edges fitted on training folds |
| Weight of evidence | Scorecards | Out-of-fold only |
| Frequency encoding | High cardinality | Frequencies from training folds |
| Target encoding (`TargetEncoder`) | High cardinality with signal | Cross-fitted inside the pipeline |
| Geospatial distance, cluster id | Location matters | Clusters fitted on training data |
| Text length, counts | Effort or verbosity signal | None needed |
| Domain flags | Business rules as inputs | Rule uses only prediction-time data |

## Selection inside the pipeline

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
pipe = Pipeline([("pre", preprocessor),
                 ("select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.5))),
                 ("model", HistGradientBoostingClassifier(random_state=SEED))])
```

Stability selection: refit the selector on 50 bootstrap samples of the training data and keep features chosen in at least 70 percent of them.

## Ablation template

```python
groups = {"base": base_cols, "+ratios": base_cols + ratio_cols, "+lags": base_cols + ratio_cols + lag_cols}
rows = []
for name, cols in groups.items():
    s = cross_val_score(make_model(cols), X_train[cols], y_train, cv=cv, scoring="average_precision")
    rows.append({"group": name, "mean": s.mean(), "sd": s.std()})
```
