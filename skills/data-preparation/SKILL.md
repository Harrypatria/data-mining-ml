---
name: data-preparation
description: Data Preparation. Clean, join and split data without leakage - data cleaning, joins and merges with row-count checks, deduplication, missing value imputation, type fixes, outlier handling, train test split, validation split, cross-validation folds, stratified, group-aware and time-series (forward-chaining) splits, scikit-learn Pipeline and ColumnTransformer. Use for data wrangling, preprocessing, data cleaning, ETL for machine learning, or any request to prepare data for a model. CRISP-DM Data Preparation.
---

# Data Preparation

Every transformation is either a deterministic cleaning step (applied once, before splitting, with no learned statistics) or a learned step (fitted inside the pipeline on training folds only). Mixing them up is the most common leak.

## Order of work

1. **Deterministic cleaning, before the split.** Fix types, parse dates, standardize units and category spellings, remove exact duplicates, drop rows with an invalid target, remove post-event columns. Print the row count after each step.
2. **Joins with assertions.**
   ```python
   n = len(left)
   out = left.merge(right, on="customer_id", how="left", validate="many_to_one")
   assert len(out) == n, f"join changed rows: {n} -> {len(out)}"
   ```
3. **Choose the split** from the data (`references/split-strategies.md`):
   - Independent rows, classification: stratified.
   - Repeated entities: group-aware (`GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit`).
   - Time-ordered data: forward-chaining (`TimeSeriesSplit` or a cutoff date), with a gap equal to the label horizon.
   Fix and record the seed. Hold out the test set now; it is touched once, at evaluation.
4. **Learned steps inside the pipeline.** Imputers, scalers, encoders, winsorizers, selectors, resamplers and target encoders go into a `ColumnTransformer` and `Pipeline` (or an imbalanced-learn `Pipeline` when resampling). Fit happens on training folds only, through cross-validation or one `fit` on the training split.
5. **Document** every transformation in the feature dictionary: what, why, and when the input is available.

## Missing values

| Mechanism | Default |
|---|---|
| MCAR, small share | `SimpleImputer(strategy="median")` |
| MAR | `IterativeImputer` or `KNNImputer` inside the pipeline |
| MNAR or informative | Imputer with `add_indicator=True`; state the assumption |
| Inference (statistics, not prediction) | Multiple imputation, pooled estimates |

Never fill with a statistic computed on the full frame before splitting.

## Outliers

Decide error or signal first. Errors are fixed or removed with a written rule and a count. Signal is kept and handled with robust scaling, winsorizing inside the pipeline, or robust losses.

## Checks before modeling

- `X_train.shape`, `X_test.shape`, positive counts per split, no shared entity ids across splits.
- `df.duplicated().sum()` before splitting is zero or explained.
- No column recorded after the prediction time.
- The pipeline object prints cleanly and `pipeline.fit(X_train, y_train)` runs.

Conditions that change preparation (skew, zero inflation, high cardinality, drift): `data-profiling/references/scenarios.md`.
