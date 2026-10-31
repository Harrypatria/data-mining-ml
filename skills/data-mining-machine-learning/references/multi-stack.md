# Multi-stack map

## Python (default)

pandas or polars, scikit-learn, imbalanced-learn, statsmodels, scipy, SHAP, Optuna; XGBoost, LightGBM or PyTorch when justified by a cross-validated gain.

## R

| Task | Package | Equivalent |
|---|---|---|
| Wrangling | dplyr, tidyr | pandas |
| Modeling pipeline | tidymodels (recipes, parsnip, workflows) | scikit-learn Pipeline |
| Resampling | rsample (`initial_split(strata =)`, `group_vfold_cv`, `rolling_origin`) | train_test_split, GroupKFold, TimeSeriesSplit |
| Tuning | tune (`tune_grid`, `tune_bayes`) | GridSearchCV, Optuna |
| Metrics | yardstick (`pr_auc`, `mcc`, `roc_auc`, `brier_class`) | sklearn.metrics |
| Imbalance | themis (`step_smote`, `step_downsample`) inside the recipe | imbalanced-learn Pipeline |
| Explainability | DALEX, vip, iml | permutation importance, SHAP |
| Persistence | `saveRDS` plus `vetiver` for versioned deployment | pickle bundle |

```r
rec <- recipe(churned ~ ., data = train) |> step_impute_median(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors()) |> step_dummy(all_nominal_predictors()) |> themis::step_smote(churned)
wf <- workflow() |> add_recipe(rec) |> add_model(logistic_reg())
res <- fit_resamples(wf, vfold_cv(train, v = 5, strata = churned), metrics = metric_set(pr_auc, roc_auc, mcc))
```

## SQL and warehouses

Push aggregation down: entity-level features with `GROUP BY`, lags with window functions (`LAG(x, 1) OVER (PARTITION BY id ORDER BY ts)`), rolling sums with `ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING`. Pull only the feature table. Snapshot features as of the prediction date to avoid leakage. Never `SELECT *` raw event tables into memory.

## Large data

- DuckDB: SQL over Parquet on a laptop; `duckdb.sql(...).df()` for the final frame.
- Polars lazy: `pl.scan_parquet(...).filter(...).group_by(...).agg(...).collect(streaming=True)`.
- Spark: feature engineering at scale; MLlib for distributed training; sample to pandas for model development when the signal allows.

## On-premises, air-gapped, .NET and Java

- Export to ONNX (`skl2onnx`) and score with ONNX Runtime in .NET or Java; or PMML (`sklearn2pmml`) with JPMML in Spring Boot.
- Wrap scoring in a REST endpoint or batch job in the host stack; validate parity against the Python pickle.
- Pin dependencies, vendor wheels, avoid runtime calls to external services (`LLM_PROVIDER=none`).

## Serving choice

| Consumer | Format |
|---|---|
| Python service | Pickle bundle plus FastAPI (default) |
| Batch in Python | Pickle bundle plus scheduled job |
| Java or .NET | ONNX or PMML |
| Warehouse | SQL scoring for linear models and trees, or UDF with ONNX |
