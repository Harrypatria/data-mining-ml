# Training, testing and persistence

## Protocol checklist

| Step | Evidence in the notebook |
|---|---|
| Split | Row and positive counts per split; seed printed |
| Pipeline | `pipeline` repr shows every step; no `fit` outside it |
| Baseline | Dummy or rule metric in the comparison table |
| CV | Mean and standard deviation per candidate |
| Threshold | Chosen on validation, rule written (`choose_threshold`) |
| Test | `metric_with_intervals` table, positive count, slices |
| Refit | Policy sentence and `tested_model_note` in metadata |
| Learning curve | Plot plus one-line conclusion |
| Bundle | `verify_bundle` returns no problems |
| Parity | Fresh-process `parity_check` passes |

## Bundle layout

```
models/
  CURRENT                 v1.2.0
  v1.2.0/
    model.pkl             fitted Pipeline (pickle protocol 5)
    model.sha256          "<hex>  model.pkl"
    metadata.json         see below
    schema.json           predictor definitions for API validation and UI controls
    background.csv        100 to 500 rows sampled from training data, identifiers removed
    model_card.md         intended use, data, metrics by slice, limits, fairness, monitoring
    feature_dictionary.csv
```

`metadata.json` keys: `model_version`, `created_utc`, `git_commit`, `seed`, `task`, `target`, `classes`, `threshold`, `cost_matrix`, `metrics` (value, low, high), `positive_count`, `training_rows`, `training_data_sha256`, `model_file`, `model_sha256`, `columns`, `dtypes`, `libraries` (versions the pipeline needs: python, numpy, pandas, scikit-learn, scipy plus any used extras), `environment` (all tracked packages), `model_purpose`, `audience`, `calibration_note`, and optional `target_label`, `limitations`, `threshold_rule`, `global_importance`, `tested_model_note`.

`schema.json` per feature: `name`, `label`, `unit`, `description`, `adjustable`, `type` (`number`, `integer`, `category`, `boolean`), `min`, `max`, `default` (median or mode), `step`, `missing_rate`, `allowed` for categories.

## Registry convention

Semver folders. Patch: retrain on new data with the same features. Minor: new features or model family with a compatible schema. Major: schema change that breaks API clients. `CURRENT` names the active version; rollback is editing one line.

## Reload parity

`parity_check` writes the sample to CSV, launches `python -c` with only `mlkit` on the path, loads the bundle through `load_bundle` (checksum and version guard), predicts, and compares with the in-memory pipeline after the same CSV round trip. A failure usually means a transformer defined in `__main__`, a lambda in a `FunctionTransformer`, or dtype drift between training and serving.

## Version guard

At load, recorded `libraries` are compared with installed versions on major and minor. Production raises `VersionMismatchError`; development warns. Pin the API image to `metadata.libraries` (the API `requirements.txt` documents this).

## Pickle security

Unpickling executes arbitrary code. Load only files you produced, from storage the operator controls, after `model.sha256` matches. The API has no upload endpoint and reads `MODEL_DIR` from configuration only. For non-Python consumers or untrusted exchange, export ONNX (skl2onnx) or PMML (sklearn2pmml) and validate predictions against the pickle.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: Can't get attribute 'X' on <module '__main__'>` | Class defined in the notebook | Move to `src/<package>/transformers.py` |
| Parity mismatch on categories | Training used categorical dtype, serving strings | `coerce_frame` from `schema.json` before predict |
| `VersionMismatchError` | Image built with other versions | Rebuild with pinned requirements or retrain |
| Checksum mismatch | File modified or truncated in transfer | Restore from registry; never edit pickles |
