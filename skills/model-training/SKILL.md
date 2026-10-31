---
name: model-training
description: Model Training and Saving. Train, test and save machine learning models - baselines, algorithm choice, scikit-learn or imbalanced-learn Pipeline, class imbalance handling, cross-validation, threshold on validation, one-time test evaluation, final refit, learning curves, pickle persistence (.pkl or .sav), model bundle with metadata, schema, checksum and background sample, reload parity test, version guard and verify-model. Use for fit a model, train classifier or regressor, build a predictive model, save or export a model, pickle, joblib, or prepare a model for deployment.
---

# Model Training and Saving

Template: `templates/notebooks/train_test_save.ipynb`. Atoms: `mlkit.workflow`, `mlkit.persistence`.

## Protocol

1. **Split once** into train, validation and test with the strategy from data-preparation. Fix and record the seed.
2. **One Pipeline** holds preprocessing, engineered features, in-fold resampling and the estimator. It is the only object saved, so serving cannot drift from training.
3. **Baseline first** (majority, mean, last value, current rule, logistic or linear regression), then candidates (`references/algorithms.md`). Compare with cross-validation on train, reporting mean and spread.
4. **Tune** with cross-validation only (hyperparameter-tuning). **Choose the threshold on validation data** (model-evaluation).
5. **Test once.** Metrics with intervals, the baseline comparison, per-slice results, the positive count. No tuning after this; if it happens, say so and get a fresh holdout.
6. **Final refit policy** (default): refit the chosen configuration on train plus validation, keep the test result as the reported estimate, record that the shipped model saw more data.
7. **Learning curve** (train versus validation score by sample size) to judge whether more data or a simpler model helps.

## Imbalance

Order: stratified splits, class weights, threshold moving, resampling inside an imbalanced-learn Pipeline, imbalance-aware algorithms (balanced random forest, `scale_pos_weight`, focal loss), anomaly framing at extreme rates. Recalibrate after reweighting or resampling. Details: `model-evaluation/references/imbalance-and-metrics.md`.

## Persistence

```python
from mlkit.persistence import parity_check, save_bundle, verify_bundle
bundle_dir = save_bundle(final, "models", X_trval, y_trval, version="1.2.0", task="binary_classification",
                         target="churned", classes=[0, 1], threshold=t, metrics=test_metrics, seed=SEED,
                         feature_info=FEATURE_INFO, model_card=card, feature_dictionary=fd)
assert parity_check(bundle_dir, X_test.head(200), final)["passed"]
```

Writes `models/v1.2.0/`: `model.pkl` (pickle, `HIGHEST_PROTOCOL`; `filename="model.sav"` for the other common name; joblib is an alternative for very large NumPy-heavy models), `model.sha256`, `metadata.json` (version, UTC time, git commit, seed, task, target, classes, threshold, cost matrix, metrics with intervals, positive count, data hash, row count, library versions used by the pipeline), `schema.json` (type, unit, range, default, step, label, description, adjustable per predictor), `background.csv` (no identifiers), optional `model_card.md` and `feature_dictionary.csv`, and `models/CURRENT`.

## Rules

- Custom transformers live in `src/<package>/transformers.py`, never in a notebook, so the API can unpickle them.
- Reload parity in a fresh process: probabilities `numpy.allclose`, labels equal. Also in pytest.
- Version guard: major or minor mismatch raises in production, warns in development.
- Security: unpickling runs code. Load only bundles you produced, from storage you control, after the checksum check. Never load a path from user input or an upload.
- Verify from the shell: `bin/data-mining-ml verify-model models/v1.2.0` (standard library only).

Full protocol and failure modes: `references/training-and-persistence.md`.
