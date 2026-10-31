---
name: model-evaluation
description: Model Evaluation. Evaluate classifiers and regressors honestly - confusion matrix, precision, recall, F1, F-beta, MCC, balanced accuracy, ROC-AUC versus PR-AUC on imbalanced data, accuracy paradox, threshold selection with a cost matrix, calibration (Platt, isotonic, Brier score), confidence intervals, slice analysis and fairness metrics, regression metrics (MAE, RMSE, MAPE, pinball loss). Use for model metrics, which metric to use, imbalanced classes, choosing a threshold, model comparison, calibration or fairness checks.
---

# Model Evaluation

## Choose the headline metric by cost

| Situation | Lead with |
|---|---|
| Rare, costly misses (fraud, screening) | Recall at a fixed precision, PR-AUC, F-beta with beta above 1 |
| Costly false alarms (review queues) | Precision at k (k = capacity) |
| Balanced costs | MCC or balanced accuracy |
| Ranking or targeting | Lift, cumulative gain, precision at k |
| Probabilities used directly | Log loss, Brier score, calibration curve |
| Regression | MAE (robust), RMSE (penalizes large errors), MAPE only when no zeros, pinball loss for quantiles |

Accuracy alone is banned on imbalanced data. ROC-AUC alone is banned on rare positives; pair it with PR-AUC and its no-skill floor (the prevalence).

## Threshold is a decision

Never use 0.5 on imbalanced data without a reason. Choose on validation data with an explicit cost matrix (`mlkit.metrics.choose_threshold(cost_fp=..., cost_fn=...)`) or a constraint (`min_precision`, `min_recall`). Show the precision-recall trade-off and expected cost per threshold. Compare models at the same operating point, or say they differ.

## Calibration

Check with a calibration curve (quantile bins) and Brier score on validation data. Resampled or reweighted models over-predict the minority probability: recalibrate with `CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")` (or Platt `sigmoid` for small samples) fitted on untouched validation data.

## Uncertainty

Every headline metric gets an interval: Wilson for recall and precision (state the denominator), stratified bootstrap for PR-AUC, ROC-AUC, F1 and MCC (`mlkit.metrics.metric_with_intervals`). State the positive count; a few dozen positives make intervals wide.

## Slices and fairness

Report recall, precision and calibration per important segment with positive counts. For people-facing decisions, pick a fairness definition and say why: equal opportunity (recall parity) when misses harm, predictive parity (precision parity) when false alarms harm, calibration within groups when scores are read as probabilities. Report the gap with intervals.

## Multi-class

Report per-class recall and precision, macro averages (each class equal), and say that micro and weighted averages follow the majority class.

## Output

Comparison table (baseline, candidates, chosen) at the chosen threshold, intervals, calibration plot, slice table, and one sentence on the decision. Playbook and anti-patterns: `references/imbalance-and-metrics.md`. Runnable lab: `templates/notebooks/imbalance_lab.ipynb`.
