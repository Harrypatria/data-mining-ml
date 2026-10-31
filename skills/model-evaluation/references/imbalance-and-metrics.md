# Imbalance and metric playbook

`templates/notebooks/imbalance_lab.ipynb` computes every effect below from simulated data at run time. This file explains; the lab shows numbers.

## 1. Confusion matrix first

|  | Predicted positive | Predicted negative |
|---|---|---|
| Actual positive | TP | FN |
| Actual negative | FP | TN |

- Precision = TP / (TP + FP): share of flagged cases that are real.
- Recall (sensitivity, true positive rate) = TP / (TP + FN): share of real cases flagged.
- Specificity = TN / (TN + FP).
- F1 = 2PR / (P + R). F-beta = (1 + b^2)PR / (b^2 P + R); b above 1 weights recall.
- Balanced accuracy = (recall + specificity) / 2.
- MCC = (TP TN - FP FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)); uses all four cells.
- Cohen's kappa: agreement beyond chance.

## 2. The accuracy paradox

A model that predicts the majority class everywhere has accuracy equal to 1 minus prevalence and recall 0. At one percent prevalence that is high accuracy for a useless model. The lab's first table computes this at prevalence 50, 20, 5, 1 and 0.1 percent and puts accuracy, precision, recall, F1, ROC-AUC and PR-AUC side by side for one classifier.

## 3. How metrics behave as prevalence falls

| Metric | Behavior | Why |
|---|---|---|
| Accuracy | Rises toward 1 minus prevalence | Dominated by true negatives |
| Recall | Prevalence-independent in expectation, noisier | Divides by positives only; ignores false alarms |
| Precision | Falls sharply at a fixed false-positive rate | Base-rate effect: FP grows with the negative class |
| ROC-AUC | Stays optimistic | FPR divides by a large negative class |
| PR-AUC | Falls toward the prevalence (no-skill floor) | Built on precision; shows the real difficulty |
| F1 | Falls; threshold-dependent | Ignores TN, inherits precision's fall |
| Log loss, Brier | Expose miscalibration after resampling | Score the probabilities themselves |

## 4. Headline metric by cost

See the table in `SKILL.md`. Write the cost matrix in ANALYSIS_MANIFEST.md and derive the metric from it.

## 5. Threshold as a decision

```python
from mlkit.metrics import choose_threshold, threshold_table
table = threshold_table(y_val, p_val, cost_fp=15, cost_fn=120)
best = choose_threshold(y_val, p_val, cost_fp=15, cost_fn=120)          # or min_precision=0.3
```

Plot precision against recall with the chosen point, and expected cost against threshold. Score the test set once at the chosen threshold.

## 6. Remedies, least to most invasive

| Remedy | Changes | Keeps | Code |
|---|---|---|---|
| Stratified splits and CV | Fold composition | Everything else | `stratify=y`, `StratifiedKFold` |
| Class or sample weights | The loss | Data | `class_weight="balanced"`, `sample_weight` |
| Threshold moving | The decision rule | Model and data | `choose_threshold` |
| Resampling in the training fold | Training data | Validation and test data | `imblearn.pipeline.Pipeline` with `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`, `ADASYN`, `SMOTETomek` |
| Imbalance-aware algorithms and losses | The learner | Data | `BalancedRandomForestClassifier`, `scale_pos_weight`, focal loss |
| Anomaly framing | The problem | | One-class models at extreme rates |
| More minority data, reframing | The data or the question | | Active labeling, longer horizon |

## 7. Calibration after reweighting or resampling

Resampling and class weights shift the learned base rate, so predicted probabilities overstate the minority class. Recalibrate on untouched validation data and report the calibration curve and Brier score before and after:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(X_val, y_val)
```

## 8. Uncertainty on rare classes

Wilson intervals for recall (denominator: positives) and precision (denominator: flagged); bootstrap with stratified resampling for curve metrics. State the positive count next to every rare-class metric.

## 9. Multi-class and multi-label

Macro averages weight classes equally and expose minority failure; micro and weighted averages follow the majority. Report per-class recall. For multi-label, add subset accuracy and Hamming loss.

## 10. Anti-patterns

| Anti-pattern | Scanner rule |
|---|---|
| SMOTE or any resampling before the split | L03 |
| Resampling the test or validation set | L03 |
| Accuracy alone | M02 |
| ROC-AUC alone on rare positives | M03 |
| Default 0.5 threshold | M04 |
| Threshold tuned on the test set | L08 |
| Resampler outside an imbalanced-learn pipeline | M09 |
| No calibration check after resampling | M13 |
| Comparing models at different thresholds without saying so | manual (ml-critic) |
