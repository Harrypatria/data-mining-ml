# Algorithm choice

| Family | Strengths | Weaknesses | Scale features | Imbalance options | Explain with |
|---|---|---|---|---|---|
| Logistic or linear regression | Strong baseline, calibrated (logistic), coefficients | Misses interactions unless engineered | Yes | `class_weight` | Coefficients after diagnostics, SHAP linear |
| Regularized linear (ridge, lasso, elastic net) | High dimension, collinearity | Same as linear | Yes | `class_weight` | Coefficients, stability selection |
| Decision tree | Readable rules | High variance | No | `class_weight` | The tree itself |
| Random forest | Robust default, few parameters | Larger models, impurity bias | No | `class_weight="balanced_subsample"`, balanced RF | Permutation importance, SHAP tree |
| Gradient boosting (HistGradientBoosting, XGBoost, LightGBM, CatBoost) | Best tabular accuracy, missing values native | Needs tuning, can overfit small data | No | `class_weight`, `scale_pos_weight` | SHAP tree, PDP, ALE |
| SVM | Small to medium n, margins | Scaling, no native probabilities | Yes | `class_weight` | Kernel SHAP, surrogate |
| k-NN | Simple, local | Slow at scale, needs scaling | Yes | Weighted voting | Neighbors as examples |
| Naive Bayes | Text, very fast | Independence assumption | No | Prior adjustment | Feature log-probabilities |
| Neural networks (MLP, PyTorch) | Images, text, sequences, very large data | Data hungry, tuning, opacity | Yes | Focal loss, weighted loss | Integrated gradients, SHAP deep |
| GLMs (Poisson, Tweedie, beta) | Counts, rates, bounded targets | Distribution choice | Yes | n/a | Coefficients as rate ratios |

## Defaults by task

- Tabular classification: logistic regression baseline, then HistGradientBoosting.
- Tabular regression: linear baseline, then gradient boosting; quantile loss for intervals.
- Counts: Poisson GLM baseline; Tweedie for insurance-style zero-inflated amounts.
- Text: TF-IDF plus logistic regression, then fine-tuned transformers only if the gap justifies it.
- Time series: seasonal naive, ETS or ARIMA, then gradient boosting on lags.

## Imbalance-aware estimators

```python
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier

ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=SEED)), ("model", LogisticRegression(max_iter=2000))])
BalancedRandomForestClassifier(n_estimators=300, sampling_strategy="all", replacement=True, bootstrap=False, random_state=SEED)
HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
# XGBoost: XGBClassifier(scale_pos_weight=neg / pos, eval_metric="aucpr", random_state=SEED)
```

## Justify complexity

Move up one row only when cross-validated gains exceed the fold spread and the business metric improves. Record the comparison table.
