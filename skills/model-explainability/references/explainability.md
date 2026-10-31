# Explainability reference

## Method choice

| Model family or data | Global | Local | Watch out |
|---|---|---|---|
| Linear and GLM | Standardized coefficients (after VIF and residual checks), permutation importance | Coefficient times value, SHAP linear | Collinearity flips signs |
| Tree ensembles | Permutation importance, SHAP summary, ALE | SHAP tree waterfall, counterfactuals | Impurity importance bias |
| Any tabular pipeline | Permutation importance on raw columns, `mlkit.explain.shapley_values` | Shapley on raw predictors, ICE, counterfactuals | Background choice changes the base value |
| Neural networks | SHAP deep or gradient, permutation | Integrated gradients | Saturation, baseline choice |
| Text | Top n-gram weights (linear), SHAP text | Token attributions | Tokenization artifacts |
| Images | Occlusion, Grad-CAM | Saliency maps | Visually convincing but unstable |
| Time series | Lag importance via permutation within blocks | SHAP on lag features | Shuffling breaks autocorrelation; permute in blocks |

## Permutation importance with intervals

```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_test, y_test, scoring="average_precision", n_repeats=20, random_state=SEED)
imp = pd.DataFrame({"feature": X_test.columns, "mean": r.importances_mean,
                    "low": np.percentile(r.importances, 2.5, axis=1), "high": np.percentile(r.importances, 97.5, axis=1)})
```

Group correlated features (for example all one-hot columns of one variable, or a ratio with its parts) and permute them together to avoid split credit.

## Shapley values on raw predictors

```python
from mlkit.explain import local_explanation
exp = local_explanation(lambda d: model.predict_proba(d)[:, 1], X_test.iloc[[0]], X_train.sample(100, random_state=SEED))
# exp["base_value"] + sum(contributions) == exp["prediction"]
```

With the `shap` package: `shap.Explainer(model.predict_proba, masker)` for model-agnostic, `shap.TreeExplainer(model[-1])` on the transformed matrix for trees (attributions then refer to transformed columns).

## Partial dependence, ICE and ALE

```python
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(model, X_test, ["tenure_months"], kind="both", subsample=200, random_state=SEED)
```

ALE (accumulated local effects) via the `alibi` or `PyALE` package when features are correlated; PDP extrapolates into impossible combinations.

## Surrogate rules

```python
from sklearn.tree import DecisionTreeClassifier, export_text
surrogate = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train_t, model.predict(X_train_t))
fidelity = (surrogate.predict(X_test_t) == model.predict(X_test_t)).mean()
rules = export_text(surrogate, feature_names=list(names))
```

Write `rules.md` with the fidelity on test data and the share of cases each rule covers.

## insights.json schema

```json
[{"finding": "string", "evidence": {"metric": "string", "value": 0, "low": 0, "high": 0},
  "segment": "string", "confidence": "string", "recommended_action": "string"}]
```

Values are filled from computed results; the example zeros above mark types only.

## Model card sections

Intended use and users; out-of-scope uses; training data and period; evaluation data; metrics with intervals overall and by slice; threshold and rule; fairness definition and results; limitations; monitoring plan with drift metric, threshold, owner and retraining trigger; version and contact.
